# -*- coding: utf-8 -*-
"""
Model v0 -- Oracle / mechanism validation.

Question this answers: the diagnosis says performance tracks the upper tail of
the NORMAL response. That is a correlation. Does *actively suppressing* that
tail actually restore detection on can / wallplugs?

The Oracle is allowed the full train/good set (80% bank / 20% tail-query) -- far
more than 1-shot or 4-shot. It is deliberately NOT the final model; it exists to
test whether the normal tail is a causal lever. If a full-normal oracle cannot
move AUROC, the diagnosis is incomplete and no amount of few-shot engineering
will help.

Ablation (one bank, one split, one seed -- only the adapter/objective differs).
Five configs, not four, so that each comparison moves ONE variable:

  A   baseline               no adapter (frozen DINOv2)
  B0  mean, no preserve      the naive objective -- shows what it degenerates to
  B   mean + preserve        control for the objective
  C   tail, no preserve      tail objective without protection
  D   tail + preserve        the Model v0 configuration

  A  vs D   does calibration help at all?
  B  vs C   does the TAIL objective beat the MEAN one, both preserved?
  C  vs D   does preservation prevent collapse?
  B0 vs B   what the preservation term is actually preventing

Usage:
  python experiments/model_v0/run_oracle.py                 # A/B/C/D, 4 objects
  python experiments/model_v0/run_oracle.py --configs A C   # subset
  python experiments/model_v0/run_oracle.py --objects can --steps 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")

from ad2_pipeline import AD2_ROOT
from pixel_metrics_binned import pixel_metrics_binned
from models import ResidualAdapter
from tail_calib import (CACHE, RESULTS, load_cache, split_train, mean_top1p,
                        train_adapter, patch_dists_all, tail_stats)

PRIMARY_GT = 0.10
OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
MAPS = os.path.join(RESULTS, "maps")
CKPT = os.path.join(RESULTS, "checkpoints")
METRICS = os.path.join(RESULTS, "metrics")

CONFIGS = {
    "A": {"adapter": False, "objective": None, "preserve": False},
    "B0": {"adapter": True, "objective": "mean", "preserve": False},
    "B": {"adapter": True, "objective": "mean", "preserve": True},
    "C": {"adapter": True, "objective": "tail", "preserve": False},
    "D": {"adapter": True, "objective": "tail", "preserve": True},
    # E is the DEFECT-SUPERVISED probe, not a candidate model. It answers
    # whether the normal tail can be suppressed *at all* without destroying
    # defect evidence. Real few-shot detection has no defect labels, so if E
    # also fails, the lever is dead; if E succeeds, the real problem is
    # estimating the preserve-direction without defect samples.
    "E": {"adapter": True, "objective": "tail", "preserve": True,
          "defect_sup": True},
    # F* are Proxy-E: identical adapter, tail objective and separation
    # objective to E -- only the SOURCE of the "defect" patches changes, from
    # real test_public defects to pseudo-defects built from train/good alone.
    # No test data is touched, so F is evaluated on the full bad set (or on the
    # half under --eval-half, to line up with E).
    "F1": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "texture"},
    "F2": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "structural"},
    "F3": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "feature"},
    # F4/F5 = F3 with the pseudo-defect set forced to span k dimensions. Only
    # the rank changes, so they isolate "is the missing ingredient coherence?"
    # from "is the missing ingredient semantics?".
    "F4": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "lowrank5"},
    "F5": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "lowrank20"},
}
CONFIG_ORDER = ["A", "B0", "B", "C", "D", "E", "F1", "F2", "F3", "F4", "F5"]
# G1/G2 bound how TRANSFERABLE the defect signal is, which is the necessary
# condition for any CLIP/VLM text prior to work. They use real test_public
# defects, so they are oracles and must be evaluated on the held-out half.
CONFIGS.update({
    "G1": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "crossdir", "real": True},
    "G2": {"adapter": True, "objective": "tail", "preserve": True,
           "defect_sup": True, "proxy": "selfdir", "real": True},
})
CONFIG_ORDER += ["G1", "G2"]
# ---- Phase 3C: how much of the REAL defect set does E actually need? --------
# Rank scan flattens the real defect cloud onto its own top-k axes; prototype
# scan replaces each patch by one of K k-means centroids. Both keep the defect
# count (so the loss weight is unchanged) and use the same split as E.
for _k in (1, 3, 5, 10, 20, 40, 70, 120):
    CONFIGS[f"R{_k}"] = {"adapter": True, "objective": "tail", "preserve": True,
                         "defect_sup": True, "proxy": f"rank{_k}", "real": True}
    CONFIG_ORDER.append(f"R{_k}")
for _K in (1, 2, 4, 8, 16, 32, 64):
    CONFIGS[f"P{_K}"] = {"adapter": True, "objective": "tail", "preserve": True,
                         "defect_sup": True, "proxy": f"proto{_K}", "real": True}
    CONFIG_ORDER.append(f"P{_K}")
CONFIGS["X1"] = {"adapter": True, "objective": "tail", "preserve": True,
                 "defect_sup": True, "proxy": "pooled", "real": True}
CONFIG_ORDER.append("X1")
# Gate 3: the last normal-only candidate, tested for CAUSAL value.
CONFIGS["Q1"] = {"adapter": True, "objective": "tail", "preserve": True,
                 "defect_sup": True, "proxy": "band2045"}
CONFIG_ORDER.append("Q1")
# Gate 4: external anomaly residuals transported into the target. Sources are
# always the OTHER objects (leave-one-object-out); no target defect is used.
for _tag, _p in (("X0", "X0raw"), ("X4", "X4white"),
                 ("X5", "X5recol"), ("X6", "X6recol1")):
    CONFIGS[_tag] = {"adapter": True, "objective": "tail", "preserve": True,
                     "defect_sup": True, "proxy": _p}
    CONFIG_ORDER.append(_tag)


def build_maps(obj, cfg_name, dists_flat, te, grid, idx=None):
    """Write the patch-distance maps in the layout pixel_metrics_binned wants.

    `idx` restricts which images are written -- for config E it excludes the
    defect images the adapter was supervised on, so the pixel metrics stay
    comparable with the other configs.
    """
    out = os.path.join(MAPS, obj, cfg_name, "test_public")
    jobs = []
    for i in (range(len(te["names"])) if idx is None else idx):
        name, typ = te["names"][i], te["types"][i]
        d = os.path.join(out, str(typ))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, str(name)[:-4] + ".npy")
        np.save(p, dists_flat[i].reshape(grid))
        gt = None
        if typ == "bad":
            gt = os.path.join(AD2_ROOT, obj, "test_public", "ground_truth",
                              "bad", str(name)[:-4] + "_mask.png")
        jobs.append((p, gt, tuple(int(v) for v in te["img_hw"][i])))
    return jobs


def run_one(obj, cfg_name, cfg, args, tr, te, device):
    t0 = time.time()
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0],
                                      args.bank_frac, args.split_seed)
    bank = torch.from_numpy(tr["feats"][bank_idx].astype(np.float32))
    tailq = torch.from_numpy(tr["feats"][tailq_idx].astype(np.float32))
    grid = tuple(int(v) for v in te["gt_frac"].shape[1:])   # (grid_h, grid_w)
    te_feats = torch.from_numpy(te["feats"].astype(np.float32))
    assert bank.shape[1] == int(np.prod(grid)), (
        f"{obj}: patch count {bank.shape[1]} != grid {grid}")

    # Defect images are split in half: one half may supervise the adapter (E
    # only), the other half is what we report on. Training and reporting on the
    # same defects would make the AUROC a fit-to-test-set number.
    gt_flat = te["gt_frac"].reshape(len(te["types"]), -1)
    bad_all = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(args.seed).permutation(len(bad_all))
    n_sup = len(bad_all) // 2
    defect_sup_idx, defect_eval_idx = bad_all[perm[:n_sup]], bad_all[perm[n_sup:]]

    defect_flat = None
    if cfg.get("proxy"):
        p = os.path.join(RESULTS, "proxy", f"{obj}_{cfg['proxy']}.npy")
        if not os.path.exists(p):
            print(f"  !! proxy {cfg['proxy']} missing for {obj}, run "
                  f"proxy_defects.py first -- skipping")
            return None
        defect_flat = torch.from_numpy(np.load(p).astype(np.float32)).to(device)
        src = ("REAL test_public defect patches -- ORACLE, not a candidate"
               if cfg.get("real") else "from train/good only")
        print(f"  proxy '{cfg['proxy']}': {defect_flat.shape[0]} pseudo-defect "
              f"patches ({src})", flush=True)
    elif cfg.get("defect_sup"):
        parts = [te_feats[i][gt_flat[i] > PRIMARY_GT]
                 for i in defect_sup_idx]
        parts = [p for p in parts if len(p)]
        if not parts:
            print(f"  !! no defect patches to supervise on, skipping E")
            return None
        defect_flat = torch.cat(parts).to(device)
        print(f"  defect-supervision: {len(parts)} imgs, "
              f"{defect_flat.shape[0]} defect patches (eval on the other "
              f"{len(defect_eval_idx)})", flush=True)

    print(f"\n=== {obj} [{cfg_name}] bank={len(bank_idx)} "
          f"tailq={len(tailq_idx)} grid={grid} ===", flush=True)

    adapter = ResidualAdapter(te_feats.shape[-1]).to(device)
    if cfg["adapter"]:
        tcfg = dict(objective=cfg["objective"], preserve=cfg["preserve"],
                    steps=args.steps, alpha=args.alpha, lr=args.lr,
                    lam_preserve=args.lam, hidden=args.hidden,
                    query_patches=args.query_patches,
                    bank_patches=args.bank_patches, seed=args.seed,
                    lam_defect=args.lam_defect if cfg.get("defect_sup") else 0.0,
                    defect_margin=args.defect_margin)

        def log(rec):
            print(f"    step {rec['step']:>4}  loss={rec['loss']:.4f} "
                  f"base={rec['base']:.4f} preserve={rec['preserve']:.4f} "
                  f"sep={rec['separation']:.4f}", flush=True)

        adapter, hist = train_adapter(tailq, bank, tcfg, device, log=log,
                                      defect=defect_flat)
        os.makedirs(CKPT, exist_ok=True)
        torch.save(adapter.state_dict(),
                   os.path.join(CKPT, f"{obj}_{cfg_name}.pt"))

    # eval bank = the same BANK images, through whatever adapter we ended with
    with torch.no_grad():
        bank_flat = bank.reshape(-1, bank.shape[-1]).to(device)
        bank_new = adapter(bank_flat)
        dists = patch_dists_all(adapter, te_feats, bank_new, device)

    # ---- image level ----
    # With defect supervision, report only on the half of the bad images the
    # adapter never saw. --eval-half forces that same subset for EVERY config,
    # which is what makes E comparable to A/D: without it E would be measured
    # on 45 bad images and A/D on 90, and the deltas would be meaningless.
    # Only E consumes real defects, so only E must shrink its eval set; the
    # proxy configs never see test data at all.
    uses_real_defects = (cfg.get("defect_sup")
                         and (not cfg.get("proxy") or cfg.get("real")))
    report_bad = (defect_eval_idx
                  if (uses_real_defects or args.eval_half) else bad_all)
    report_idx = np.sort(np.concatenate(
        [np.where(te["types"] == "good")[0], report_bad]))
    y = (te["types"][report_idx] == "bad").astype(int)
    scores = np.array([mean_top1p(dists[i]) for i in report_idx])
    img_auroc = roc_auc_score(y, scores)

    # ---- mechanism stats ----
    # patch_dists_all returns flat (n_img, n_patch); gt_frac is (n_img, gh, gw),
    # so the per-image distances must be reshaped before masking.
    good_mask = te["types"] == "good"
    normal_d = dists[good_mask].ravel()
    defect_d = np.concatenate([dists[i].reshape(grid)[te["gt_frac"][i] > PRIMARY_GT]
                               for i in report_bad])
    clean_d = np.concatenate([dists[i].reshape(grid)[te["gt_frac"][i] == 0.0]
                              for i in report_bad])
    st = tail_stats(normal_d, defect_d)
    st["clean_mean"] = float(clean_d.mean())

    # ---- pixel level (official AD2 metrics, same code path as the baseline) ----
    if args.no_pixel:
        m = {"px_AUROC": np.nan, "AUPRO": np.nan}
    else:
        jobs = build_maps(obj, cfg_name, dists, te, grid, idx=report_idx)
        m = pixel_metrics_binned(jobs, pro_limit=0.05)

    row = {"object": obj, "config": cfg_name + args.tag,
           "img_AUROC": img_auroc * 100,
           "px_AUROC": m["px_AUROC"] * 100, "AUPRO@0.05": m["AUPRO"] * 100,
           "n_bank_imgs": len(bank_idx), "n_tailq_imgs": len(tailq_idx),
           **st, "minutes": (time.time() - t0) / 60}
    print(f"  img_AUROC={row['img_AUROC']:.1f} px_AUROC={row['px_AUROC']:.1f} "
          f"AU-PRO={row['AUPRO@0.05']:.1f} | normal_p99={st['normal_p99']:.3f} "
          f"defect_mean={st['defect_mean']:.3f} "
          f"below_p99={100*st['frac_defect_below_p99']:.0f}%", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", default=OBJECTS)
    ap.add_argument("--configs", nargs="*", default=CONFIG_ORDER)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1.0, help="preserve weight")
    ap.add_argument("--lam-defect", type=float, default=1.0, dest="lam_defect",
                    help="defect-separation weight (config E only)")
    ap.add_argument("--defect-margin", type=float, default=0.05,
                    dest="defect_margin")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--query-patches", type=int, default=6144,
                    dest="query_patches")
    ap.add_argument("--bank-patches", type=int, default=8192, dest="bank_patches")
    ap.add_argument("--bank-frac", type=float, default=0.8, dest="bank_frac")
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-half", action="store_true", dest="eval_half",
                    help="report every config on the held-out defect half, so "
                         "results are comparable with the E probe")
    ap.add_argument("--tag", default="", help="suffix appended to config names")
    ap.add_argument("--no-pixel", action="store_true", dest="no_pixel",
                    help="skip the pixel metrics; the scans only need the "
                         "mechanism statistics and this cuts runtime ~3x")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(METRICS, exist_ok=True)
    rows = []
    for obj in args.objects:
        tr, te = load_cache(obj)
        assert tr["feats"].shape[1] == te["feats"].shape[1], \
            f"{obj}: train/test patch counts differ"
        for cfg_name in args.configs:
            rows.append(run_one(obj, cfg_name, CONFIGS[cfg_name], args, tr, te,
                                args.device))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no config produced a row (see messages above)")
    out = os.path.join(METRICS, "oracle_ablation.csv")
    # Merge rather than overwrite, so configs can be run in separate passes
    # (e.g. the defect-supervised probe E after the main ablation) without
    # losing the rows already computed.
    if os.path.exists(out):
        prev = pd.read_csv(out)
        key = ["object", "config"]
        prev = prev[~prev.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values(["object", "config"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    show = df[["object", "config", "img_AUROC", "px_AUROC", "AUPRO@0.05",
               "normal_p99", "defect_mean", "frac_defect_below_p99",
               "defect_normal_margin"]].round(3)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
