# -*- coding: utf-8 -*-
"""
PHASE 8 -- External Defect Support Acquisition.

Everything through Gate 7 assumed the defect support had to come from the
TARGET class. Phase 8 drops that assumption: the support is built from DEFECT
PATCHES OF OTHER CLASSES. No target-class defect patch enters any bank, and no
target anomaly label is used anywhere -- not to size the bank, not to choose
which external class to draw from, not to fuse.

Detector is frozen: no adapter, no training.
    S_n  per-patch 1-NN distance to the TARGET's NORMAL bank (train/good)
    S_d  per-patch 1-NN distance to the EXTERNAL defect bank
    score = S_n - S_d                       (raw fusion, beta = 1)

Three acquisition strategies:
    R0  RANDOM EXTERNAL BANK     uniform random external defect patches
    R1  MATCHED CATEGORY         rank external classes by how well their NORMAL
                                 manifold covers the target's NORMAL manifold,
                                 then draw defects from the best-matching ones
    R2  LARGE POOLED BANK        every external defect patch

The built-in null: if the external bank is irrelevant to the target, S_d is
near-constant across the target's images and the raw fusion reduces to S_n up to
an additive constant -- which cannot change AUROC. So R_* ~ baseline means "no
transfer"; any lift over baseline is real, and R1 > R0 at matched size is the
specific claim that RETRIEVAL does work.

Because the pool is the other AD2 classes, a positive result licenses
cross-CLASS transfer within one dataset, not transfer from a separate corpus.

Evaluation uses ALL target test_public images (no support/eval split is possible
or needed: no target defect is used). The frozen AnomalyDINO baseline is
recomputed under this identical protocol below, so deltas are apples-to-apples
rather than compared against numbers from a different split.

Usage: python experiments/model_v0/gate8_external.py [--splits N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate7b3r import make_dist  # noqa: E402
from tail_calib import RESULTS, load_cache, split_train, mean_top1p  # noqa: E402

OBJECTS = ["wallplugs", "vial", "sheet_metal", "can"]
SIZES = [50, 100, 500, 1000, -1]
MATCH_K = [1, 3]
METRICS = os.path.join(RESULTS, "metrics")
EXTERNAL = os.path.join(RESULTS, "external")


def load_external(target):
    """All defect patches from classes other than `target`, plus per-class pools."""
    pools = {}
    for f in sorted(os.listdir(EXTERNAL)):
        if not f.endswith("_defect.npy"):
            continue
        obj = f[:-len("_defect.npy")]
        if obj == target:
            continue                      # hard guard: no target defects, ever
        d = np.load(os.path.join(EXTERNAL, f))
        if len(d):
            pools[obj] = d
    return pools


def class_match_ranking(target, target_normals, device="cuda"):
    """Rank external classes by how well their NORMALS cover the target's.

    Mean 1-NN distance from each target normal patch to the external class's
    normal patches; lower is a better match. Uses only normal data from both
    sides, so no anomaly label of any class is involved.
    """
    q = torch.from_numpy(target_normals).to(device)
    out = []
    for obj in sorted(os.listdir(EXTERNAL)):
        if not obj.endswith("_normal.npy") or obj[:-len("_normal.npy")] == target:
            continue
        cname = obj[:-len("_normal.npy")]
        cn = np.load(os.path.join(EXTERNAL, obj))
        if len(cn) == 0:
            continue
        with torch.no_grad():
            d = make_dist(torch.from_numpy(cn), device)(q).mean().item()
        out.append({"class": cname, "match_dist": d})
    return pd.DataFrame(out).sort_values("match_dist").reset_index(drop=True)


def evaluate(obj, pools, ranking, device="cuda"):
    tr, te = load_cache(obj)
    n_train = tr["feats"].shape[0]
    bank_idx, _ = split_train(n_train, 1.0, 0)   # ALL train normals: nothing is
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)      # held out
                             .astype(np.float32)).to(device)
    types = list(te["types"])
    n_img = len(types)
    gt = te["gt_frac"].reshape(n_img, -1)
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(n_img)]
    y = np.array([1 if t == "bad" else 0 for t in types])
    dist_n = make_dist(fbank, device)
    with torch.no_grad():
        Sn = [dist_n(z) for z in feats]

    def run(bank, label, size):
        if bank is None:
            sc = np.array([mean_top1p(Sn[i]) for i in range(n_img)])
            sd = np.full(n_img, np.nan)
        else:
            dist_d = make_dist(torch.from_numpy(bank.astype(np.float32)), device)
            Sd = [dist_d(z) for z in feats]
            # branch diagnostic: could S_d alone rank the target's anomalies?
            sd = np.array([-np.sort(Sd[i])[:max(1, int(len(Sd[i]) * .01))].mean()
                           for i in range(n_img)])
            sc = np.array([mean_top1p(Sn[i] - Sd[i]) for i in range(n_img)])
        return {
            "object": obj, "config": label, "size": size,
            "n_eval_good": int((y == 0).sum()), "n_eval_bad": int((y == 1).sum()),
            "img_AUROC": roc_auc_score(y, sc) * 100,
            "Sd_alone_AUROC": (roc_auc_score(y, sd) * 100
                               if bank is not None else np.nan),
            # spread of the S_d term across images: if this is ~0 the bank is
            # inert and the fusion degenerates to the baseline by construction
            "Sd_spread": (float(np.nanstd(sd)) if bank is not None else np.nan),
        }

    rows = [run(None, "baseline_Sn", 0)]
    allp = np.concatenate([pools[k] for k in sorted(pools)])
    rng = np.random.default_rng(0)

    # ---- R0 random external ----
    for n in SIZES:
        k = len(allp) if n == -1 else min(n, len(allp))
        sel = rng.choice(len(allp), size=k, replace=False)
        rows.append(run(allp[sel], "R0_random_external", k))

    # ---- R1 matched category ----
    for k_top in MATCH_K:
        cls = list(ranking["class"])[:k_top]
        sub = np.concatenate([pools[c] for c in cls if c in pools])
        for n in SIZES:
            k = len(sub) if n == -1 else min(n, len(sub))
            sel = rng.choice(len(sub), size=k, replace=False)
            rows.append(run(sub[sel], f"R1_matched_top{k_top}", k))

    # ---- R2 pooled ----
    rows.append(run(allp, "R2_pooled_external", len(allp)))
    return rows, ranking, {"n_pool": len(allp),
                           "n_classes": len(pools)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1,
                    help="R0/R1 subsample at different seeds; baseline and R2 "
                         "are deterministic so extra splits only resample banks")
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)
    rows, ranks, info = [], [], []
    for o in OBJECTS:
        pools = load_external(o)
        tr, _ = load_cache(o)
        tn = tr["feats"].reshape(-1, 384).astype(np.float32)
        rng = np.random.default_rng(0)
        if len(tn) > 20000:
            tn = tn[rng.choice(len(tn), 20000, replace=False)]
        r = class_match_ranking(o, tn)
        r.insert(0, "target", o)
        ranks.append(r)
        print(f"\n=== {o} === external classes by normal-manifold match "
              f"(lower = closer)")
        print(r.to_string(index=False), flush=True)
        for _ in range(args.splits):
            rr, _, ii = evaluate(o, pools, r)
            rows.extend(rr)
            info.append({"object": o, **ii})
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate8_external.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, "gate8_external.csv"), index=False)
    pd.concat(ranks).to_csv(os.path.join(METRICS, "gate8_class_match.csv"),
                            index=False)

    print("\n" + "=" * 100)
    print("PHASE 8 -- external defect banks, frozen detector, raw fusion S_n - S_d")
    print("=" * 100)
    piv = df.pivot_table(index="config", columns="object", values="img_AUROC")
    print(piv.round(1).to_string())
    print("\n  baseline_Sn = frozen AnomalyDINO under this same protocol")
    print("\n  size actually used:")
    print(df.pivot_table(index="config", columns="object", values="size")
          .astype(int).to_string())
    print("\n  S_d alone (ranked as a detector, external bank vs target anomalies):")
    print(df.pivot_table(index="config", columns="object", values="Sd_alone_AUROC")
          .round(1).to_string())
    print("\n  S_d spread across images (0 => bank inert => fusion == baseline):")
    print(df.pivot_table(index="config", columns="object", values="Sd_spread")
          .round(4).to_string())


if __name__ == "__main__":
    main()
