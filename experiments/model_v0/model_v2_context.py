# -*- coding: utf-8 -*-
"""
Model V2 smoke -- context-conditioned normal matching (training-free).

F2 replicated, on VisA with pre-registered criteria, the F1 mechanism: both
detectors' low-FPR localisation is governed by how separable a defect is from
its LOCAL normal structure. The natural model response is to stop asking
"how far is this patch from ANY normal patch" and start asking "how far is it
from normal patches that sit in a similar local context" -- i.e. model
p(z | context) instead of p(z).

    s_i = min_{j in N(c_i)} d(z_i, z_j)
    c_i = mean of the 8 neighbours' features around patch i (3x3, centre out)
    N(c_i) = the K bank patches whose contexts are closest to c_i (cosine)

Everything else is held identical to agg_all3: same frozen features, same
k-shot bank images and draws, same per-layer z-score-then-average aggregation,
same evaluator at pro_limit=0.05. No training, no anomaly labels, no
hyperparameter search beyond the pre-declared sensitivity set.

PRE-REGISTERED SMOKE CRITERIA (frozen before the run):

  M1  the 4 lowest-C_feat objects (transistor, screw, zipper, cable) gain
      AUPRO >= +2 on average vs agg_all3
  M2  no single one of those objects loses more than 3 AUPRO
      (so one object cannot carry the result)
  M3  mean image AUROC over the smoke set is not worse than agg_all3 by > 0.3

K = 64 is the PRIMARY configuration and the only one M1-M3 are judged on.
K in {16, 256} is reported as a sensitivity table, NOT as a verdict -- picking
whichever K wins after seeing the numbers would be the same criterion-shopping
that closed the earlier branches.

Control objects (grid, carpet) are the two HIGHEST C_feat objects; they are
reported but not part of M1/M2.

Usage: python experiments/model_v0/model_v2_context.py
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, MAPS, ML, MLI, V1,  # noqa: E402
                                   draw_images, gt_path, l2n, load_test,
                                   load_train, nn_dist, zscores)
from sklearn.metrics import roc_auc_score  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
MAPS_CTX = os.path.join(RESULTS, "maps_model_v2_ctx")
HARD = ["transistor", "screw", "zipper", "cable"]
CTRL = ["grid", "carpet"]
GRID = (32, 32)          # MVTec: every object's test grid is 32x32
K_PRIMARY = 64
K_SENS = [16, 256]


def context_maps(feats, grid):
    """feats (P, D) for one image -> context descriptor (P, D). Mean of the 8
    neighbours with the centre removed; borders average what exists."""
    gh, gw = grid
    F = feats.reshape(gh, gw, -1)
    acc = np.zeros_like(F, dtype=np.float32)
    cnt = np.zeros((gh, gw, 1), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sy0, sy1 = max(0, -dy), min(gh, gh - dy)
            sx0, sx1 = max(0, -dx), min(gw, gw - dx)
            ty0, ty1 = sy0 + dy, sy1 + dy
            tx0, tx1 = sx0 + dx, sx1 + dx
            acc[sy0:sy1, sx0:sx1] += F[ty0:ty1, tx0:tx1]
            cnt[sy0:sy1, sx0:sx1] += 1.0
    return (acc / np.maximum(cnt, 1.0)).reshape(-1, F.shape[-1])


def build_bank(layer, obj, idx):
    """Bank patches and their contexts, from the drawn images."""
    feats, offs = load_train(layer, obj)
    zp, zc = [], []
    for i in idx:
        f = l2n(torch.from_numpy(feats[offs[i]:offs[i + 1]]
                                .astype(np.float32)).to(DEV)).cpu().numpy()
        zp.append(f)
        zc.append(l2n(torch.from_numpy(
            context_maps(f, GRID)).to(DEV)).cpu().numpy())
    return (np.concatenate(zp).astype(np.float32),
            np.concatenate(zc).astype(np.float32))


def ctx_dist_all(zq, cq, zb, cb, Ks):
    """s_i = min over the K context-nearest bank patches of d(z_i, z_j), for
    every K in Ks. One topk at max(Ks): its output is sorted, so the first K
    columns are exactly the top-K selection -- computing a separate similarity
    matrix per K would repeat the same matmul three times."""
    Kmax = max(Ks)
    zq_t = torch.from_numpy(zq).to(DEV)
    zb_t = torch.from_numpy(zb).to(DEV)
    cq_t = torch.from_numpy(cq).to(DEV)
    cb_t = torch.from_numpy(cb).to(DEV)
    topk = torch.topk(cq_t @ cb_t.T, Kmax, dim=1).indices
    out = {K: [] for K in Ks}
    for a in range(0, len(zq), 2048):
        idx = topk[a:a + 2048]
        cand = zb_t[idx]                                  # (b, Kmax, D)
        dd = 1.0 - (cand * zq_t[a:a + 2048].unsqueeze(1)).sum(-1)
        for K in Ks:
            out[K].append(dd[:, :K].min(dim=1).values)
    return {K: torch.cat(v).cpu().numpy() for K, v in out.items()}


def run(obj, shot, split, TESTS, Ks):
    per = {K: {l: [] for l in LAYERS} for K in Ks}
    for layer in LAYERS:
        feats, offs = load_train(layer, obj)
        idx = draw_images(offs, shot, split, obj)
        zb, cb = build_bank(layer, obj, idx)
        te = TESTS[layer]
        for i in range(len(te["types"])):
            f = l2n(torch.from_numpy(
                te["feats"][i].astype(np.float32)).to(DEV)).cpu().numpy()
            cq = l2n(torch.from_numpy(context_maps(f, GRID)).to(DEV)).cpu().numpy()
            res = ctx_dist_all(f, cq, zb, cb, Ks)
            for K in Ks:
                per[K][layer].append(res[K])
    out = {}
    for K in Ks:
        Z = {l: zscores(per[K][l]) for l in LAYERS}
        out[K] = [(a + b + c) / 3 for a, b, c in
                  zip(Z["mid"], Z["midlate"], Z["final"])]
    return out


def evaluate(maps, te, obj, tag, shot, split):
    """NOTE: the map path must carry (shot, split). Without it every cell
    overwrites the previous cell's maps in the same directory, so only the last
    cell survives on disk. That did NOT affect any metric here -- jobs are
    built and scored inside this same call, before the next cell writes -- but
    it makes the saved maps useless for later inspection, which is exactly the
    kind of silent artifact loss this project has been bitten by before."""
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    sc = np.array([mean_top1p(m) for m in maps])
    d = os.path.join(MAPS_CTX, tag, obj, f"{shot}shot_s{split}")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for i in range(len(y)):
        p = os.path.join(d, str(te["names"][i])[:-4].replace("/", "_") + ".npy")
        g2 = maps[i].reshape(gh, gw)
        assert g2.shape == (gh, gw)
        np.save(p, g2)
        g = gt_path(obj, te["names"][i]) if y[i] else None
        jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
    m = pixel_metrics_binned(jobs, pro_limit=0.05)
    return {"img_AUROC": roc_auc_score(y, sc) * 100,
            "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(HARD + CTRL))
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="model_v2_context_smoke")
    a = ap.parse_args()
    objs = a.objects.split(",")
    shots = [int(s) for s in a.shots.split(",")]
    Ks = [K_PRIMARY] + K_SENS
    os.makedirs(MAPS_CTX, exist_ok=True)

    rows = []
    t0 = time.time()
    for obj in objs:
        TESTS = {l: load_test(l, obj) for l in LAYERS}
        te = TESTS["final"]
        for shot in shots:
            for sp in range(a.splits):
                out = run(obj, shot, sp, TESTS, Ks)
                for K in Ks:
                    r = evaluate(out[K], te, obj, f"ctx_K{K}", shot, sp)
                    rows.append({"object": obj, "shot": shot, "split": sp,
                                 "config": f"ctx_K{K}", **r})
        print(f"  {obj:<12} done ({time.time() - t0:5.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    report(df, objs, shots)


def report(df, objs, shots):
    """Merge the agg_all3 baseline and print the PRE-REGISTERED verdict.

    One implementation, shared by the sharded run and the merge script -- the
    thresholds must not be re-typed, or they can drift after the fact.
    """
    ref = pd.read_csv(os.path.join(METRICS, "gate_r2_layer_confirm.csv"))
    ref = ref[(ref.config == "agg_all3") & (ref.object.isin(objs)) &
              (ref.shot.isin(shots))][["object", "shot", "split",
                                       "img_AUROC", "px_AUROC", "AUPRO"]]
    ref["config"] = "agg_all3"
    df = pd.concat([ref, df], ignore_index=True)
    df.to_csv(os.path.join(METRICS, "model_v2_context_smoke_merged.csv"),
              index=False)

    print("\n" + "=" * 100)
    print("MODEL V2 SMOKE -- context-conditioned normal matching")
    print("=" * 100)
    g = df.groupby(["object", "config"])[["img_AUROC", "px_AUROC", "AUPRO"]].mean()
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        piv = g[met].unstack()
        piv = piv[["agg_all3", "ctx_K64", "ctx_K16", "ctx_K256"]]
        print(f"\n  {met}")
        print(piv.round(2).to_string())
        d = piv["ctx_K64"] - piv["agg_all3"]
        print("    ctx_K64 - agg_all3 : " +
              "  ".join(f"{o}={v:+.2f}" for o, v in d.items()))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED SMOKE CRITERIA (judged on K=64 only)")
    print("=" * 100)
    piv = df.pivot_table(index=["object", "shot", "split"], columns="config",
                         values="AUPRO")
    dpr = piv["ctx_K64"] - piv["agg_all3"]
    present = [o for o in HARD if o in dpr.index.get_level_values(0)]
    missing = [o for o in HARD if o not in present]
    if missing:
        print(f"\n  !! hard objects missing from this run: {missing} -- "
              f"criteria NOT evaluable on a subset")
        return
    hard = dpr.loc[present].groupby(level=0).mean()
    m1 = hard.mean() >= 2.0
    m2 = hard.min() >= -3.0
    pimg = df.pivot_table(index=["object", "shot", "split"], columns="config",
                          values="img_AUROC")
    di = (pimg["ctx_K64"] - pimg["agg_all3"])
    m3 = di.mean() >= -0.3
    print("  hard-object AUPRO gain (K=64): " +
          "  ".join(f"{o}={v:+.2f}" for o, v in hard.items()))
    print(f"  M1 mean gain {hard.mean():+.2f} (need >= +2.0)  "
          f"-> {'OK' if m1 else 'FAIL'}")
    print(f"  M2 worst object {hard.min():+.2f} (need >= -3.0)  "
          f"-> {'OK' if m2 else 'FAIL'}")
    print(f"  M3 mean image AUROC {di.mean():+.2f} (need >= -0.3)  "
          f"-> {'OK' if m3 else 'FAIL'}")
    print(f"\n  -> {'GO: run the full 15 objects' if (m1 and m2 and m3) else 'NO-GO: stop'}")


if __name__ == "__main__":
    main()
