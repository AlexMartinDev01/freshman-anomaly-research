# -*- coding: utf-8 -*-
"""
Phase R step 3 (robustness) -- is agg_all3's gain an artifact of the WEIGHTS?

`agg_all3` averages the three layers after z-scoring each with the object's
TEST-set mean/std. That z-score cannot change any reported metric on its own
(one affine transform shared by every map of a layer leaves all rankings, and
therefore AUROC and AUPRO, unchanged). Its only effect is to set the mixing
ratio: agg = mean(m_l / sigma_l) up to a constant.

So the one legitimate criticism is that sigma_l is estimated transductively.
This script removes that by re-running the aggregation with weights that never
touch the test set:

    train_z    per layer, (mu, sigma) estimated from a fixed sample of TRAIN
               patches scored by 1-NN against a fixed subsample of the same
               object's TRAIN pool -- no test data anywhere
    raw_mean   no weights at all, plain mean of the three raw distance maps

If agg_all3's advantage over final-only survives both, the transductive
objection is dead. If it survives only the test-set-weighted version, the
advantage is a weighting artifact and must be reported as one.

Only the two baselines of interest are recomputed; this does not re-run the
full grid, it re-weights maps that the confirmation run already produced.

Usage: python experiments/model_v0/gate_r2_weight_robustness.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, MAPS, ML, MLI, V1,  # noqa: E402
                                   draw_images, gt_path, l2n, load_test,
                                   load_train, nn_dist, obj_seed)

METRICS = os.path.join(RESULTS, "metrics")
QUERY_N = 1024          # train patches used to estimate (mu, sigma)
REF_N = 20000           # train patches used as the 1-NN reference


def train_layer_stats(layer, obj, seed=12345):
    """(mu, sigma) of 1-NN distances for this object/layer, from TRAIN only.

    Queries and reference are disjoint halves of the train pool, so these
    distances are the train-side analogue of the test maps -- same statistic,
    estimated without ever looking at a test image.
    """
    feats, offs = load_train(layer, obj)
    pool = feats.reshape(-1, 384).astype(np.float32)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pool), size=min(QUERY_N + REF_N, len(pool)),
                     replace=False)
    q_idx, r_idx = sel[:QUERY_N], sel[QUERY_N:]
    q = l2n(torch.from_numpy(pool[q_idx]).to(DEV))
    ref = l2n(torch.from_numpy(pool[r_idx]).to(DEV))
    d = nn_dist(q, ref).cpu().numpy()
    return float(d.mean()), float(d.std() + 1e-12)


def run(obj, shot, split, TESTS, STATS):
    per = {}
    for layer in LAYERS:
        feats, offs = load_train(layer, obj)
        te = TESTS[layer]
        idx = draw_images(offs, shot, split, obj)
        bank = l2n(torch.from_numpy(
            np.concatenate([feats[offs[i]:offs[i + 1]] for i in idx]
                           ).astype(np.float32)).to(DEV))
        maps = []
        for i in range(len(te["types"])):
            q = torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bank).cpu().numpy())
        per[layer] = maps

    def z_test(v):
        a = np.concatenate([m.ravel() for m in v])
        mu, sd = a.mean(), a.std() + 1e-12
        return [(m - mu) / sd for m in v]

    Zt = {k: z_test(v) for k, v in per.items()}
    Zi = {k: [(m - STATS[k][0]) / STATS[k][1] for m in v]
          for k, v in per.items()}
    M = {
        "agg_all3_testz": [(a + b + c) / 3 for a, b, c in
                           zip(Zt["mid"], Zt["midlate"], Zt["final"])],
        "agg_all3_trainz": [(a + b + c) / 3 for a, b, c in
                            zip(Zi["mid"], Zi["midlate"], Zi["final"])],
        "raw_mean": [(a + b + c) / 3 for a, b, c in
                     zip(per["mid"], per["midlate"], per["final"])],
        "final_raw": per["final"],
    }

    te = TESTS["final"]
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    rows = []
    for name, maps in M.items():
        sc = np.array([mean_top1p(m) for m in maps])
        d = os.path.join(MAPS + "_weights", obj, f"{shot}shot_s{split}", name)
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(len(y)):
            p = os.path.join(d,
                             str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            g2 = maps[i].reshape(gh, gw)
            assert g2.shape == (gh, gw), f"map shape {g2.shape}"
            np.save(p, g2)
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "config": name,
                     "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="gate_r2_weight_robustness")
    a = ap.parse_args()
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1)
        if os.path.isdir(os.path.join(V1, d, "train")))
    shots = [int(s) for s in a.shots.split(",")]

    rows = []
    for obj in objs:
        TESTS = {l: load_test(l, obj) for l in LAYERS}
        STATS = {l: train_layer_stats(l, obj) for l in LAYERS}
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(obj, shot, sp, TESTS, STATS))
        print(f"  {obj:<12} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)

    piv = df.pivot_table(index=["object", "shot", "split"], columns="config",
                         values=["img_AUROC", "px_AUROC", "AUPRO"])
    print("\n" + "=" * 100)
    print("WEIGHT ROBUSTNESS -- is agg_all3's gain just the test-set weights?")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        base = piv[(met, "final_raw")]
        print(f"\n  {met}   final_raw {base.mean():.2f}")
        for c in ["agg_all3_testz", "agg_all3_trainz", "raw_mean"]:
            d = piv[(met, c)] - base
            print(f"    {c:<18} {piv[(met, c)].mean():6.2f}   "
                  f"gain {d.mean():+6.2f}   better {int((d > 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
