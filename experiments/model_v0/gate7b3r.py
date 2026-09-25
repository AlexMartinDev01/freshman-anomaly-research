# -*- coding: utf-8 -*-
"""
Gate 7B-3R -- Defect Support Coverage Curve (raw fusion, no MAD).

Gate 7B-3 was run with MAD standardisation, which 7B-2 showed is actively
harmful (wallplugs 77.0 raw vs 45.9 MAD), so its flat curve says nothing about
support coverage. This reruns it with the raw fusion S_n - S_d and fixes a
second flaw: sampling PATCHES treats patches from one image as independent
samples, when they are not. The curves here are indexed by defect IMAGE count.

Three questions:
  1. CURVE      does detection rise with the number of distinct defect images
                in the support? Rising -> coverage is the bottleneck and a
                generator has a job. Flat -> few supports already suffice and
                the real problem is SELECTION, not generation.
  2. DIVERSITY  at a matched budget, random patches vs farthest-point
                (k-center) patches. If diversity wins, a generator's target is
                support COVERAGE rather than density fitting.
  3. COVERAGE   for each held-out defect patch, its distance to the nearest
                support patch; then whether poorly-covered patches are the ones
                that fail. This is how we test the hypothesis that can is
                support-limited rather than feature-limited.

Usage: python experiments/model_v0/gate7b3r.py [--splits N] [--pixel]
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
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from models.tail_adapter import l2norm
from tail_calib import RESULTS, load_cache, split_train, mean_top1p

OBJECTS = ["wallplugs", "sheet_metal", "vial", "can"]
IMG_COUNTS = [1, 2, 4, 8, 16, -1]
BUDGETS = [20, 50, 100, 400]
METRICS = os.path.join(RESULTS, "metrics")


def nn_dist(query, bank, chunk=1024):
    q, b = l2norm(query.float()), l2norm(bank.float())
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ b.T).max(dim=1).values)
    return torch.cat(out)


def kcenter(X, k, seed=0):
    """Greedy farthest-point sampling; deterministic apart from the first pick."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = 1.0 - l2norm(X) @ l2norm(X[idx[0]]).T
    d = d.ravel()
    for _ in range(min(k, len(X)) - 1):
        i = int(torch.argmax(d))
        idx.append(i)
        nd = 1.0 - l2norm(X) @ l2norm(X[i]).T
        d = torch.minimum(d, nd.ravel())
    return np.array(idx[:k])


def run(obj, split, device="cuda"):
    tr, te = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                             .astype(np.float32)).to(device)
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    bad = np.where(~(np.array(types) == "good"))[0]
    perm = np.random.default_rng(1000 + split).permutation(len(bad))
    sup_idx = bad[perm[:len(bad) // 2]]
    ev_idx = bad[perm[len(bad) // 2:]]
    good = [i for i in range(len(types)) if types[i] == "good"]
    ev = good + list(ev_idx)
    y = np.array([0 if types[i] == "good" else 1 for i in ev])
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(len(types))]

    def patches_of(imgs):
        return torch.from_numpy(np.concatenate(
            [te["feats"][i].astype(np.float32)[gt[i] > 0.10] for i in imgs]))

    with torch.no_grad():
        Sn = [nn_dist(z, fbank).cpu().numpy() for z in feats]
    rows = []
    rng = np.random.default_rng(7 + split)

    # ---- curve 1: by defect IMAGE count ----
    for n_img in IMG_COUNTS:
        imgs = sup_idx if n_img == -1 else rng.choice(
            sup_idx, size=min(n_img, len(sup_idx)), replace=False)
        dsupp = patches_of(imgs).to(device)
        with torch.no_grad():
            Sd = [nn_dist(feats[i], dsupp).cpu().numpy() for i in ev]
        sc = np.array([mean_top1p(Sn[i] - Sd[k]) for k, i in enumerate(ev)])
        rows.append({"object": obj, "split": split, "axis": "image_count",
                     "n": len(imgs), "n_patches": len(dsupp),
                     "method": "all", "img_AUROC": roc_auc_score(y, sc) * 100})

    # ---- curve 2: matched budget, random vs k-center ----
    full = patches_of(sup_idx).to(device)
    for bud in BUDGETS:
        if bud > len(full):
            continue
        for method in ("random", "kcenter"):
            sel = (rng.choice(len(full), size=bud, replace=False)
                   if method == "random" else kcenter(full, bud, seed=7 + split))
            dsupp = full[torch.from_numpy(sel).to(device)]
            with torch.no_grad():
                Sd = [nn_dist(feats[i], dsupp).cpu().numpy() for i in ev]
            sc = np.array([mean_top1p(Sn[i] - Sd[k]) for k, i in enumerate(ev)])
            rows.append({"object": obj, "split": split, "axis": "budget",
                         "n": bud, "n_patches": bud, "method": method,
                         "img_AUROC": roc_auc_score(y, sc) * 100})

    # ---- coverage analysis ----
    dsupp = full
    cov = []
    for k, i in enumerate(ev):
        if types[i] == "good":
            continue
        with torch.no_grad():
            d = nn_dist(feats[i], dsupp).cpu().numpy()
        m = gt[i] > 0.10
        if m.sum() == 0:
            continue
        cov.append({"object": obj, "split": split, "image": str(te["names"][i]),
                    "cov_mean": float(d[m].mean()),
                    "cov_min": float(d[m].min()),
                    "score": float(mean_top1p(Sn[i] - d)),
                    "n_def_patches": int(m.sum())})
    return rows, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)
    rows, covs = [], []
    for o in OBJECTS:
        for s in range(args.splits):
            r, c = run(o, s)
            rows.extend(r)
            covs.extend(c)
            print(f"  {o} split {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, "gate7b3r_curve.csv"), index=False)
    pd.DataFrame(covs).to_csv(os.path.join(METRICS, "gate7b3r_coverage.csv"),
                              index=False)

    print("\n" + "=" * 100)
    print("7B-3R (1) CURVE BY DEFECT IMAGE COUNT -- raw fusion, no MAD")
    print("=" * 100)
    p = df[df.axis == "image_count"].pivot_table(
        index="n", columns="object", values="img_AUROC")
    print(p.round(1).to_string())
    print("\n  (-1 = all available support images)")

    print("\n" + "=" * 100)
    print("7B-3R (2) MATCHED BUDGET: random vs k-center (diversity)")
    print("=" * 100)
    p2 = df[df.axis == "budget"].pivot_table(
        index=["method", "n"], columns="object", values="img_AUROC")
    print(p2.round(1).to_string())

    print("\n" + "=" * 100)
    print("7B-3R (3) COVERAGE -- distance of held-out defect patches to the "
          "support")
    print("=" * 100)
    c = pd.DataFrame(covs)
    for o in OBJECTS:
        s = c[c.object == o]
        if s.empty:
            continue
        # split images by coverage and ask whether poorly-covered ones score lower
        med = s.cov_mean.median()
        lo, hi = s[s.cov_mean <= med], s[s.cov_mean > med]
        print(f"  {o:<12} n={len(s):>4}  cov_mean median={med:.3f}  "
              f"score(well-covered)={lo.score.mean():>7.3f}  "
              f"score(poorly-covered)={hi.score.mean():>7.3f}  "
              f"def_patches_img mean={s.n_def_patches.mean():.1f}")


if __name__ == "__main__":
    main()
