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


def make_dist(bank, device="cuda", q_chunk=1024, b_chunk=8192):
    """1-NN cosine distance to a bank, normalised once and chunked on BOTH sides.

    nn_dist() above is fine for small banks but sheet_metal's normal bank is
    450k patches: (1024 x 384) @ (384 x 450560) materialises a 1.85 GB
    intermediate per chunk, per test image.

    The chunk shapes are measured, not guessed -- on an RTX 5060, against a
    450k-patch bank with 4096 queries, per image:
        1024 x 8192 -> 0.24 s    256 x 2048 -> 0.24 s
         128 x 1024 -> 0.88 s     64 x  512 -> 7.17 s
    Smaller tiles are much WORSE, so do not "optimise" these downward. fp16 is
    not faster here either (0.24 s at the same tile), so fp32 stays for accuracy.

    Note VRAM: this holds the whole bank plus the normalised copy. One process
    for sheet_metal peaks near 2.5 GB; two concurrent processes exceeded 8 GB and
    the Windows driver paged VRAM to system RAM, turning a 30 s step into 20+
    minutes with no error. Run these sweeps SERIALLY.
    """
    bn = l2norm(bank.to(device).float())

    def dist(z):
        q = l2norm(z.float())
        out = []
        for i in range(0, len(q), q_chunk):
            qc = q[i:i + q_chunk]
            best = torch.full((len(qc),), -1.0, device=qc.device)
            for j in range(0, len(bn), b_chunk):
                best = torch.maximum(best, (qc @ bn[j:j + b_chunk].T).max(dim=1).values)
            out.append(1.0 - best)
        return torch.cat(out).cpu().numpy()
    return dist


def kcenter(X, k, seed=0):
    """Greedy farthest-point sampling; deterministic apart from the first pick."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = (1.0 - l2norm(X) @ l2norm(X[idx[0]]).unsqueeze(0).T).ravel()
    for _ in range(min(k, len(X)) - 1):
        i = int(torch.argmax(d))
        idx.append(i)
        nd = 1.0 - l2norm(X) @ l2norm(X[i]).unsqueeze(0).T
        d = torch.minimum(d, nd.ravel())
    return np.array(idx[:k])


def prepare(obj, device="cuda"):
    """Load once per object; the split loop then reuses it.

    Previously each split called load_cache, re-reading a 250-400 MB npz three
    times per object for nothing.
    """
    tr, te = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                             .astype(np.float32)).to(device)
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    bad = np.where(~(np.array(types) == "good"))[0]
    good = [i for i in range(len(types)) if types[i] == "good"]
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(len(types))]
    with torch.no_grad():
        Sn = [nn_dist(z, fbank).cpu().numpy() for z in feats]
    return te, gt, feats, Sn, bad, good


def run(obj, split, ctx, device="cuda"):
    te, gt, feats, Sn, bad, good = ctx
    perm = np.random.default_rng(1000 + split).permutation(len(bad))
    sup_idx = bad[perm[:len(bad) // 2]]
    ev_idx = bad[perm[len(bad) // 2:]]
    ev = good + list(ev_idx)
    y = np.array([0 if te["types"][i] == "good" else 1 for i in ev])

    def patches_of(imgs):
        parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10] for i in imgs]
        parts = [p for p in parts if len(p)]
        return (torch.from_numpy(np.concatenate(parts)) if parts
                else torch.zeros((0, 384), dtype=torch.float32))

    def fused(dsupp):
        # patches_of returns CPU tensors while the budget branch indexes a
        # CUDA tensor; normalise the device here so both paths agree
        dsupp = dsupp.to(device)
        with torch.no_grad():
            Sd = [nn_dist(feats[i], dsupp).cpu().numpy() for i in ev]
        sc = np.array([mean_top1p(Sn[i] - Sd[k]) for k, i in enumerate(ev)])
        return roc_auc_score(y, sc) * 100

    rows = []
    rng = np.random.default_rng(7 + split)
    # only images that actually carry a defect patch may serve as support:
    # can's defects are so small that a random image can contribute none
    usable = [i for i in sup_idx if (gt[i] > 0.10).sum() > 0]

    for n_img in IMG_COUNTS:
        imgs = usable if n_img == -1 else rng.choice(
            usable, size=min(n_img, len(usable)), replace=False)
        dsupp = patches_of(imgs)
        if len(dsupp) == 0:
            continue
        rows.append({"object": obj, "split": split, "axis": "image_count",
                     "n": len(imgs), "n_patches": len(dsupp), "method": "all",
                     "img_AUROC": fused(dsupp)})

    # single device transfer: patches_of builds on CPU, everything
    # downstream (fused, kcenter, coverage) assumes the device
    full = patches_of(usable).to(device)
    if len(full):
        for bud in BUDGETS:
            if bud > len(full):
                continue
            for method in ("random", "kcenter"):
                sel = (rng.choice(len(full), size=bud, replace=False)
                       if method == "random"
                       else kcenter(full, bud, seed=7 + split))
                rows.append({"object": obj, "split": split, "axis": "budget",
                             "n": bud, "n_patches": bud, "method": method,
                             "img_AUROC": fused(full[torch.from_numpy(sel)])})

    cov = []
    if len(full):
        for i in ev:
            if te["types"][i] == "good":
                continue
            m = gt[i] > 0.10
            if m.sum() == 0:
                continue
            with torch.no_grad():
                d = nn_dist(feats[i], full).cpu().numpy()
            cov.append({"object": obj, "split": split,
                        "image": str(te["names"][i]),
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
    cpath = os.path.join(METRICS, "gate7b3r_curve.csv")
    for o in OBJECTS:
        ctx = prepare(o)
        for sp in range(args.splits):
            r, c = run(o, sp, ctx)
            rows.extend(r)
            covs.extend(c)
            # persist after EVERY split: a crash on the last object used to
            # discard the whole run
            pd.DataFrame(rows).to_csv(cpath, index=False)
            pd.DataFrame(covs).to_csv(
                os.path.join(METRICS, "gate7b3r_coverage.csv"), index=False)
            print(f"  {o} split {sp} done", flush=True)
    df = pd.DataFrame(rows)

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
