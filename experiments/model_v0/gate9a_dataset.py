# -*- coding: utf-8 -*-
"""
Gate 9A -- dataset for the learned Target-Conditioned Defect Selector.

The question Gate 8 left open: external defect patches carry transferable,
defect-specific information, but WHICH external class helps a given target is
not predictable from geometry (within-target rho = -0.04 for the normal-manifold
criterion; +0.29 with the wrong sign for an oracle defect-manifold criterion).
So this builds the supervised version of that question:

  given  (target's normal appearance)  x  (a candidate external defect bank)
  predict  how much that bank would improve the target's detection

Each row is one candidate bank for one target episode:
  h_target   features of the target's normal patches
  h_bank     features of the candidate defect bank
  utility    AUROC of the raw fusion S_n - S_d using exactly that bank

Protocol decisions, and why:
  - The normal bank used for S_n IS the shot set. The selector then sees exactly
    the evidence the detector has, and nothing about the target's own defects is
    used to build any bank. Eval covers ALL of the target's test images, which is
    legitimate precisely because no target defect ever enters S_d.
  - Every candidate bank for a given episode shares one eval set, so the
    utilities being ranked are on a common footing.
  - MVTec AD v1 is the training pool (15 classes, 210 ordered pairs) and AD2 is
    the held-out dataset (8 classes). Using v1 for training means NO AD2 class
    ever serves as a bank during fitting.

Writes results/model_v0/metrics/selector_<dataset>.npz

Usage: python experiments/model_v0/gate9a_dataset.py --dataset v1
       python experiments/model_v0/gate9a_dataset.py --dataset ad2
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate7b3r import make_dist  # noqa: E402
from models.tail_adapter import l2norm  # noqa: E402
from tail_calib import RESULTS, materialize, mean_top1p  # noqa: E402

AD2_CACHE = os.path.join(RESULTS, "cache")
V1_CACHE = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
POOLS = os.path.join(RESULTS, "selector_pools")
SHOTS = [1, 2, 4, -1]
SEEDS = [0, 1, 2]
SIZES = [200, 500, 1000]
DEV = "cuda"


def cache_dir(ds):
    return V1_CACHE if ds == "v1" else AD2_CACHE


def classes(ds):
    d = cache_dir(ds)
    return sorted(f[:-len("_test.npz")] for f in os.listdir(d)
                  if f.endswith("_test.npz"))


def load(ds, obj, split):
    return materialize(np.load(
        os.path.join(cache_dir(ds), f"{obj}_{split}.npz"), allow_pickle=True))


def defect_pool(ds, obj):
    """All of `obj`'s defect patches (gt_frac > 0.10), cached to a compact npy."""
    os.makedirs(POOLS, exist_ok=True)
    p = os.path.join(POOLS, f"{ds}_{obj}_defect.npy")
    if os.path.exists(p):
        return np.load(p)
    te = load(ds, obj, "test")
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
             for i in range(len(types))
             if types[i] == "bad" and (gt[i] > 0.10).sum()]
    pool = np.concatenate(parts) if parts else np.zeros((0, 384), np.float32)
    np.save(p, pool)
    return pool


def spread_of(X, seed=0, n=400):
    """Mean 1-NN cosine distance to the nearest OTHER patch in the set.

    Must exclude self-matches: a distance-to-the-same-bank call would return ~0
    for every patch and measure nothing.
    """
    if len(X) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    sub = X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]
    t = l2norm(torch.from_numpy(sub.astype(np.float32)).to(DEV))
    with torch.no_grad():
        D = 1.0 - t @ t.T
    D.fill_diagonal_(float("inf"))
    return float(D.min(dim=1).values.mean().item())


def target_feats(shot_patches, n_shot):
    X = shot_patches
    mu, sd = X.mean(0), X.std(0)
    scal = np.array([np.log1p(len(X)), spread_of(X), np.log1p(max(n_shot, 1))],
                    dtype=np.float32)
    return np.concatenate([mu, sd, scal]).astype(np.float32)


def bank_feats(bank):
    mu, sd = bank.mean(0), bank.std(0)
    scal = np.array([np.log1p(len(bank)), spread_of(bank)], dtype=np.float32)
    return np.concatenate([mu, sd, scal]).astype(np.float32)


def build(ds):
    os.makedirs(METRICS, exist_ok=True)
    objs = classes(ds)
    pools = {o: defect_pool(ds, o) for o in objs}
    print(f"  {ds}: {len(objs)} classes; defect pools "
          f"{ {o: len(p) for o, p in pools.items()} }", flush=True)

    H_T, H_B, UT, META = [], [], [], []
    for S in objs:
        tr, te = load(ds, S, "train"), load(ds, S, "test")
        types = list(te["types"])
        n_img = len(types)
        feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
                 for i in range(n_img)]
        y = np.array([0 if types[i] == "good" else 1 for i in range(n_img)])
        n_train = tr["feats"].shape[0]
        cands = [B for B in objs if B != S and len(pools[B])]

        for shot in SHOTS:
            for seed in SEEDS:
                rng_shot = np.random.default_rng(
                    (seed + 1) * 1000 + (shot if shot != -1 else 0))
                if shot == -1:
                    idx = np.arange(n_train)
                else:
                    idx = rng_shot.choice(n_train, size=min(shot, n_train),
                                          replace=False)
                shot_patches = tr["feats"][idx].reshape(-1, 384).astype(np.float32)
                tfeat = target_feats(shot_patches, shot if shot != -1 else n_train)
                nb = torch.from_numpy(shot_patches).to(DEV)
                with torch.no_grad():
                    dist_n = make_dist(nb, DEV)
                    Sn = [dist_n(z) for z in feats]
                base = roc_auc_score(y, [mean_top1p(d) for d in Sn]) * 100

                for B in cands:
                    pool = pools[B]
                    for size in SIZES:
                        # bank draw seeded by (B, size, seed) only, so the SAME
                        # banks are used at every shot level and the shot
                        # comparison is not confounded with bank resampling
                        rng_bank = np.random.default_rng(
                            hash((B, size, seed)) % 2 ** 31)
                        k = min(size, len(pool))
                        sel = rng_bank.choice(len(pool), size=k, replace=False)
                        bank = pool[sel]
                        dist_d = make_dist(torch.from_numpy(bank).to(DEV), DEV)
                        with torch.no_grad():
                            Sd = [dist_d(z) for z in feats]
                        sc = [mean_top1p(Sn[i] - Sd[i]) for i in range(n_img)]
                        util = roc_auc_score(y, sc) * 100
                        H_T.append(tfeat)
                        H_B.append(bank_feats(bank))
                        UT.append(util)
                        META.append((S, B, shot, seed, size, base))
                print(f"    {S:<14} shot={shot:<3} seed={seed} "
                      f"baseline={base:5.1f}  ({len(cands)} banks done)",
                      flush=True)

    out = os.path.join(METRICS, f"selector_{ds}.npz")
    np.savez_compressed(out, h_t=np.stack(H_T), h_b=np.stack(H_B),
                        utility=np.array(UT, dtype=np.float32),
                        meta=np.array(META, dtype=object))
    print(f"\nwrote {out}  rows={len(UT)}  h_t={np.stack(H_T).shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["v1", "ad2"], required=True)
    a = ap.parse_args()
    build(a.dataset)
