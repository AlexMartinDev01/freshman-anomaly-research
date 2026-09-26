# -*- coding: utf-8 -*-
"""
Gate 15A -- Target-Normal Local Covariance Mixture. No training, no source data.

Thirteen gates have produced a consistent split. Everything learned from SOURCE
defect labels hurts an unseen target, twice now even after conditioning
(13B shared residual 84.8-85.8, 14B conditioned 82.0, against the 86.7
covariance baseline). Everything derived from the TARGET's own normal patches
helps, without any labels or training: 84.7 -> 86.7 with gamma = 0.5, and the
same-protocol oracle rises with it, 89.8 -> 91.8. The learned-residual branch is
therefore closed, and the question becomes whether the target normal distribution
contains structure richer than a single global covariance.

The most obvious such structure is multi-modality. A single Sigma_t averages
over metal/plastic/edge/hole/connector regions that plausibly occupy different
normal modes, and that averaging is a candidate source of the remaining 5.1
points.

    L0  raw DINO                            84.7
    L1  global covariance, gamma=0.5        86.7   <- the baseline to beat
    L2  local covariance, K=2
    L3  local covariance, K=4
    L4  local covariance, K=8
    L5  local covariance, soft assignment

A query patch is assigned to its nearest normal cluster centroid and then scored
with THAT cluster's fractional covariance:
    f(z) = normalize( (z - mu_k) (Sigma_k + eps I)^(-gamma/2) )
So the metric becomes not just object-specific but patch-condition-specific.

Two things are load-bearing:
  - LOCAL COVARIANCES ARE NOT ESTIMATED RAW. A cluster holds far fewer patches
    than the object, and 384 dims from a few thousand samples is badly
    conditioned. Shrink toward the GLOBAL covariance, which is already known to
    work:  Sigma*_k = (1-rho) Sigma_k + rho Sigma_t.  rho=1 IS L1, rho=0 is fully
    local.
  - THE COUNTERFACTUAL: keep every patch's cluster ASSIGNMENT but permute the
    local covariances among clusters. If that costs nothing, the pairing between
    a normal mode and its covariance is not doing any work -- the same test that
    validated global whitening (own +2.4 vs shuffled -0.3).

Pre-registered BEFORE the run (baseline is L1 = 86.7, not 84.7):
    Basic PASS   local > 86.7 with mean gain >= +1.0 and >= 10/15 targets
    Strong PASS  >= 12/15, mean >= +1.5, AND shuffled-local clearly below
                 correct-local
    FAIL         local <= 86.7 -> stop adding statistics on the final-layer
                 normal distribution; go to multi-layer DINO statistics

Usage: python experiments/model_v0/gate15a_local_mixture.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import Index, l2norm  # noqa: E402
from gate13b_shared_residual import recall_one  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
DEV = "cuda"
GAMMA = 0.5
EPS_FRAC = 0.05
KS = [2, 4, 8]
RHOS = [0.25, 0.5, 0.75]
NORMAL_CAP = 20000
N_QUERY = 150


def normal_patches(idx, c, rng, cap=NORMAL_CAP):
    nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float32)
    if len(nm) > cap:
        nm = nm[rng.choice(len(nm), cap, replace=False)]
    return nm


def frac_matrix(X):
    """(Sigma + eps I)^(-gamma/2) with the eigenvalue floor used throughout."""
    mu = X.mean(0).astype(np.float64)
    Xc = X.astype(np.float64) - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    return mu, C


def frac_pow(C, gamma=GAMMA, eps_frac=EPS_FRAC):
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, eps_frac * w.mean(), None)
    return ((V * w ** (-gamma / 2.0)) @ V.T).astype(np.float32)


def local_bank(idx, c, K, rho, rng):
    """Cluster the object's normal patches; shrink each cluster covariance
    toward the global one. Returns centroids and per-cluster (mu, W)."""
    X = normal_patches(idx, c, rng)
    gmu, gC = frac_matrix(X)
    km = MiniBatchKMeans(n_clusters=K, n_init=5, batch_size=2048,
                         random_state=0).fit(X)
    cents, mus, Ws = [], [], []
    for k in range(K):
        m = km.labels_ == k
        if m.sum() < 50:                     # too small to estimate anything
            mus.append(gmu)
            Ws.append(frac_pow(gC))
            cents.append(km.cluster_centers_[k])
            continue
        mu_k, C_k = frac_matrix(X[m])
        C_s = (1 - rho) * C_k + rho * gC
        mus.append(mu_k)
        Ws.append(frac_pow(C_s))
        cents.append(km.cluster_centers_[k])
    return (np.stack(cents).astype(np.float32), np.stack(mus).astype(np.float32),
            np.stack(Ws))


@torch.no_grad()
def embed_local(idx, oi, bank, soft=False, perm=None):
    """Assign each patch to a normal mode, apply that mode's metric."""
    cents, mus, Ws = bank
    if perm is not None:
        Ws = Ws[perm]                       # permute covariances, keep assignments
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    Z = idx.Z[g]
    d = torch.cdist(Z, torch.from_numpy(cents))
    if not soft:
        k = d.argmin(1).numpy()
        out = torch.stack([((Z[i] - torch.from_numpy(mus[k[i]]))
                           @ torch.from_numpy(Ws[k[i]]).T)
                           for i in range(len(Z))])
    else:
        w = torch.softmax(-d / (d.std() + 1e-6), dim=1).numpy()
        out = torch.zeros(len(Z), 384)
        for k in range(len(cents)):
            out += torch.from_numpy(w[:, k:k + 1].astype(np.float32)) * (
                (Z - torch.from_numpy(mus[k])) @ torch.from_numpy(Ws[k]).T)
    E = l2norm(out)
    O = torch.zeros(len(idx.Z), E.shape[1])
    O[g] = E
    return O


def main():
    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    rng = np.random.default_rng(0)

    rows = []
    W0 = l2norm(idx.Z)
    for T in files:
        oi = files.index(T)
        rows.append({"target": T, "config": "L0_raw",
                     "recall@1": recall_one(idx, W0, oi)})
    print("  L0 done", flush=True)

    # L1: the global covariance baseline, recomputed here so every config is
    # scored by THIS script's recall_one
    banks = {}
    for T in files:
        oi = files.index(T)
        c, mu, W = local_bank(idx, T, 1, 1.0, np.random.default_rng(0))
        banks[(T, 1, 1.0)] = (c, mu, W)
        rows.append({"target": T, "config": "L1_global",
                     "recall@1": recall_one(idx, embed_local(idx, oi,
                                                            (c, mu, W)), oi)})
    print("  L1 done", flush=True)

    for K in KS:
        for rho in RHOS:
            for T in files:
                oi = files.index(T)
                b = local_bank(idx, T, K, rho, np.random.default_rng(0))
                banks[(T, K, rho)] = b
                rows.append({"target": T, "config": f"L{K}_rho{rho}",
                             "recall@1": recall_one(idx, embed_local(idx, oi, b), oi)})
                if rho == 0.5:
                    perm = np.random.default_rng(1).permutation(K)
                    rows.append({"target": T, "config": f"L{K}_shuf",
                                 "recall@1": recall_one(
                                     idx, embed_local(idx, oi, b, perm=perm), oi)})
            print(f"  K={K} rho={rho} done", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(METRICS, "gate15a_local.csv"), index=False)

    for T in files:                       # L5 soft assignment at rho=0.5, K=4
        oi = files.index(T)
        b = banks[(T, 4, 0.5)]
        rows.append({"target": T, "config": "L5_soft",
                     "recall@1": recall_one(idx,
                                            embed_local(idx, oi, b, soft=True), oi)})
    print("  L5 done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate15a_local.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")

    print("\n" + "=" * 110)
    print("GATE 15A -- target-normal local covariance mixture (cross-image Recall@1)")
    print("=" * 110)
    print(P.round(1).to_string())

    base = P["L1_global"]
    print(f"\n  {'config':<16}{'mean':>8}{'vs L1':>9}{'better':>9}")
    for c in P.columns:
        if c in ("L0_raw", "L1_global"):
            continue
        d = P[c] - base
        print(f"  {c:<16}{P[c].mean():>8.1f}{d.mean():>+9.2f}"
              f"{int((d > 0).sum()):>7}/15")
    print(f"\n  L0 raw          {P['L0_raw'].mean():.1f}")
    print(f"  L1 global cov   {P['L1_global'].mean():.1f}  <- baseline to beat")

    print("\n" + "=" * 110)
    print("PRE-REGISTERED (baseline L1 = 86.7, not L0 = 84.7)")
    print("=" * 110)
    best, bestv = None, -1e9
    for c in P.columns:
        if c.startswith("L2") or c.startswith("L3") or c.startswith("L4") \
                or c == "L5_soft":
            if c.endswith("shuf"):
                continue
            if P[c].mean() > bestv:
                best, bestv = c, P[c].mean()
    d = P[best] - base
    ok = d.mean() >= 1.0 and int((d > 0).sum()) >= 10
    print(f"  best local config: {best}  mean {bestv:.1f}  ({d.mean():+.2f} vs L1, "
          f"{int((d > 0).sum())}/15)")
    print(f"  Basic PASS (>=+1.0 and >=10/15): {'PASS' if ok else 'FAIL'}")
    for K in KS:
        cs, cw = f"L{K}_shuf", f"L{K}_rho0.5"
        if cs in P and cw in P:
            ds = P[cw] - P[cs]
            print(f"  K={K}: correct - shuffled = {ds.mean():+.2f} "
                  f"({int((ds > 0).sum())}/15)")


if __name__ == "__main__":
    main()
