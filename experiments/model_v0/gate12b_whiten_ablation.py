# -*- coding: utf-8 -*-
"""
Gate 12B, steps 1-2 -- where does the whitening gain come from, and is it stable?

D1.5 gave +2.4 (11/15) over raw with a clean shuffled-context control (+2.6,
12/15 own minus shuffled). The proposed mechanism -- "the useful part of the
target context is the cross-dimensional covariance" -- is not yet established,
because centring and per-dimension scaling could account for some or all of it.

    A0  raw
    A1  centring             z - mu_t                    (no covariance at all)
    A2  diagonal z-score     (z - mu_t) / sigma_t       (marginal scales only)
    A3  full whitening       Sigma_t^-1/2 (z - mu_t)    (cross-dimensional)

Only A3 - A2 isolates the cross-dimensional structure. If A1/A2 already capture
most of the gain, the story is "recentre and rescale", not "canonicalise object
geometry", and Gate 12B's premise weakens.

Second: Sigma^-1/2 on 384 dims is ill-conditioned, and patches within one image
are not independent samples. So the same ablation is repeated across three
regularisation schemes, and A3 counts only if the gain holds across them:
    floor_eps   eigenvalues clipped at eps * mean eigenvalue
    shrink_lam  Sigma_lambda = (1-lam) Sigma + lam * mean_eig * I
    pca_k       whiten the top-k directions, project the rest away

Usage: python experiments/model_v0/gate12b_whiten_ablation.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import Index, l2norm  # noqa: E402
from gate12_conditional import recall_and_health  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
DEV = "cuda"


# ------------------------------------------------------------------ transforms
def stats(normal):
    X = normal.numpy().astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    return mu, C


def eig(C):
    w, V = np.linalg.eigh(C)
    return np.clip(w, 1e-12, None), V


def build(mu, C, mode, param):
    """Return (mu, W) with the transform z -> (z - mu) @ W.T."""
    w, V = eig(C)
    mw = w.mean()
    if mode == "center":
        return mu, np.eye(len(mu))
    if mode == "zscore":
        return mu, np.diag(1.0 / np.sqrt(np.diag(C)))
    if mode == "floor":
        wi = 1.0 / np.sqrt(np.clip(w, param * mw, None))
        return mu, (V * wi) @ V.T
    if mode == "shrink":
        wl = (1 - param) * w + param * mw
        return mu, (V * (1.0 / np.sqrt(wl))) @ V.T
    if mode == "pca":
        k = int(param)
        keep = np.argsort(w)[::-1][:k]
        wi = np.zeros_like(w)
        wi[keep] = 1.0 / np.sqrt(w[keep])
        return mu, (V * wi) @ V.T
    raise ValueError(mode)


@torch.no_grad()
def apply(E0, g, Zg, mu, W):
    Wt = torch.from_numpy(W.astype(np.float32))
    mut = torch.from_numpy(mu.astype(np.float32))
    out = E0.clone()
    out[g] = l2norm((Zg - mut.to(Zg.device)) @ Wt.to(Zg.device).T).cpu()
    return out


def main():
    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    Z = idx.Z.to(DEV)
    E0 = l2norm(idx.Z)
    rng = np.random.default_rng(0)

    # precompute per-object normal statistics once
    S = {}
    for c in files:
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float32)
        if len(nm) > 20000:
            nm = nm[rng.choice(len(nm), 20000, replace=False)]
        S[c] = stats(torch.from_numpy(nm))

    CONFIGS = [("A0 raw", None, None, None),
               ("A1 center", "center", None, None),
               ("A2 zscore", "zscore", None, None),
               ("A3 floor.01", "floor", 0.01, None),
               ("A3 floor.05", "floor", 0.05, None),
               ("A3 floor.10", "floor", 0.10, None),
               ("A3 floor.20", "floor", 0.20, None),
               ("A3 shrink.01", "shrink", 0.01, None),
               ("A3 shrink.10", "shrink", 0.10, None),
               ("A3 shrink.50", "shrink", 0.50, None),
               ("A3 pca64", "pca", 64, None),
               ("A3 pca128", "pca", 128, None),
               ("A3 pca256", "pca", 256, None)]

    rows = []
    for T in files:
        oi = files.index(T)
        g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
        mu, C = S[T]
        for name, mode, param, _ in CONFIGS:
            if mode is None:
                E = E0
            else:
                m2, W2 = build(mu, C, mode, param)
                E = apply(E0, g, Z[g], m2, W2)
            h = recall_and_health(idx, oi, E, None)
            rows.append({"target": T, "config": name, "recall@1": h["recall@1"],
                         "eff_rank": h["eff_rank"]})
        print(f"  {T:<12} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate12b_ablation.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate12b_ablation.csv"), index=False)
    P = R.pivot(index="target", columns="config", values="recall@1")
    print("\n" + "=" * 104)
    print("WHITENING ABLATION -- cross-image same-defect Recall@1")
    print("=" * 104)
    print(P.round(1).to_string())

    base = P["A0 raw"]
    print("\n" + "=" * 104)
    print("GAIN OVER RAW")
    print("=" * 104)
    print(f"  {'config':<15}{'mean':>8}{'median':>9}{'better':>9}   "
          f"<- only A3-A2 isolates cross-dimensional structure")
    for name, _, _, _ in CONFIGS[1:]:
        d = P[name] - base
        print(f"  {name:<15}{d.mean():>+8.2f}{d.median():>+9.2f}"
              f"{int((d > 0).sum()):>7}/15")
    d3 = P["A3 floor.05"] - P["A2 zscore"]
    print(f"\n  A3 floor.05 - A2 zscore: mean {d3.mean():+.2f}  "
          f"better {int((d3 > 0).sum())}/15   <- the covariance effect itself")
    print(f"\n  A3 stability across regularisation: "
          f"min {P[[c for c in P.columns if c.startswith('A3')]].mean().min():.1f}  "
          f"max {P[[c for c in P.columns if c.startswith('A3')]].mean().max():.1f}")


if __name__ == "__main__":
    main()
