# -*- coding: utf-8 -*-
"""
Gate 14A -- can a low-rank residual family even EXPRESS the remaining headroom?

Gate 13A froze the base metric at gamma = 0.5 (the balanced operating point, not
the unique Pareto point -- gamma=1 has a higher baseline and gamma=0.5 a higher
oracle, so neither dominates). Gate 13B then showed a shared residual cannot add
to it: E1 86.7, E2 84.8-85.8, while E3 (gamma* oracle) is 91.8.

Before training any conditioner Sigma_t -> U_t, there is a cheaper question that
must be answered first, because a failure here makes a conditioner pointless:

    if the target's OWN defect labels may fit the residual, can a rank-r
    M_t = Sigma_t^{-0.5} + U_t U_t^T actually reach the 91.8 oracle?

If a low-rank residual family cannot express the headroom even with target
labels, the problem is the model family, not the conditioning, and no
hypernetwork over Sigma_t will fix it.

Two follow-ups ride along, both label-free at test time:
  (2) do objects with similar covariance need similar residuals?
      log-Euclidean D_Sigma vs residual-subspace principal-angle distance.
  (3) covariance-nearest-neighbour residual transfer: borrow the residual of the
      source object whose covariance is closest, or a softmax-weighted mixture.
      This is target-conditioned with no training and no target defect labels.

Usage: python experiments/model_v0/gate14a_residual_oracle.py [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import Index, l2norm  # noqa: E402
from gate13b_shared_residual import (covariance_parts, embed_with,  # noqa: E402
                                     recall_one, train_residual)
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g14a")
DEV = "cuda"
GAMMA = 0.5
RANKS = [4, 8, 16]
E3 = 91.8                     # gamma* oracle from gate 13A, same sampler


def log_cov(idx, cap=20000, eps_frac=0.05, seed=0):
    """log Sigma per object, and the eigendecomposition, for the geometry study."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in idx.classes:
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float64)
        if len(nm) > cap:
            nm = nm[rng.choice(len(nm), cap, replace=False)]
        Xc = nm - nm.mean(0)
        C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps_frac * w.mean(), None)
        out[c] = (V * np.log(w)) @ V.T
    return out


def ortho(U):
    Q, _ = np.linalg.qr(U)
    return Q[:, :U.shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    centred, wpart = covariance_parts(idx, GAMMA)
    lcov = log_cov(idx)
    OI = {c: i for i, c in enumerate(files)}
    Wn = l2norm(wpart)

    rows, subs = [], {}
    E1_vals = {}
    for T in files:
        oi = OI[T]
        E1_vals[T] = recall_one(idx, Wn, oi)
        rows.append({"target": T, "config": "E1_cov", "recall@1": E1_vals[T]})

    # ---- (1) target-specific residual oracle ----
    for rank in RANKS:
        for T in files:
            oi = OI[T]
            R, h = train_residual(idx, centred, wpart, [T], rank, a.steps)
            torch.save(R.state_dict(), os.path.join(CKPT, f"{T}_r{rank}.pt"))
            U = R.U.weight.detach().cpu().numpy().T          # (384, rank)
            subs[(T, rank)] = ortho(U)
            E = embed_with(idx, centred, wpart, oi, R)
            rows.append({"target": T, "config": f"E1+U_r{rank}",
                         "recall@1": recall_one(idx, E, oi), **h})
        print(f"  rank {rank} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate14a_residual_oracle.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate14a_residual_oracle.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")
    print("\n" + "=" * 100)
    print("GATE 14A -- target-specific low-rank residual oracle (gamma=0.5 base)")
    print("=" * 100)
    print(P.round(1).to_string())
    print(f"\n  {'config':<14}{'mean':>8}{'vs E1':>9}{'better':>9}"
          f"{'gap to 91.8 closed':>22}")
    for c in P.columns:
        if c == "E1_cov":
            continue
        d = P[c] - P["E1_cov"]
        rec = (P[c].mean() - P["E1_cov"].mean()) / (E3 - P["E1_cov"].mean()) * 100
        print(f"  {c:<14}{P[c].mean():>8.1f}{d.mean():>+9.2f}"
              f"{int((d > 0).sum()):>7}/15{rec:>20.1f}%")
    print(f"\n  E1 (covariance only)  = {P['E1_cov'].mean():.1f}")
    print(f"  E3 (full oracle)      = {E3}")
    print(f"  headroom to explain   = {E3 - P['E1_cov'].mean():.1f}")

    # ---- (2) does covariance similarity predict residual similarity? ----
    print("\n" + "=" * 100)
    print("(2) COVARIANCE <-> RESIDUAL GEOMETRY")
    print("=" * 100)
    for rank in RANKS:
        ds, dr = [], []
        for i, A in enumerate(files):
            for B in files[i + 1:]:
                ds.append(np.linalg.norm(lcov[A] - lcov[B]))
                Pa, Pb = subs[(A, rank)], subs[(B, rank)]
                s = np.linalg.svd(Pa.T @ Pb, compute_uv=False)
                dr.append(float(np.mean(np.arccos(np.clip(s, -1, 1)))))
        ds, dr = np.array(ds), np.array(dr)
        r = np.corrcoef(ds, dr)[0, 1]
        rho = pd.Series(ds).corr(pd.Series(dr), method="spearman")
        print(f"  rank {rank:<3} n={len(ds)}  pearson {r:+.3f}  spearman {rho:+.3f}"
              f"   (positive => similar covariance wants similar residual)")

    # ---- (3) covariance-NN / weighted residual transfer, no training ----
    print("\n" + "=" * 100)
    print("(3) COVARIANCE-CONDITIONED RESIDUAL TRANSFER (label-free, no training)")
    print("=" * 100)
    rows3 = []
    for rank in RANKS:
        Un = {c: torch.from_numpy(subs[(c, rank)]).float().to(DEV)
              for c in files}
        for T in files:
            oi = OI[T]
            gd = torch.as_tensor(idx.defect_idx[idx.OBJ[idx.defect_idx] == oi])
            dist = {s: np.linalg.norm(lcov[T] - lcov[s])
                    for s in files if s != T}
            src = sorted(dist, key=dist.get)
            for tag, k in [("nn1", 1), ("nn3", 3), ("nn5", 5)]:
                U = torch.stack([Un[s] for s in src[:k]]).mean(0)   # (384, rank)
                extra = (centred[gd].to(DEV) @ U).cpu()             # (n, rank)
                E = torch.zeros(len(idx.Z), Wn.shape[1] + rank)
                E[gd] = l2norm(torch.cat([Wn[gd], extra], dim=1))
                rows3.append({"target": T, "rank": rank, "config": tag,
                              "recall@1": recall_one(idx, E, oi)})
        print(f"  rank {rank} done", flush=True)

    R3 = pd.DataFrame(rows3)
    R3.to_csv(os.path.join(METRICS, "gate14a_transfer.csv"), index=False)
    T3 = R3.pivot_table(index=["config", "rank"], values="recall@1")
    base = P["E1_cov"].mean()
    print(f"  E1 baseline = {base:.1f}")
    for (cfg, rank), r in T3.iterrows():
        print(f"  {cfg:<5} rank {rank:<3} mean {r['recall@1']:.1f}  "
              f"({r['recall@1'] - base:+.2f} vs E1)")
    bad = R.dropna(subset=["update_norm"])
    print(f"\n  health: update_norm mean {bad.update_norm.mean():.2f} "
          f"min {bad.update_norm.min():.2f}  active {bad.active_triplet.mean():.3f}")


def centered_apply(idx, centred, gd, U):
    """U^T (z - mu) in the raw shared basis."""
    return U.T @ centred[gd].to(DEV).T


if __name__ == "__main__":
    main()
