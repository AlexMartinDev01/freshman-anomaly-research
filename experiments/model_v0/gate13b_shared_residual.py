# -*- coding: utf-8 -*-
"""
Gate 13B -- does a SHARED structured residual add anything on top of the target
covariance metric?

Gate 13A found a Pareto point at gamma = 0.5: baseline 84.7 -> 86.7 AND oracle
89.8 -> 91.8, both improving together, 12/15 targets better. So partial
covariance correction is the representation; the remaining question is whether
source defect supervision can add to it.

That is deliberately NOT the question Gate 11 asked. Gate 11 tested a shared
metric REPLACING the raw metric (M_t = M_shared, 79.6 vs 84.7). Gate 12B tested a
shared metric replacing the whitened metric (63.8 vs 87.1). Neither tested a
small shared residual ADDED to a target-specific covariance baseline:

    M_t = Sigma_t^{-gamma*}  +  U U^T          both in the RAW DINO basis
    f(z) = normalize( [ Sigma_t^{-gamma*/2}(z-mu_t) , U^T (z-mu_t) ] )

The covariance term is per-target and needs no labels; the shared term carries
whatever defect directions transfer. E3 (the gamma* oracle, 91.8) bounds what is
achievable when the residual may be fitted with target labels.

Pre-registered (from the gate plan):
    PASS  E2 > E1 with a clear margin on a majority of targets, and E2 > E0
    FAIL  E2 <= E1 while E3 >> E1 -> the residual defect correction is itself
          object-dependent -> covariance-conditioned residual (Gate 14)

Health is instrumented and a run with update_norm ~0 is INVALID, never FAIL.

Usage: python experiments/model_v0/gate13b_shared_residual.py [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import Index, l2norm  # noqa: E402
from gate11_transfer import Sub  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g13b")
DEV = "cuda"
GAMMA = 0.5
EPS_FRAC = 0.05
RANKS = [4, 8, 16]


class Residual(nn.Module):
    """Shared, rank-r, defined in the RAW DINO basis."""

    def __init__(self, d=384, r=8):
        super().__init__()
        self.U = nn.Linear(d, r, bias=False)
        nn.init.normal_(self.U.weight, std=0.05)   # not zero: bilinear saddle

    def forward(self, zc):
        return self.U(zc)


def covariance_parts(idx, gamma=GAMMA, cap=20000, eps_frac=EPS_FRAC, seed=0):
    """Per-object centred features and their (Sigma+epsI)^(-gamma/2) image.

    Computed once: they are target properties, not model parameters.
    """
    rng = np.random.default_rng(seed)
    centred = torch.zeros_like(idx.Z)
    wpart = torch.zeros_like(idx.Z)
    for c in idx.classes:
        oi = idx.classes.index(c)
        g = np.where(idx.OBJ == oi)[0]
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float64)
        if len(nm) > cap:
            nm = nm[rng.choice(len(nm), cap, replace=False)]
        mu = nm.mean(0)
        Xc = nm - mu
        C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps_frac * w.mean(), None)
        W = (V * w ** (-gamma / 2.0)) @ V.T
        zc = idx.Z[g] - torch.from_numpy(mu.astype(np.float32))
        centred[g] = zc
        wpart[g] = zc @ torch.from_numpy(W.astype(np.float32)).T
    return centred, wpart


def train_residual(idx, centred, wpart, src_objs, rank, steps=2000, lr=1e-3,
                   seed=0):
    torch.manual_seed(seed)
    R = Residual(r=rank).to(DEV)
    opt = torch.optim.AdamW(R.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    sub = Sub(idx, [idx.classes.index(c) for c in src_objs], "type")
    Cc, Wp = centred.to(DEV), wpart.to(DEV)
    init = R.U.weight.detach().clone()
    gn, active = [], []
    for _ in range(steps):
        if not sub.keys:
            break
        cols = []
        for _ in range(48):
            k = sub.keys[rng.integers(len(sub.keys))]
            cand = sub.by_ot[k]
            a = cand[rng.integers(len(cand))]
            alt = [i for i in cand if sub.IMG[i] != sub.IMG[a]]
            o = sub.OBJ[a]
            oth = np.where((sub.OBJ == o) & (sub.TYPE != sub.TYPE[a]))[0]
            if not alt or not len(oth):
                continue
            cols.append((a, alt[rng.integers(len(alt))],
                         oth[rng.integers(len(oth))]))
        if not cols:
            break
        sel = torch.as_tensor(np.array(cols).T.ravel()).to(DEV)
        f = l2norm(torch.cat([Wp[sel], R(Cc[sel])], dim=1))
        n = len(cols)
        d_ap = 1 - (f[:n] * f[n:2 * n]).sum(1)
        d_an = 1 - (f[:n] * f[2 * n:]).sum(1)
        loss = F.softplus(d_ap - d_an).mean()
        opt.zero_grad(); loss.backward()
        gn.append(float(torch.nn.utils.clip_grad_norm_(R.parameters(), 1e9)))
        active.append(float((d_an > d_ap).float().mean()))
        opt.step()
    R.eval()
    return R, {"grad_norm": float(np.mean(gn)) if gn else 0.0,
               "update_norm": float((R.U.weight - init).norm()),
               "active_triplet": float(np.mean(active)) if active else 0.0}


@torch.no_grad()
def embed_with(idx, centred, wpart, oi, R=None):
    """Full index-space tensor: recall_one indexes with original ids, so
    returning only the object's rows silently mixes index spaces."""
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    parts = [wpart[g]]
    if R is not None:
        parts.append(R(centred[g].to(DEV)).cpu())
    E = l2norm(torch.cat(parts, dim=1))
    out = torch.zeros(len(idx.Z), E.shape[1])
    out[g] = E
    return out


@torch.no_grad()
def recall_one(idx, E, oi, n_query=150, seed=0):
    rng = np.random.default_rng(seed)
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    # draw ALL queries up front. Interleaving other rng calls between draws
    # advances the stream, so two scripts with "the same" sampling rule end up
    # scoring different query patches -- that alone moved E1 by ~5 AUROC.
    q = rng.choice(g, size=min(n_query, len(g)), replace=False)
    hits = []
    for i in q:
        gal = g[idx.IMG[g] != idx.IMG[i]]
        if len(gal) < 10:
            continue
        order = np.argsort(-(E[gal] @ E[i]).numpy())
        hits.append((idx.TYPE[gal][order] == idx.TYPE[i])[:1].mean())
    return float(np.mean(hits)) * 100 if hits else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    centred, wpart = covariance_parts(idx)
    OI = {c: i for i, c in enumerate(files)}

    # E1 = covariance only; E0 = raw
    E0 = l2norm(idx.Z)
    rows = []
    for T in files:
        oi = OI[T]
        rows.append({"target": T, "config": "E0", "recall@1":
                     recall_one(idx, E0, oi)})
        # l2norm here: `wpart` is the raw whitened image (row norms ~45), and
        # recall_one ranks by plain dot product, which on unnormalised vectors
        # is dominated by magnitude rather than direction -- that alone cost
        # E1 about 5 AUROC against gate 13A's identical representation.
        rows.append({"target": T, "config": "E1",
                     "recall@1": recall_one(idx, l2norm(wpart), oi)})

    for rank in RANKS:
        src = [c for c in files if True]
        for T in files:
            oi = OI[T]
            tr_src = [c for c in files if c != T]
            R, h = train_residual(idx, centred, wpart, tr_src, rank, a.steps)
            torch.save(R.state_dict(), os.path.join(CKPT, f"{T}_r{rank}.pt"))
            E = embed_with(idx, centred, wpart, oi, R)
            rows.append({"target": T, "config": f"E2_r{rank}",
                         "recall@1": recall_one(idx, E, oi),
                         **h})
            if rank == RANKS[0]:
                print(f"  {T:<12} E1 {rows[-1-0]['recall@1'] if False else ''}"
                      f" E2_r{rank} {rows[-1]['recall@1']:5.1f}  "
                      f"update_norm {h['update_norm']:.3f}  "
                      f"active {h['active_triplet']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate13b_shared_residual.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate13b_shared_residual.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")

    print("\n" + "=" * 100)
    print("GATE 13B -- covariance (gamma=0.5) + shared structured residual")
    print("=" * 100)
    print(P.round(1).to_string())
    print("\n  means: " + "  ".join(f"{c} {P[c].mean():.1f}" for c in P.columns))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED")
    print("=" * 100)
    for rank in RANKS:
        c = f"E2_r{rank}"
        d1 = P[c] - P["E1"]
        d0 = P[c] - P["E0"]
        print(f"  {c}: E1 {P['E1'].mean():.1f} -> {P[c].mean():.1f}  "
              f"(vs E1 {d1.mean():+.2f}, better {int((d1 > 0).sum())}/15;  "
              f"vs E0 {d0.mean():+.2f}, better {int((d0 > 0).sum())}/15)")
    H = R.dropna(subset=["update_norm"])
    if len(H):
        print(f"\n  health: update_norm mean {H.update_norm.mean():.3f} "
              f"min {H.update_norm.min():.3f}  |grad| {H.grad_norm.mean():.4f}  "
              f"active {H.active_triplet.mean():.3f}")
        print(f"  INVALID runs (update_norm < 0.01): "
              f"{int((H.update_norm < 0.01).sum())}/{len(H)}")


if __name__ == "__main__":
    main()
