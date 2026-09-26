# -*- coding: utf-8 -*-
"""
Gate 13A -- Partial (fractional) covariance correction. Zero learning.

Gate 12B left two facts that point in opposite directions:

    full whitening   baseline 84.7 -> 87.1   (good: removes object nuisance)
    full whitening   oracle   92.6 -> 90.8   (bad: also removes real structure)

So whitening is not a lossless canonicalisation -- it buys the unsupervised
baseline by spending learnable headroom, and stacking a shared metric on it
collapses (W3 63.8 vs W1 87.1). The natural question is whether full whitening
over-corrects, and whether some intermediate strength keeps both.

    M_t(gamma) = (Sigma_t + eps I)^(-gamma)      gamma in [0, 1]
        gamma = 0   raw DINO
        gamma = 1   full whitening

No training is involved at gamma < 1 for the baseline. What matters is whether
some gamma gives baseline up AND oracle flat -- a Pareto point where nuisance is
removed without destroying transferable defect directions. If one exists, the
representation interface is settled before any model is trained.

Both curves are reported per gamma, plus effective rank and per-object
consistency, because a mean improvement driven by two objects is not a sweet spot.

Usage: python experiments/model_v0/gate13a_fractional.py [--steps N]
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
from gate10b_structured import Index, ProjXY, l2norm  # noqa: E402
from gate11_transfer import Sub  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g13a")
DEV = "cuda"
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS_FRAC = 0.05


# ------------------------------------------------------------------ transforms
def fractional(idx, gamma, cap=20000, eps_frac=EPS_FRAC, seed=0):
    """(Sigma+epsI)^(-gamma/2) applied per object, then L2-normalised.

    gamma = 0 returns raw DINO untouched (no centring either): centring alone
    was measured at -0.58 in Gate 12B, so it is not doing work and folding it in
    would make gamma=0 disagree with the established raw baseline.
    """
    if gamma == 0.0:
        return l2norm(idx.Z)
    rng = np.random.default_rng(seed)
    Zt = torch.zeros_like(idx.Z)
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
        Zt[g] = l2norm((idx.Z[g] - torch.from_numpy(mu.astype(np.float32)))
                       @ torch.from_numpy(W.astype(np.float32)).T)
    return Zt


# -------------------------------------------------------------------- training
def train_shared(idx, Zs, src_objs, steps, lr=1e-3, seed=0):
    """softplus triplet on the shared representation; health is instrumented."""
    torch.manual_seed(seed)
    P = ProjXY(use_xy=False).to(DEV)
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    sub = Sub(idx, [idx.classes.index(c) for c in src_objs], "type")
    Zdev = Zs.to(DEV)
    init = {k: v.detach().clone() for k, v in P.state_dict().items()}
    gnorm, active = [], []
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
        h = l2norm(P(Zdev[sel]))
        n = len(cols)
        d_ap = 1 - (h[:n] * h[n:2 * n]).sum(1)
        d_an = 1 - (h[:n] * h[2 * n:]).sum(1)
        loss = F.softplus(d_ap - d_an).mean()
        opt.zero_grad(); loss.backward()
        gnorm.append(float(torch.nn.utils.clip_grad_norm_(P.parameters(), 1e9)))
        active.append(float((d_an > d_ap).float().mean()))
        opt.step()
    delta = sum(float((P.state_dict()[k] - init[k]).norm()) for k in init)
    P.eval()
    return P, {"grad_norm": float(np.mean(gnorm)) if gnorm else 0.0,
               "update_norm": delta,
               "active_triplet": float(np.mean(active)) if active else 0.0}


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def recall_one(idx, E, oi, n_query=150, seed=0):
    rng = np.random.default_rng(seed)
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    hits, ap, an = [], [], []
    by_ot = {}
    for i in g:
        by_ot.setdefault(idx.TYPE[i], []).append(i)
    of_obj = np.where(idx.OBJ == oi)[0]
    for i in rng.choice(g, size=min(n_query, len(g)), replace=False):
        gal = g[idx.IMG[g] != idx.IMG[i]]
        if len(gal) < 10:
            continue
        order = np.argsort(-(E[gal] @ E[i]).numpy())
        hits.append((idx.TYPE[gal][order] == idx.TYPE[i])[:1].mean())
        alt = [j for j in by_ot[idx.TYPE[i]] if idx.IMG[j] != idx.IMG[i]]
        oth = of_obj[idx.TYPE[of_obj] != idx.TYPE[i]]
        if alt and len(oth):
            ap.append(1 - float(E[i] @ E[alt[rng.integers(len(alt))]]))
            an.append(1 - float(E[i] @ E[oth[rng.integers(len(oth))]]))
    if not hits:
        return {}
    G = E[g].numpy().astype(np.float64)
    Gc = G - G.mean(0, keepdims=True)
    sv = np.linalg.svd(Gc, compute_uv=False) ** 2
    sv = sv / sv.sum()
    return {"recall@1": float(np.mean(hits)) * 100,
            "eff_rank": float(np.exp(-(sv * np.log(sv + 1e-12)).sum())),
            "d_ap": float(np.mean(ap)) if ap else np.nan,
            "d_an": float(np.mean(an)) if an else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)

    rows, health = [], []
    for gamma in GAMMAS:
        Zg = fractional(idx, gamma)
        P, h = train_shared(idx, Zg, files, a.steps)     # oracle: all objects
        torch.save(P.state_dict(), os.path.join(CKPT, f"gamma{gamma}.pt"))
        with torch.no_grad():
            Eo = l2norm(P(Zg.to(DEV))).cpu()
        for T in files:
            oi = files.index(T)
            for cfg, E in [("base", Zg), ("oracle", Eo)]:
                r = recall_one(idx, E, oi)
                r.update({"target": T, "gamma": gamma, "config": cfg})
                rows.append(r)
        health.append({"gamma": gamma, **h})
        b = np.mean([r["recall@1"] for r in rows
                     if r["gamma"] == gamma and r["config"] == "base"])
        o = np.mean([r["recall@1"] for r in rows
                     if r["gamma"] == gamma and r["config"] == "oracle"])
        print(f"  gamma={gamma:<5} baseline {b:5.1f}   oracle {o:5.1f}   "
              f"health update_norm {h['update_norm']:.2f} "
              f"active {h['active_triplet']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate13a_fractional.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate13a_fractional.csv"), index=False)
    B = R[R.config == "base"].pivot(index="target", columns="gamma",
                                    values="recall@1")
    O = R[R.config == "oracle"].pivot(index="target", columns="gamma",
                                      values="recall@1")
    Q = R[R.config == "base"].pivot(index="target", columns="gamma",
                                    values="eff_rank")

    print("\n" + "=" * 100)
    print("GATE 13A -- fractional covariance correction (baseline / oracle)")
    print("=" * 100)
    print("\n  BASELINE (no target defect supervision):")
    print(B.round(1).to_string())
    print("\n  ORACLE (target-supervised metric in the same representation):")
    print(O.round(1).to_string())
    print("\n  effective rank of the baseline representation:")
    print(Q.round(1).to_string())

    print("\n" + "=" * 100)
    print("PARETO VIEW  (does any gamma raise baseline without dropping oracle?)")
    print("=" * 100)
    print(f"  {'gamma':<8}{'baseline':>10}{'oracle':>9}{'base-raw':>10}"
          f"{'oracle-raw':>12}{'eff_rank':>10}{'n_base>raw':>12}")
    b0, o0 = B[0.0].mean(), O[0.0].mean()
    for g in GAMMAS:
        d = B[g] - B[0.0]
        print(f"  {g:<8}{B[g].mean():>10.1f}{O[g].mean():>9.1f}"
              f"{B[g].mean()-b0:>+10.2f}{O[g].mean()-o0:>+12.2f}"
              f"{Q[g].mean():>10.1f}{int((d > 0).sum()):>9}/15")
    print(f"\n  raw reference: baseline {b0:.1f}  oracle {o0:.1f}")
    pd.DataFrame(health).to_csv(os.path.join(METRICS, "gate13a_health.csv"),
                                index=False)


if __name__ == "__main__":
    main()
