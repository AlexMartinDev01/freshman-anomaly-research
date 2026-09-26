# -*- coding: utf-8 -*-
"""
Gate 14B -- Covariance-conditioned residual metric.

Gate 14A established the three facts that make this gate worth running:
    (1) a target-fitted rank-16 residual closes 39% of the E1->E3 headroom
        (88.7 vs E1 86.7 and oracle 91.8), 10/15 targets better -- so the
        residual FAMILY can express what is missing;
    (2) objects with similar covariance need similar residuals, r = +0.435 at
        rank 16, p < 0.001 by permutation over objects;
    (3) naive covariance-nearest-neighbour residual borrowing fails badly
        (-10 to -17), which is a statement about wholesale transfer, not about
        whether covariance is informative.

So the model family is right and the conditioning signal exists. What remains is
to LEARN the map Sigma_t -> residual:

    M_t = Sigma_t^{-0.5}  +  sum_k alpha_k(C_t) B_k B_k^T

    P        fixed shared basis (D x kp) from SOURCE normals only, per fold
    C_t      P^T Sigma_t P, reduced from 384x384 to kp x kp
    alpha    conditioner h_theta, applied to log(C_t + eps I); SPD, so log-space
    B_k      K shared residual bases in the RAW DINO basis, rank r each

Counterfactuals, because a conditioner that ignores its input would pass a
plain "better than baseline" test:
    F0  covariance only                    86.7
    F1  correct covariance                 the model
    F2  shuffled covariance                another object's Sigma
    F3  constant context                   mean Sigma
F1 must beat F2 and F3 or the conditioning claim is void.

Pre-registered:
    PASS         F1 - F0 >= +1.0, >= 10/15, AND F1 > F2, AND F1 > F3
    Strong PASS  >= 12/15 and (F1 - F0)/(91.8 - 86.7) >= 30%
    FAIL         F1 <= F0 while F3 still trails -> covariance is not enough

Usage: python experiments/model_v0/gate14b_conditioner.py [--steps N]
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
from gate13b_shared_residual import covariance_parts, recall_one  # noqa: E402
from gate11_transfer import Sub  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g14b")
DEV = "cuda"
GAMMA = 0.5
KP = 32          # shared basis dimension for the covariance summary
K_BASIS = 4      # number of shared residual bases
RANK = 16        # rank of each basis
E3 = 91.8


def covariances(idx, cap=20000, eps_frac=0.05, seed=0):
    """Full 384x384 object covariances (needed for P^T Sigma P)."""
    rng = np.random.default_rng(seed)
    C = {}
    for c in idx.classes:
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float64)
        if len(nm) > cap:
            nm = nm[rng.choice(len(nm), cap, replace=False)]
        Xc = nm - nm.mean(0)
        C[c] = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    return C


def basis_from_sources(idx, src, kp=KP, cap=50000, seed=0):
    """Fixed shared basis P from SOURCE normals only -- never the target."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in src:
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float32)
        parts.append(nm[rng.choice(len(nm), min(len(nm), cap // len(src)),
                                   replace=False)])
    X = np.concatenate(parts).astype(np.float64)
    X = X - X.mean(0)
    # top-kp right singular directions
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:kp].T.astype(np.float32)          # (384, kp)


class Cond(nn.Module):
    """log(C_t + epsI) -> alpha in R^K. SPD, so log-space beats vectorising."""

    def __init__(self, kp=KP, K=K_BASIS, hid=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(kp * kp, hid), nn.ReLU(),
                                 nn.Linear(hid, K))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, C):
        x = matrix_log(C).reshape(len(C), -1)
        return F.softmax(self.net(x), dim=-1) * self.K


def matrix_log(C):
    """True matrix logarithm of an SPD matrix.

    torch.log is ELEMENTWISE. P^T Sigma P is PSD but has negative off-diagonal
    entries, so elementwise log returns NaN and the conditioner silently emits
    NaN alphas -- which then survive `0 * NaN = NaN` through the residual gate
    and leave every target scored on garbage.
    """
    C = 0.5 * (C + C.transpose(-1, -2))
    w, V = torch.linalg.eigh(C)
    w = w.clamp_min(1e-6)
    return (V * w.log().unsqueeze(-2)) @ V.transpose(-1, -2)


class Model(nn.Module):
    def __init__(self, kp=KP, K=K_BASIS, r=RANK):
        super().__init__()
        self.cond = Cond(kp, K)
        # B is small so `extra` (norm ~1) is dwarfed by the covariance block
        # (norm ~45): training therefore STARTS essentially at F0, but B still
        # receives a real gradient. gating with a learnable scalar initialised to
        # zero does NOT work -- d(extra)/dB = scale * (...) = 0 kills B's
        # gradient, and the run sits at F0 for 300 steps (scale stayed 0.0,
        # update_norm 0.004). Same bilinear-saddle family as gate 12's D3.
        self.B = nn.Parameter(torch.randn(K, 384, r) * 0.05)

    def forward(self, centred, wpart, C):
        a = self.cond(C)                                  # (n, K)
        extra = torch.einsum("nk,nkr->nr", a,
                             torch.einsum("nd,kdr->nkr", centred, self.B))
        return l2norm(torch.cat([wpart, extra], dim=1))


def train_fold(idx, centred, wpart, Cts, src, steps, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    M = Model().to(DEV)
    opt = torch.optim.AdamW(M.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    sub = Sub(idx, [idx.classes.index(c) for c in src], "type")
    Cc, Wp = centred.to(DEV), wpart.to(DEV)
    Ct = {c: torch.from_numpy(Cts[c]).float().to(DEV) for c in src}
    init = {k: v.detach().clone() for k, v in M.state_dict().items()}
    gn, active, alphas = [], [], []
    for _ in range(steps):
        if not sub.keys:
            break
        # ONE object per step: each object has its own covariance, so its
        # triplets must be conditioned on that covariance and no other. Drawing
        # from several objects and conditioning on the first is a silent
        # mislabelling.
        o = rng.choice(list(sub.by_obj))
        of_obj = sub.by_obj[o]
        cols = []
        for _ in range(48):
            cand = of_obj[sub.TYPE[of_obj] != 0]
            if not len(cand):
                break
            a = cand[rng.integers(len(cand))]
            alt = [i for i in cand if sub.IMG[i] != sub.IMG[a]
                   and sub.TYPE[i] == sub.TYPE[a]]
            oth = of_obj[sub.TYPE[of_obj] != sub.TYPE[a]]
            if not alt or not len(oth):
                continue
            cols.append((a, alt[rng.integers(len(alt))],
                         oth[rng.integers(len(oth))]))
        if not cols:
            continue
        sel = torch.as_tensor(np.array(cols).T.ravel()).to(DEV)
        Cf = Ct[idx.classes[o]].unsqueeze(0)
        f = M(Cc[sel], Wp[sel], Cf)
        n = len(cols)
        d_ap = 1 - (f[:n] * f[n:2 * n]).sum(1)
        d_an = 1 - (f[:n] * f[2 * n:]).sum(1)
        loss = F.softplus(d_ap - d_an).mean()
        opt.zero_grad(); loss.backward()
        gn.append(float(torch.nn.utils.clip_grad_norm_(M.parameters(), 1e9)))
        active.append(float((d_an > d_ap).float().mean()))
        with torch.no_grad():
            alphas.append(M.cond(Cf).cpu().numpy().ravel())
        opt.step()
    upd = sum(float((M.state_dict()[k] - init[k]).norm()) for k in init)
    M.eval()
    return M, {"grad_norm": float(np.mean(gn)) if gn else 0.0,
               "update_norm": upd,
               "active_triplet": float(np.mean(active)) if active else 0.0}


@torch.no_grad()
def embed_fold(idx, M, centred, wpart, oi, Cvec):
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    f = M(centred[g].to(DEV), wpart[g].to(DEV),
          torch.as_tensor(Cvec).float().to(DEV).unsqueeze(0))
    out = torch.zeros(len(idx.Z), f.shape[1])
    out[g] = f.cpu()
    return out


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
    Sigma = covariances(idx)
    Wn = l2norm(wpart)
    OI = {c: i for i, c in enumerate(files)}

    rows = []
    for T in files:
        oi = OI[T]
        rows.append({"target": T, "config": "F0_cov", "recall@1":
                     recall_one(idx, Wn, oi)})
    print("  F0 baseline computed", flush=True)

    F0 = {r['target']: r['recall@1'] for r in rows}
    for T in files:
        oi = OI[T]
        src = [c for c in files if c != T]
        Pv0 = F0[T]
        P = basis_from_sources(idx, src)
        Cts = {c: P.T @ Sigma[c] @ P for c in files}
        ck = os.path.join(CKPT, f"{T}.pt")
        M, h = train_fold(idx, centred, wpart, Cts, src, a.steps)
        torch.save(M.state_dict(), ck)
        Cbar = np.mean([Cts[c] for c in src], axis=0)
        for tag, Cv in [("F1_correct", Cts[T]), ("F2_shuffled", Cts[src[0]]),
                        ("F3_constant", Cbar)]:
            E = embed_fold(idx, M, centred, wpart, oi, Cv)
            rows.append({"target": T, "config": tag,
                         "recall@1": recall_one(idx, E, oi), **h})
        print(f"  {T:<12} F0 {Pv0:5.1f}  "
              f"F1 {rows[-3]['recall@1']:5.1f}  F2 {rows[-2]['recall@1']:5.1f}  "
              f"F3 {rows[-1]['recall@1']:5.1f}  "
              f"(upd {h['update_norm']:.1f}, act {h['active_triplet']:.2f})",
              flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate14b_conditioner.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate14b_conditioner.csv"), index=False)
    Pv = R.pivot_table(index="target", columns="config", values="recall@1")
    print("\n" + "=" * 100)
    print("GATE 14B -- covariance-conditioned residual metric")
    print("=" * 100)
    print(Pv.round(1).to_string())
    print("\n  means: " + "  ".join(f"{c} {Pv[c].mean():.1f}" for c in Pv.columns))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED")
    print("=" * 100)
    d1 = Pv["F1_correct"] - Pv["F0_cov"]
    d2 = Pv["F1_correct"] - Pv["F2_shuffled"]
    d3 = Pv["F1_correct"] - Pv["F3_constant"]
    rec = d1.mean() / (E3 - Pv["F0_cov"].mean()) * 100
    print(f"  F1 - F0: {d1.mean():+.2f}  better {int((d1 > 0).sum())}/15")
    print(f"  F1 - F2: {d2.mean():+.2f}  better {int((d2 > 0).sum())}/15")
    print(f"  F1 - F3: {d3.mean():+.2f}  better {int((d3 > 0).sum())}/15")
    print(f"  recovery (F1-F0)/(91.8-F0) = {rec:+.1f}%")
    ok = (d1.mean() >= 1.0 and int((d1 > 0).sum()) >= 10
          and d2.mean() > 0 and d3.mean() > 0)
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    H = R.dropna(subset=["update_norm"])
    print(f"  health: update_norm mean {H.update_norm.mean():.2f} "
          f"min {H.update_norm.min():.2f}  active {H.active_triplet.mean():.3f}")


if __name__ == "__main__":
    main()
