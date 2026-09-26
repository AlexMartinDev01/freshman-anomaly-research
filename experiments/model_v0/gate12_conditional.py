# -*- coding: utf-8 -*-
"""
Gate 12 -- Target-normal-conditioned defect metric.

Gate 11 showed a universal source-trained metric is worse than raw on every
target (79.6 vs 84.7), and Gate 12 step 1 asks whether that is because no single
source helps any target. If so, the metric must be GENERATED from the target
itself rather than selected. The only target-side signal available at deployment
is the target's NORMAL samples, so:

    c_t = [mean, std] of the target's normal patches      (no defect labels)
    M_t = F_theta(c_t)                                    (the learned part)
    z'  = metric applied to z, then L2-normalised

Configs:
    D0        raw DINO                                    (floor, no training)
    D1        universal source metric                     (Gate 11 C2a)
    D1.5      target-normal whitening Sigma_t^-1/2        (no learning at all)
    D2        normal-conditioned DIAGONAL metric  w(c_t) * z
    D3        normal-conditioned LOW-RANK adapter z + U V^T z
    D3-shuf   D3 evaluated with ANOTHER object's context  (counterfactual)
    D3-const  D3 evaluated with the mean context          (counterfactual)
    D4        target-supervised metric                    (Gate 10B B2, oracle)

The counterfactuals are the point of the gate. If D3 with the wrong object's
context scores the same as D3 with the right one, the model is not conditioning
on the target at all -- it is just another universal metric wearing a hat.

Meta-episodic training: each source object plays the role of the unseen target.
Its context comes only from its NORMAL patches; its defect labels supply the
triplet loss. Nothing about the held-out target's defects is ever seen.

Pre-registered BEFORE the run:
    Basic PASS    D3 > D0 on >= 11/15 targets, mean gain >= +1.5, recovery > 0
                  on a majority
    Strong PASS   >= 12/15 AND mean recovery >= 30% AND D3 > D3-shuffle
    FAIL          D2, D3 <= D0 while D4 stays high
                  -> target-specific metric exists but normal statistics cannot
                     predict it -> stop scaling the conditional net; move to
                     multi-level / local structural representation

Usage: python experiments/model_v0/gate12_conditional.py [--steps N]
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
from gate10b_structured import Index, ProjXY, embed, l2norm  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g12c")
DEV = "cuda"
RANK = 4


# ------------------------------------------------------------------ metric heads
class DiagMetric(nn.Module):
    """D2: z' = softplus(F(c)) * z -- per-target feature reweighting."""

    def __init__(self, d_ctx=768, d=384, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_ctx, hid), nn.ReLU(),
                                 nn.Linear(hid, d))

    def forward(self, z, c):
        w = F.softplus(self.net(c))
        return z * w


class LowRankMetric(nn.Module):
    """D3: z' = z + U(c) V(c)^T z. Rank-limited on purpose: a full 384x384
    Mahalanobis is ~150k parameters per object and there are 14 training
    objects."""

    def __init__(self, d_ctx=768, d=384, hid=128, r=RANK):
        super().__init__()
        self.r = r
        self.net = nn.Sequential(nn.Linear(d_ctx, hid), nn.ReLU(),
                                 nn.Linear(hid, 2 * d * r))
        self.d = d
        # Small random, NOT zero. z' = z + U V^T z is bilinear: with U = V = 0
        # the gradient is exactly zero for BOTH factors (do/dU ~ V, do/dV ~ U),
        # so a zero-init here is a saddle the adapter can never leave. The
        # zero-init residual trick only works when the zeroed layer is a
        # summand, not a factor of a product.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.mul_(0.1)

    def forward(self, z, c):
        u = self.net(c)
        U = u[:, :self.d * self.r].view(-1, self.d, self.r)
        V = u[:, self.d * self.r:].view(-1, self.d, self.r)
        return z + torch.bmm(U, torch.bmm(V.transpose(1, 2),
                                          z.unsqueeze(-1))).squeeze(-1)


# ------------------------------------------------------------------- whitening
def whitener(normal, eps_frac=0.05):
    """Sigma^-1/2 with an eigenvalue floor: 384 dims from a few thousand patches
    is ill-conditioned and an unfloored inverse amplifies noise directions."""
    X = normal.numpy().astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps_frac * w.mean(), None)
    W = (V / np.sqrt(w)) @ V.T
    return (torch.from_numpy(mu.astype(np.float32)),
            torch.from_numpy(W.astype(np.float32)))


def apply_white(z, mu, W):
    return l2norm((z - mu.to(z.device)) @ W.to(z.device).T)


# --------------------------------------------------------------- contexts
def build_contexts(idx, cap=20000, seed=0):
    """c_t = [mean, std] of the target's NORMAL patches -- the only target-side
    signal a deployment can have."""
    rng = np.random.default_rng(seed)
    ctx, norms = {}, {}
    for c in idx.classes:
        nm = idx.tr[c]["feats"].reshape(-1, 384).astype(np.float32)
        if len(nm) > cap:
            nm = nm[rng.choice(len(nm), cap, replace=False)]
        ctx[c] = np.concatenate([nm.mean(0), nm.std(0)]).astype(np.float32)
        norms[c] = nm
    return ctx, norms


# --------------------------------------------------------------------- training
def build_src_index(idx, objs):
    """Per-object partner pools, computed ONCE.

    Rebuilding these inside the training step is O(defect patches) of Python per
    step -- the same mistake that made one Gate 10B config take 30+ minutes.
    """
    out = {}
    for o in objs:
        di = idx.defect_idx[idx.OBJ[idx.defect_idx] == o]
        if len(di) < 30:
            continue
        by_ot = {}
        for i in di:
            by_ot.setdefault(idx.TYPE[i], []).append(i)
        out[o] = (by_ot, np.where(idx.OBJ == o)[0])
    return out


def src_batches(idx, pre, rng, bs_per_obj=16):
    """One batch of (anchor, positive, negative, object) per source object."""
    out = []
    for o, (by_ot, of_obj) in pre.items():
        A, P, N = [], [], []
        for _ in range(bs_per_obj):
            k = list(by_ot)[rng.integers(len(by_ot))]
            cand = by_ot[k]
            a = cand[rng.integers(len(cand))]
            alt = [i for i in cand if idx.IMG[i] != idx.IMG[a]]
            oth = of_obj[idx.TYPE[of_obj] != idx.TYPE[a]]
            if not alt or not len(oth):
                continue
            A.append(a); P.append(alt[rng.integers(len(alt))])
            N.append(oth[rng.integers(len(oth))])
        if A:
            out.append((o, np.array(A), np.array(P), np.array(N)))
    return out


def train_conditional(idx, ctx, src_objs, config, steps=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    M = (DiagMetric() if config == "D2" else LowRankMetric()).to(DEV)
    opt = torch.optim.AdamW(M.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    Z, POS = idx.Z.to(DEV), idx.POS.to(DEV)
    C = torch.stack([torch.as_tensor(ctx[c]) for c in idx.classes]).to(DEV)
    pre = build_src_index(idx, src_objs)
    for _ in range(steps):
        batch = src_batches(idx, pre, rng)
        if not batch:
            break
        loss = 0
        for o, A, P, N in batch:
            sel = torch.as_tensor(np.concatenate([A, P, N])).to(DEV)
            c = C[o].unsqueeze(0).expand(len(sel), -1)
            h = l2norm(M(Z[sel], c))
            n = len(A)
            a, pos, neg = h[:n], h[n:2 * n], h[2 * n:]
            d_ap = 1 - (a * pos).sum(1)
            d_an = 1 - (a * neg).sum(1)
            # softplus, NOT relu(  d_ap - d_an + margin ). The low-rank adapter
            # starts as the identity map, and identity already satisfies the
            # hinge on raw DINO (d_ap + 0.2 < d_an), so the hinge sits at 0 with
            # an identically zero gradient and the adapter never leaves identity.
            loss = loss + F.softplus((d_ap - d_an + 0.2) / 0.1).mean() * 0.1
        loss = loss / len(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    M.eval()
    return M


# -------------------------------------------------------------------- evaluation
@torch.no_grad()
def recall_and_health(idx, oi, E, raw_E):
    """E must be the FULL index-space embedding: g, by_ot and of_obj all hold
    original indices, so passing E[g] here silently mixes index spaces."""
    rng = np.random.default_rng(0)
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    hits, ap, an = [], [], []
    by_ot = {}
    for i in g:
        by_ot.setdefault(idx.TYPE[i], []).append(i)
    of_obj = np.where(idx.OBJ == oi)[0]
    for i in rng.choice(g, size=min(150, len(g)), replace=False):
        gal = g[idx.IMG[g] != idx.IMG[i]]
        if len(gal) < 10:
            continue
        sim = (E[gal] @ E[i]).numpy()
        order = np.argsort(-sim)
        hits.append((idx.TYPE[gal][order] == idx.TYPE[i])[:1].mean())
        alt = [j for j in by_ot[idx.TYPE[i]] if idx.IMG[j] != idx.IMG[i]]
        oth = of_obj[idx.TYPE[of_obj] != idx.TYPE[i]]
        if alt and len(oth):
            p = alt[rng.integers(len(alt))]
            n = oth[rng.integers(len(oth))]
            ap.append(1 - float(E[i] @ E[p]))
            an.append(1 - float(E[i] @ E[n]))
    if not hits:
        return {}
    # health on THIS object's patches only: the full 131k x 384 embedding would
    # make G @ G.T a 17 GB intermediate
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
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    ctx, norms = build_contexts(idx)
    cmean = np.stack([ctx[c] for c in files]).mean(0).astype(np.float32)
    print(f"  contexts built for {len(files)} objects", flush=True)

    E0 = l2norm(idx.Z)
    P4 = ProjXY(use_xy=False).to(DEV)
    P4.load_state_dict(torch.load(os.path.join(
        RESULTS, "checkpoints_g10b", "B2.pt"), map_location=DEV)); P4.eval()
    E4 = torch.cat([embed(P4, idx.Z[i:i + 20000], idx.POS[i:i + 20000]).cpu()
                    for i in range(0, len(idx.Z), 20000)])

    targets = files if a.targets == "all" else a.targets.split(",")
    rows = []
    for T in targets:
        oi = files.index(T)
        g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
        src = [c for c in files if c != T]

        # D0 / D4: object-independent embeddings, sliced
        for cfg, E in [("D0", E0), ("D4", E4)]:
            h = recall_and_health(idx, oi, E, None)
            h.update({"target": T, "config": cfg}); rows.append(h)

        # D1.5: target-normal whitening, no learning
        mu, W = whitener(torch.from_numpy(norms[T]))
        with torch.no_grad():
            Ew = E0.clone()
            Ew[g] = apply_white(idx.Z[g], mu, W).cpu()
        h = recall_and_health(idx, oi, Ew, None)
        h.update({"target": T, "config": "D1.5"}); rows.append(h)
        print(f"  {T:<12} D1.5 whitening  r@1 {h['recall@1']:5.1f}", flush=True)

        for cfg in ["D2", "D3"]:
            ck = os.path.join(CKPT, f"{T}_{cfg}.pt")
            M = (DiagMetric() if cfg == "D2" else LowRankMetric()).to(DEV)
            if os.path.exists(ck):
                M.load_state_dict(torch.load(ck, map_location=DEV))
            else:
                M = train_conditional(idx, ctx, src, cfg, steps=a.steps)
                torch.save(M.state_dict(), ck)
            M.eval()
            with torch.no_grad():
                Em = E0.clone()
                c = torch.as_tensor(ctx[T]).to(DEV).unsqueeze(0).expand(len(g), -1)
                Em[g] = l2norm(M(idx.Z[g].to(DEV), c)).cpu()
            h = recall_and_health(idx, oi, Em, None)
            h.update({"target": T, "config": cfg}); rows.append(h)
            print(f"  {T:<12} {cfg:<4}          r@1 {h['recall@1']:5.1f}  "
                  f"eff_rank {h['eff_rank']:5.1f}  "
                  f"d_ap {h['d_ap']:.3f} < d_an {h['d_an']:.3f}", flush=True)

            # counterfactuals only for the low-rank adapter
            if cfg == "D3":
                for tag, cvec in [("D3-shuf", ctx[src[0]]), ("D3-const", cmean)]:
                    with torch.no_grad():
                        cc = torch.as_tensor(cvec).to(DEV).unsqueeze(0).expand(
                            len(g), -1)
                        Ec = E0.clone()
                        Ec[g] = l2norm(M(idx.Z[g].to(DEV), cc)).cpu()
                    h2 = recall_and_health(idx, oi, Ec, None)
                    h2.update({"target": T, "config": tag}); rows.append(h2)
                print(f"  {T:<12} D3 shuf/const r@1 "
                      f"{rows[-2]['recall@1']:5.1f} / {rows[-1]['recall@1']:5.1f}",
                      flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate12_conditional.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate12_conditional.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")

    print("\n" + "=" * 100)
    print("GATE 12 -- target-normal-conditioned metric (cross-image Recall@1)")
    print("=" * 100)
    print(P.round(1).to_string())
    print("\n  effective rank:")
    print(R.pivot_table(index="target", columns="config",
                        values="eff_rank").round(1).to_string())

    print("\n" + "=" * 100)
    print("ORACLE RECOVERY  (D - D0) / (D4 - D0)")
    print("=" * 100)
    base, ceil = P["D0"], P["D4"]
    rec = {}
    for cfg in ["D1.5", "D2", "D3", "D3-shuf", "D3-const"]:
        if cfg not in P:
            continue
        d = P[cfg] - base
        r_ = d / (ceil - base).replace(0, np.nan) * 100
        rec[cfg] = r_
        print(f"  {cfg:<9} gain {d.mean():+6.1f}  better {int((d > 0).sum())}/"
              f"{len(d)}   recovery {r_.mean():+6.1f}%   "
              f">30%: {int((r_ > 30).sum())}/{r_.notna().sum()}")
    if rec:
        pd.DataFrame(rec).to_csv(os.path.join(METRICS, "gate12_recovery.csv"))

    d3 = P["D3"] - P["D0"]
    print("\n  PRE-REGISTERED:")
    print(f"    Basic  D3 > D0 in >= 11/15: {int((d3 > 0).sum())}/15, "
          f"mean {d3.mean():+.1f}")
    if "D3-shuf" in P:
        ds = P["D3"] - P["D3-shuf"]
        print(f"    Strong D3 > D3-shuffle:    {int((ds > 0).sum())}/15, "
              f"mean {ds.mean():+.1f}")


if __name__ == "__main__":
    main()
