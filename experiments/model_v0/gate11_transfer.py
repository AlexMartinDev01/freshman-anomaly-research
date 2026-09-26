# -*- coding: utf-8 -*-
"""
Gate 11 -- Leave-One-Object-Out Structured Metric Transfer.

Gate 10B-O established that raw DINO already carries defect-type structure
(chance-adjusted cross-image Recall@1 = 67.3) and that a structured triplet
metric trained on TARGET labels lifts it to 83.9, while the binary objective
collapses the representation to effective rank 1.7. But that metric used the
target's own defect labels -- it is an oracle.

Gate 11 removes the target supervision. The projection is trained ONLY on source
objects' defect labels, then evaluated on a held-out object:

    C0   no training            raw DINO                     (floor)
    C1   source binary anomaly  normal vs defect             (negative control)
    C2a  source defect TYPE     dataset defect classes
    C2b  source MORPHOLOGY      coarse cross-object primitives
    C3   target defect TYPE     Gate 10B's B2                (oracle ceiling)

C2b answers a separate question: is the transferable unit the dataset's
per-object defect taxonomy, or a cross-object morphological primitive? Seven
blunt groups on purpose -- a fine taxonomy would make this taxonomy engineering
rather than a test of the hypothesis.

Headline metric is ORACLE RECOVERY RATIO, per target:
    Recovery = (C2 - C0) / (C3 - C0)
0% = no cross-object transfer; 100% = target labels bought nothing extra.

Representation health is ASSERTED, not merely printed. Gate 10B's first triplet
loss pulled negatives together, collapsed every patch onto one point and
produced a confident false negative in the direction of the story we already
believed. Any row where E[d_ap] >= E[d_an] or effective rank collapses is INVALID.

Pre-registered BEFORE the run:
    Representation PASS  C2a > C0 on cross-image Recall@1 in >= 12/15 targets,
                         mean gain clearly positive, r_eff(C2a) not collapsed
    Strong PASS          Recovery >= 30% on a majority of eligible targets
    if C2 ~ C0 but C3 >> C0 -> defect structure is object-conditional rather
                         than transferable -> object-conditioned metric

Usage: python experiments/model_v0/gate11_transfer.py [--steps N]
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
from gate10b_structured import (Index, ProjXY, embed, l2norm,  # noqa: E402
                                oracle_headroom)
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g11")
DEV = "cuda"

# Coarse cross-object morphological primitives. Seven groups, deliberately blunt.
MORPH = {
    "crack_like": ["crack", "broken", "broken_large", "broken_small",
                   "broken_teeth", "split_teeth", "cut", "cut_inner_insulation",
                   "cut_outer_insulation", "cut_lead", "scratch",
                   "scratch_head", "scratch_neck", "gray_stroke"],
    "hole_missing": ["hole", "missing_cable", "missing_wire", "poke"],
    "contamination": ["contamination", "metal_contamination", "oil", "liquid",
                      "glue", "glue_strip", "fabric_border", "fabric_interior"],
    "deformation": ["bent", "bent_wire", "bent_lead", "flip", "squeeze",
                    "squeezed_teeth", "manipulated_front", "damaged_case",
                    "misplaced", "faulty_imprint"],
    "surface_texture": ["rough", "thread", "thread_side", "thread_top", "print"],
    "color_change": ["color", "pill_type", "cable_swap"],
    "other": ["combined", "defective"],
}
NAME2MORPH = {n: g for g, ns in MORPH.items() for n in ns}


def morph_of(t):
    return "normal" if t == "normal" else NAME2MORPH.get(t, "other")


# --------------------------------------------------------------- source subsets
class Sub:
    """A source-only view over the parent Index; Z/POS are shared by reference."""

    def __init__(self, parent, objs, label_key):
        self.Z, self.POS = parent.Z, parent.POS
        self.OBJ, self.TYPE, self.IMG = parent.OBJ, parent.TYPE, parent.IMG
        self.LAB = parent.MORPH if label_key == "morph" else parent.TYPE
        objs = set(objs)
        keep = np.array([o in objs for o in parent.OBJ])
        self.defect_idx = np.where(keep & (parent.TYPE != 0))[0]
        self.by_obj = {o: np.where(keep & (parent.OBJ == o))[0] for o in objs}
        by_ot = {}
        for i in self.defect_idx:
            by_ot.setdefault((self.OBJ[i], self.LAB[i]), []).append(i)
        self.by_ot = {k: np.array(v) for k, v in by_ot.items()}
        self.alt = {}
        for o in objs:
            idxs = self.by_obj[o]
            for t in {self.LAB[i] for i in idxs}:
                self.alt[(o, t)] = idxs[self.LAB[idxs] != t]
        self.keys = [k for k, v in self.by_ot.items()
                     if len({self.IMG[i] for i in v}) >= 2]
        self.nrm = {o: self.by_obj[o][self.TYPE[self.by_obj[o]] == 0]
                    for o in objs}

    def defect_mask(self):
        return torch.as_tensor(self.TYPE != 0)


def _partners(s, rng):
    k = s.keys[rng.integers(len(s.keys))]
    cand = s.by_ot[k]
    a = cand[rng.integers(len(cand))]
    pos = a
    for _ in range(20):                 # positive always from another image
        pos = cand[rng.integers(len(cand))]
        if s.IMG[pos] != s.IMG[a]:
            break
    o, t = k
    nrm = s.nrm[o]
    oth = s.defect_idx[s.OBJ[s.defect_idx] != o]
    return (a, pos,
            s.alt[(o, t)][rng.integers(len(s.alt[(o, t)]))],
            nrm[rng.integers(len(nrm))] if len(nrm) else a,
            oth[rng.integers(len(oth))] if len(oth) else a)


def train_triplet(s, steps=2000, bs=48, lr=1e-3, margin=0.2, seed=0):
    torch.manual_seed(seed)
    P = ProjXY(use_xy=False).to(DEV)
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Z, XY = s.Z.to(DEV), s.POS.to(DEV)
    for _ in range(steps):
        if not s.keys:
            break
        cols = [x for x in (_partners(s, rng) for _ in range(bs))]
        sel = torch.as_tensor(np.array(cols).T.ravel()).to(DEV)
        h = l2norm(P(Z[sel], XY[sel]))
        n = bs
        a, pos = h[:n], h[n:2 * n]
        d_ap = 1 - (a * pos).sum(1)
        loss = 0
        for ng in (h[2 * n:3 * n], h[3 * n:4 * n], h[4 * n:5 * n]):
            loss = loss + F.relu(d_ap - (1 - (a * ng).sum(1)) + margin)
        loss = loss.mean() / 3
        opt.zero_grad(); loss.backward(); opt.step()
    P.eval()
    return P


def train_binary_src(s, steps=1500, bs=256, lr=1e-3, seed=0):
    """C1: source-only binary anomaly projection -- the Gate 10A objective."""
    torch.manual_seed(seed)
    P = ProjXY(use_xy=False).to(DEV)
    head = torch.nn.Linear(128, 1).to(DEV)
    opt = torch.optim.AdamW(list(P.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Z = s.Z.to(DEV)
    y = torch.as_tensor(s.TYPE != 0).float().to(DEV)
    negpool = np.where(~(s.TYPE != 0))[0]
    for _ in range(steps):
        sel = torch.as_tensor(np.concatenate([
            rng.choice(len(Z), bs // 2, replace=False),
            rng.choice(negpool, bs // 2, replace=False)])).to(DEV)
        loss = F.binary_cross_entropy_with_logits(
            head(l2norm(P(Z[sel]))).squeeze(-1), y[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    P.eval()
    return P


# ------------------------------------------------------------------ health
@torch.no_grad()
def health(P, idx, oi, n_pairs=1500, seed=0):
    """d_ap (same object+type, different image) MUST be below d_an."""
    rng = np.random.default_rng(seed)
    di = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    if len(di) < 50:
        return {}
    by_ot = {}
    for i in di:
        by_ot.setdefault((idx.OBJ[i], idx.TYPE[i]), []).append(i)
    # Negative must be a DIFFERENT defect type of the SAME object (normal
    # patches count). Restricting to one object makes "another object's defect"
    # impossible, which silently emptied every negative pool.
    of_obj = np.where(idx.OBJ == oi)[0]
    A, Pp, Nn = [], [], []
    for _ in range(n_pairs):
        k = list(by_ot)[rng.integers(len(by_ot))]
        cand = by_ot[k]
        a = cand[rng.integers(len(cand))]
        alt = [i for i in cand if idx.IMG[i] != idx.IMG[a]]
        oth = of_obj[idx.TYPE[of_obj] != idx.TYPE[a]]
        if not alt or not len(oth):
            continue
        A.append(a)
        Pp.append(alt[rng.integers(len(alt))])
        Nn.append(oth[rng.integers(len(oth))])
    if not A:
        return {}
    sel = np.array(A + Pp + Nn)
    E = (l2norm(idx.Z[sel]) if P is None
         else embed(P, idx.Z[sel], idx.POS[sel])).cpu()
    n = len(A)
    ea, ep, en = E[:n], E[n:2 * n], E[2 * n:]
    s2 = rng.choice(di, size=min(3000, len(di)), replace=False)
    G = (l2norm(idx.Z[s2]) if P is None
         else embed(P, idx.Z[s2], idx.POS[s2])).cpu().numpy().astype(np.float64)
    Gt = torch.as_tensor(G)                      # un-centred: mean_dist must be
    D = 1 - Gt @ Gt.T                            # a real pairwise cosine distance
    iu = np.triu_indices(len(G), 1)
    mean_d = float(D.numpy()[iu].mean())
    Gc = G - G.mean(0, keepdims=True)            # centring is only for the rank
    sv = np.linalg.svd(Gc, compute_uv=False) ** 2
    sv = sv / sv.sum()
    return {"eff_rank": float(np.exp(-(sv * np.log(sv + 1e-12)).sum())),
            "mean_dist": mean_d,
            "d_ap": float((1 - (ea * ep).sum(1)).mean()),
            "d_an": float((1 - (ea * en).sum(1)).mean())}


def retrieval_one(idx, E, oi, n_query=150, seed=0):
    """Cross-image same-defect-type retrieval restricted to ONE object."""
    rng = np.random.default_rng(seed)
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    hits = {1: [], 5: [], 10: []}
    if len(g) < 20:
        return {"recall@1": np.nan, "recall@5": np.nan, "recall@10": np.nan,
                "n_query": 0}
    for i in rng.choice(g, size=min(n_query, len(g)), replace=False):
        gal = g[idx.IMG[g] != idx.IMG[i]]
        if len(gal) < 10:
            continue
        order = np.argsort(-(E[gal] @ E[i]).numpy())
        same = idx.TYPE[gal][order] == idx.TYPE[i]
        for k in hits:
            hits[k].append(same[:k].mean())
    return {"recall@1": float(np.mean(hits[1])) * 100,
            "recall@5": float(np.mean(hits[5])) * 100,
            "recall@10": float(np.mean(hits[10])) * 100,
            "n_query": len(hits[1])}


@torch.no_grad()
def embed_all(P, idx, batch=20000):
    return torch.cat([embed(P, idx.Z[i:i + batch], idx.POS[i:i + batch]).cpu()
                      for i in range(0, len(idx.Z), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--targets", default="all")
    ap.add_argument("--reload", action="store_true",
                    help="reuse saved checkpoints instead of retraining")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    idx.MORPH = np.array([morph_of(t) for t in idx.TNAME])
    groups = sorted(set(idx.MORPH) - {"normal"})
    print(f"  morphology: {len(groups)} groups {groups}", flush=True)

    targets = files if a.targets == "all" else a.targets.split(",")
    E0 = l2norm(idx.Z)                                   # C0, object-independent
    b2 = os.path.join(RESULTS, "checkpoints_g10b", "B2.pt")
    P3 = ProjXY(use_xy=False).to(DEV)
    P3.load_state_dict(torch.load(b2, map_location=DEV)); P3.eval()
    E3 = embed_all(P3, idx)                              # C3, object-independent

    rows, hlth = [], []
    # C0 / C3 are the same embedding for every target
    for T in targets:
        oi = files.index(T)
        for cfg, E in [("C0", E0), ("C3", E3)]:
            r = retrieval_one(idx, E, oi)
            r.update({"target": T, "config": cfg})
            rows.append(r)
            h = health(None if cfg == "C0" else P3, idx, oi)
            h.update({"target": T, "config": cfg})
            hlth.append(h)

    for T in targets:
        oi = files.index(T)
        src = [files.index(c) for c in files if c != T]
        for cfg in ["C1", "C2a", "C2b"]:
            ck = os.path.join(CKPT, f"{T}_{cfg}.pt")
            if a.reload and os.path.exists(ck):
                P = ProjXY(use_xy=False).to(DEV)
                P.load_state_dict(torch.load(ck, map_location=DEV)); P.eval()
            else:
                s = Sub(idx, src, "morph" if cfg == "C2b" else "type")
                P = (train_binary_src(s) if cfg == "C1"
                     else train_triplet(s, steps=a.steps))
                torch.save(P.state_dict(), ck)
            E = embed_all(P, idx)
            r = retrieval_one(idx, E, oi)
            r.update({"target": T, "config": cfg})
            rows.append(r)
            h = health(P, idx, oi)
            h.update({"target": T, "config": cfg})
            hlth.append(h)
            print(f"  {T:<12} {cfg:<4} r@1 {r['recall@1']:5.1f}  "
                  f"eff_rank {h['eff_rank']:6.1f}  "
                  f"d_ap {h['d_ap']:.3f} < d_an {h['d_an']:.3f}"
                  f"  {'OK' if h['d_ap'] < h['d_an'] else 'INVALID'}", flush=True)

    R = pd.DataFrame(rows)
    H = pd.DataFrame(hlth)
    O = pd.DataFrame([{**oracle_headroom(idx, None if c == "C0" else P).set_index(
        "object").loc[T].to_dict(), "target": T, "config": c}
        for c, P in [("C0", None), ("C3", P3)] for T in targets])
    R.to_csv(os.path.join(METRICS, "gate11_retrieval.csv"), index=False)
    H.to_csv(os.path.join(METRICS, "gate11_health.csv"), index=False)
    O.to_csv(os.path.join(METRICS, "gate11_oracle.csv"), index=False)

    P_ = R.pivot_table(index="target", columns="config", values="recall@1")
    print("\n" + "=" * 96)
    print("GATE 11 -- leave-one-object-out structured metric transfer (Recall@1)")
    print("=" * 96)
    print(P_.round(1).to_string())
    print("\n  effective rank:")
    print(H.pivot_table(index="target", columns="config",
                        values="eff_rank").round(1).to_string())

    print("\n" + "=" * 96)
    print("ORACLE RECOVERY RATIO  (C - C0) / (C3 - C0), per target")
    print("=" * 96)
    base = P_["C0"]
    rec = {}
    for cfg in ["C1", "C2a", "C2b"]:
        d = P_[cfg] - base
        r_ = (d / (P_["C3"] - base).replace(0, np.nan) * 100)
        rec[cfg] = r_
        print(f"  {cfg:<4} gain {d.mean():+6.1f}  better {int((d > 0).sum())}/"
              f"{len(d)}   recovery {r_.mean():+6.1f}%   "
              f">30%: {int((r_ > 30).sum())}/{r_.notna().sum()}")
    pd.DataFrame(rec).to_csv(os.path.join(METRICS, "gate11_recovery.csv"))

    bad = H[H.d_ap.notna() & (H.d_ap >= H.d_an)]
    print(f"\n  HEALTH: {len(bad)}/{len(H)} rows with d_ap >= d_an")
    if len(bad):
        print(bad[["target", "config", "d_ap", "d_an", "eff_rank"]].to_string())
    d2 = P_["C2a"] - P_["C0"]
    print(f"\n  PRE-REGISTERED Representation PASS: C2a > C0 in >= 12/15")
    print(f"    C2a better in {int((d2 > 0).sum())}/{len(d2)}, mean {d2.mean():+.1f}")


if __name__ == "__main__":
    main()
