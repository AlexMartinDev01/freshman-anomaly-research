# -*- coding: utf-8 -*-
"""
Gate 12, step 1 -- source-metric -> target transfer matrix (15 x 15).

Gate 11 trained one metric on ALL 14 sources pooled and found it worse than raw
everywhere (79.6 vs 84.7). But pooling hides structure: it cannot distinguish
"no source transfers to anything" from "a few sources transfer to a few targets
and the rest cancel out". Those imply very different next steps:

    off-diagonal structure exists  -> conditional metric can SELECT/adapt a
                                      source metric -> D2/D3 are well-posed
    nearly all off-diagonal <= 0   -> no existing metric helps any target; the
                                      metric must be GENERATED, not selected

Rows: metric trained on source object s (its defect labels only).
Cols: evaluated on target object t (cross-image same-defect-type Recall@1).
Diagonal is deliberately left empty -- training and testing on the same object
is the Gate 10B oracle, not a transfer.

Usage: python experiments/model_v0/gate12_matrix.py [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import Index, embed, l2norm  # noqa: E402
from gate11_transfer import Sub, train_triplet  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g12")
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--sources", default="all")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    E0 = l2norm(idx.Z)                          # raw baseline, per target
    raw = {}
    for T in files:                             # masked by the object's own patches
        g = idx.defect_idx[idx.OBJ[idx.defect_idx] == files.index(T)]
        raw[T] = _recall(_View(idx, g), E0[g])

    srcs = files if a.sources == "all" else a.sources.split(",")
    rows = []
    for s in srcs:
        ck = os.path.join(CKPT, f"{s}.pt")
        if os.path.exists(ck):
            from gate10b_structured import ProjXY
            P = ProjXY(use_xy=False).to(DEV)
            P.load_state_dict(torch.load(ck, map_location=DEV)); P.eval()
        else:
            sub = Sub(idx, [files.index(s)], "type")
            P = train_triplet(sub, steps=a.steps)
            torch.save(P.state_dict(), ck)
        # embed only the targets we need -- 210 x full-corpus embedding is
        # needless work when each row only reads one object's patches
        for t in files:
            if t == s:
                continue
            oi = files.index(t)
            g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
            with torch.no_grad():
                E = embed(P, idx.Z[g], idx.POS[g]).cpu()
            sub_idx = _View(idx, g)
            r = _recall(sub_idx, E)
            rows.append({"source": s, "target": t, "recall@1": r,
                         "raw": raw[t], "delta": r - raw[t]})
        print(f"  source {s:<12} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate12_matrix.csv"), index=False)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(METRICS, "gate12_matrix.csv"), index=False)
    M = d.pivot(index="source", columns="target", values="recall@1")

    print("\n" + "=" * 100)
    print("SOURCE-METRIC -> TARGET TRANSFER (cross-image same-defect Recall@1)")
    print("  rows = source the metric was trained on, cols = target evaluated on")
    print("=" * 100)
    print(M.round(1).to_string())

    print("\n  delta vs raw DINO for that target:")
    D = d.pivot(index="source", columns="target", values="delta")
    print(D.round(1).to_string())

    print("\n" + "=" * 100)
    print("STRUCTURE")
    print("=" * 100)
    print(f"  off-diagonal cells: {len(d)}")
    print(f"  delta > 0: {int((d.delta > 0).sum())}  ({100*(d.delta>0).mean():.0f}%)")
    print(f"  delta > +2: {int((d.delta > 2).sum())}   delta < -2: {int((d.delta < -2).sum())}")
    print(f"  mean delta {d.delta.mean():+.2f}   median {d.delta.median():+.2f}")
    print(f"  per-source mean delta (does any source help broadly?):")
    rs = d.groupby("source").delta.agg(["mean", "median",
                                        lambda x: int((x > 0).sum())])
    rs.columns = ["mean", "median", "n_targets_better"]
    print(rs.round(2).sort_values("mean", ascending=False).to_string())
    print(f"\n  per-target mean delta (is any target reachable?):")
    rt = d.groupby("target").delta.agg(["mean", "median",
                                        lambda x: int((x > 0).sum())])
    rt.columns = ["mean", "median", "n_sources_better"]
    print(rt.round(2).sort_values("mean", ascending=False).to_string())
    print(f"\n  best single cell: {d.loc[d.delta.idxmax()].to_dict()}")
    print(f"  worst single cell: {d.loc[d.delta.idxmin()].to_dict()}")


class _View:
    """Minimal stand-in so retrieval_one can run on a subset of the index."""

    def __init__(self, idx, g):
        self.defect_idx = g
        self.OBJ = idx.OBJ[g]
        self.IMG = idx.IMG[g]
        self.TYPE = idx.TYPE[g]


def _recall(v, E, n_query=150, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(len(v.defect_idx))
    hits = []
    for i in rng.choice(g, size=min(n_query, len(g)), replace=False):
        gal = g[v.IMG[g] != v.IMG[i]]
        if len(gal) < 10:
            continue
        with torch.no_grad():
            sim = (E[gal] @ E[i]).numpy()
        order = np.argsort(-sim)
        hits.append((v.TYPE[gal][order] == v.TYPE[i])[:1].mean())
    return float(np.mean(hits)) * 100 if hits else np.nan


if __name__ == "__main__":
    main()
