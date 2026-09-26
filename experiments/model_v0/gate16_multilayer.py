# -*- coding: utf-8 -*-
"""
FINAL representation gate -- bounded multi-layer DINO statistics.

Scope is fixed in advance and deliberately small, because this is the last
representation-side experiment. Three depths, two interfaces each, one untrained
fusion:

    mid       block 5
    midlate   block 8
    final     block 11

    raw     no correction
    cov     (Sigma + eps I)^-0.5, gamma = 0.5, the one interface that has ever
            worked in this project

plus `fuse`, the fixed equal-weight average of the three layers' covariance-
corrected anomaly distances. No layer sweep, no gamma re-tuning, no learned
fusion, no new statistics.

Everything shares the split, the sampler, the regularisation and the scoring code
already used by gates 12-15, and the own-vs-shuffled covariance counterfactual is
carried over unchanged.

Pre-registered, frozen before the run:
    ADOPT multi-layer  iff  the best multi-layer configuration beats the
                            final-layer covariance baseline by >= +1.0 mean
                            Recall@1 AND improves >= 10/15 objects
    otherwise          KEEP the final layer; no further representation gates

This is the hard stop: whatever comes out, Model V1 is frozen afterwards and the
project moves to real anomaly detection.

Usage: python experiments/model_v0/gate16_multilayer.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10b_structured import l2norm  # noqa: E402
from gate13b_shared_residual import recall_one  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

ML = os.path.join(RESULTS, "cache_ml")
METRICS = os.path.join(RESULTS, "metrics")
GAMMA = 0.5
EPS_FRAC = 0.05
LAYERS = ["mid", "midlate", "final"]
CAP = 20000


# ------------------------------------------------------------- lightweight index
class MLIndex:
    """Same interface as gate 10's Index, but per layer and from cache_ml."""

    def __init__(self, files, layer, seed=0):
        rng = np.random.default_rng(seed)
        self.classes = files
        Z, OBJ, TY, IMG, TNM = [], [], [], [], []
        self.te, self.normal = {}, {}
        type_of = {"normal": 0}
        img_of = {}
        for oi, c in enumerate(files):
            tr = np.load(os.path.join(ML, layer, f"{c}_train.npz"),
                         allow_pickle=True)
            te = np.load(os.path.join(ML, layer, f"{c}_test.npz"),
                         allow_pickle=True)
            self.te[c] = {k: np.asarray(te[k]) for k in te.files}
            nm = np.asarray(tr["feats"]).reshape(-1, 384).astype(np.float32)
            self.normal[c] = nm
            types = list(self.te[c]["types"])
            gt = self.te[c]["gt_frac"].reshape(len(types), -1)
            for i in range(len(types)):
                if types[i] != "bad":
                    continue
                m = gt[i] > 0.10
                if not m.any():
                    continue
                tname = str(self.te[c]["names"][i]).split("/")[0]
                if tname not in type_of:
                    type_of[tname] = len(type_of)
                k = int(m.sum())
                Z.append(self.te[c]["feats"][i].astype(np.float32)[m])
                OBJ += [oi] * k
                TY += [type_of[tname]] * k
                IMG += [img_of.setdefault(f"{c}/{i}", len(img_of))] * k
                TNM += [tname] * k
        self.Z = torch.from_numpy(np.concatenate(Z))
        self.OBJ, self.TYPE, self.IMG = (np.array(OBJ), np.array(TY),
                                         np.array(IMG))
        self.TNAME = np.array(TNM)
        self.defect_idx = np.where(self.TYPE != 0)[0]


def frac_pow(C, gamma=GAMMA, eps_frac=EPS_FRAC):
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, eps_frac * w.mean(), None)
    return ((V * w ** (-gamma / 2.0)) @ V.T).astype(np.float32)


def bank(idx, c, layer, rng):
    X = idx.normal[c]
    if len(X) > CAP:
        X = X[rng.choice(len(X), CAP, replace=False)]
    Xd = X.astype(np.float64)
    mu = Xd.mean(0)
    Xc = Xd - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    return mu.astype(np.float32), frac_pow(C)


@torch.no_grad()
def transform(idx, oi, mu, W, perm=None):
    g = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]
    Z = idx.Z[g]
    W2 = W if perm is None else W[perm]
    E = l2norm((Z - torch.from_numpy(mu)) @ torch.from_numpy(W2).T)
    O = torch.zeros(len(idx.Z), E.shape[1])
    O[g] = E
    return O


def main():
    files = sorted(f[:-len("_train.npz")] for f in
                   os.listdir(os.path.join(ML, "final"))
                   if f.endswith("_train.npz"))
    if not files:
        raise SystemExit("cache_ml is empty -- run cache_multilayer.py first")
    print(f"  {len(files)} objects", flush=True)

    idx = {}
    rows, score = [], {}
    for layer in LAYERS:
        idx[layer] = MLIndex(files, layer)
        print(f"  {layer} index built ({len(idx[layer].Z)} patches)", flush=True)

    for T in files:
        oi = files.index(T)
        dists = {}
        for layer in LAYERS:
            Ix = idx[layer]
            rng = np.random.default_rng(0)
            mu, W = bank(Ix, T, layer, rng)
            Eraw = l2norm(Ix.Z)
            Eraw_o = torch.zeros(len(Ix.Z), 384)
            g = Ix.defect_idx[Ix.OBJ[Ix.defect_idx] == oi]
            Eraw_o[g] = Eraw[g]
            rows.append({"target": T, "config": f"{layer}_raw",
                         "recall@1": recall_one(Ix, Eraw_o, oi)})
            Ec = transform(Ix, oi, mu, W)
            rows.append({"target": T, "config": f"{layer}_cov",
                         "recall@1": recall_one(Ix, Ec, oi)})
            if layer == "final":
                perm = np.random.default_rng(1).permutation(len(mu))
                rows.append({"target": T, "config": "final_cov_shuffled",
                             "recall@1": recall_one(
                                 Ix, transform(Ix, oi, mu, W, perm), oi)})
            dists[layer] = Ec
        # fixed equal-weight fusion of the three layers' corrected distances
        for layer in LAYERS:
            Ix = idx[layer]
            if layer == LAYERS[0]:
                fused = dists[layer].clone()
            else:
                fused = fused + dists[layer]
        rows.append({"target": T, "config": "fuse_cov",
                     "recall@1": recall_one(idx["final"], fused, oi)})
        print(f"  {T:<12} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate16_multilayer.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate16_multilayer.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")

    print("\n" + "=" * 100)
    print("FINAL representation gate -- bounded multi-layer (cross-image Recall@1)")
    print("=" * 100)
    print(P.round(1).to_string())
    print("\n  means: " + "  ".join(f"{c} {P[c].mean():.1f}" for c in P.columns))

    base = P["final_cov"]
    print("\n" + "=" * 100)
    print("PRE-REGISTERED: adopt multi-layer iff >= +1.0 mean AND >= 10/15 vs "
          "final_cov")
    print("=" * 100)
    print(f"  final_cov baseline = {base.mean():.1f}")
    for c in P.columns:
        if c in ("final_cov", "final_cov_shuffled"):
            continue
        d = P[c] - base
        flag = ""
        if c in ("fuse_cov", "mid_cov", "midlate_cov"):
            flag = "  <- eligible" if (d.mean() >= 1.0 and int((d > 0).sum()) >= 10) else ""
        print(f"  {c:<20}{P[c].mean():>7.1f}{d.mean():>+9.2f}"
              f"{int((d > 0).sum()):>7}/15{flag}")
    d = base - P["final_cov_shuffled"]
    print(f"\n  final own - shuffled: {d.mean():+.2f} ({int((d > 0).sum())}/15)")

    best = max(("fuse_cov", "mid_cov", "midlate_cov"),
               key=lambda c: P[c].mean())
    dbest = P[best] - base
    adopt = dbest.mean() >= 1.0 and int((dbest > 0).sum()) >= 10
    print(f"\n  best eligible: {best} ({P[best].mean():.1f})")
    print(f"  VERDICT: {'ADOPT multi-layer' if adopt else 'KEEP final layer only'}")
    print("\n  Model V1 is now frozen. No further representation gates.")


if __name__ == "__main__":
    main()
