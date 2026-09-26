# -*- coding: utf-8 -*-
"""
Gate 10B-O -- Structured Defect Oracle Diagnostic.

Gate 10A's most important number was not R2 < R1. It was this:

    target-defect oracle headroom:  raw +0.78  ->  R1 +0.01  ->  R2 +0.00

R1 has no adversarial term at all, yet its binary defect-ness objective already
destroys what retrieval needs. A binary head can collapse scratch / crack / hole
/ dent into one "anomaly region", score ~1.00 anomaly accuracy, and leave nothing
1-NN can use. So the question is no longer "how do we remove object identity":

    does raw DINO carry defect-TYPE / defect-MORPHOLOGY structure at all,
    and was it merely unreadable through the binary interface?

This is a diagnostic that uses target defect labels. It is NOT a deployable
method. It exists to choose between two very different next steps: keep DINO and
change the objective, or abandon the frozen single-layer interface.

Configs (few on purpose):
    B0  raw DINO, plain 1-NN
    B1  binary anomaly projection          (the Gate 10A objective)
    B2  structured defect metric           triplet on defect TYPE
    B3  structured metric + (x, y)         is position part of defect semantics?

Measured -- NOT anomaly classification accuracy, because 10A proved that number
can be near-perfect while the representation is useless for retrieval:
    1. cross-image same-defect Recall@K   gallery EXCLUDES the query's own image,
       otherwise the metric just learns image instance
    2. target-defect oracle headroom      can it be restored above raw?

Pre-registered BEFORE the run:
    PASS  B2 > B0 on cross-image Recall@1 in >= 12/15 objects AND oracle
          headroom (B2) clearly positive and above raw's
    if B3 >> B2   -> position is part of defect semantics -> structure-aware Gate 11
    if B2/B3 fail -> close the frozen single-layer DINO interface -> Gate 11A

Usage: python experiments/model_v0/gate10b_structured.py [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate7b3r import make_dist  # noqa: E402
from models.tail_adapter import l2norm  # noqa: E402
from tail_calib import RESULTS, materialize, mean_top1p  # noqa: E402

V1 = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
DEV = "cuda"
CONFIGS = ["B0", "B1", "B2", "B3"]


def xy_grid(gh, gw):
    """Normalised (x, y) per patch index in row-major order."""
    pos = np.arange(gh * gw)
    return np.stack([(pos % gw) / max(gw - 1, 1),
                     (pos // gw) / max(gh - 1, 1)], 1).astype(np.float32)


# ------------------------------------------------------------------------ index
class Index:
    """Every patch with (object, defect type, image, position).

    'normal' is registered as type id 0 up front, so `TYPE != 0` is the defect
    mask. Letting it be assigned lazily made the first defect type get id 0.
    """

    def __init__(self, classes, max_normal_per_obj=4000, seed=0):
        rng = np.random.default_rng(seed)
        Z, OBJ, TY, IMG, POS, TNM = [], [], [], [], [], []
        self.classes, self.grids, self.te, self.tr = classes, {}, {}, {}
        type_of = {"normal": 0}
        obj_of = {c: i for i, c in enumerate(classes)}
        img_of = {}
        for c in classes:
            tr = materialize(np.load(os.path.join(V1, f"{c}_train.npz"),
                                     allow_pickle=True))
            te = materialize(np.load(os.path.join(V1, f"{c}_test.npz"),
                                     allow_pickle=True))
            self.tr[c], self.te[c] = tr, te
            types = list(te["types"])
            gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
            self.grids[c] = (gh, gw)
            grid = xy_grid(gh, gw)
            gt = te["gt_frac"].reshape(len(types), -1)
            for i in range(len(types)):
                if types[i] != "bad":
                    continue
                m = gt[i] > 0.10
                if not m.any():
                    continue
                tname = str(te["names"][i]).split("/")[0]
                if tname not in type_of:
                    type_of[tname] = len(type_of)
                k = int(m.sum())
                Z.append(te["feats"][i].astype(np.float32)[m])
                OBJ += [obj_of[c]] * k
                TY += [type_of[tname]] * k
                TNM += [tname] * k
                IMG += [img_of.setdefault(f"{c}/{i}", len(img_of))] * k
                POS.append(grid[m])
            # normals of the same object, with their true grid positions
            nm = tr["feats"].reshape(-1, 384).astype(np.float32)
            if len(nm) > max_normal_per_obj:
                nm = nm[rng.choice(len(nm), max_normal_per_obj, replace=False)]
            k = len(nm)
            Z.append(nm)
            OBJ += [obj_of[c]] * k
            TY += [0] * k
            TNM += ["normal"] * k
            IMG += [img_of.setdefault(f"{c}/normal", len(img_of))] * k
            POS.append(np.tile(grid, (int(np.ceil(k / len(grid))), 1))[:k])
        self.Z = torch.from_numpy(np.concatenate(Z))
        self.OBJ = np.array(OBJ)
        self.TYPE = np.array(TY)
        self.TNAME = np.array(TNM)   # name per patch, for morphology remapping
        self.IMG = np.array(IMG)
        self.POS = torch.from_numpy(np.concatenate(POS))
        self.n_type = len(type_of) - 1
        # Precompute every lookup the sampler needs. Rebuilding these inside
        # sample_batch made training O(N) Python work PER STEP (~220k iterations
        # x 2000 steps) and a single config never finished.
        self.defect_idx = np.where(self.TYPE != 0)[0]
        self.by_obj = {}
        for i in range(len(self.Z)):
            self.by_obj.setdefault(self.OBJ[i], []).append(i)
        self.by_obj = {k: np.array(v) for k, v in self.by_obj.items()}
        self.di_by_obj = {o: self.defect_idx[self.OBJ[self.defect_idx] == o]
                          for o in self.by_obj}
        self.by_ot = {}
        for i in self.defect_idx:
            self.by_ot.setdefault((self.OBJ[i], self.TYPE[i]), []).append(i)
        self.by_ot = {k: np.array(v) for k, v in self.by_ot.items()}
        # (obj, type) -> indices sharing the object but a DIFFERENT type
        self.alt = {}
        for o, idxs in self.by_obj.items():
            for t in {self.TYPE[i] for i in idxs}:
                self.alt[(o, t)] = idxs[self.TYPE[idxs] != t]
        # a plain list of tuples: np.array() would make each key an ndarray,
        # which is unhashable and cannot index the dict
        self.keys = [k for k, v in self.by_ot.items()
                     if len({self.IMG[i] for i in v}) >= 2]
        print(f"  index: {len(self.Z)} patches, {len(classes)} objects, "
              f"{self.n_type} defect types, {len(self.keys)} usable "
              f"(obj,type) groups", flush=True)

    def defect_mask(self):
        return torch.as_tensor(self.TYPE != 0)


# ----------------------------------------------------------------------- model
class ProjXY(nn.Module):
    def __init__(self, use_xy, d=384, hid=256, out=128, n_pos=16):
        super().__init__()
        self.use_xy, self.n_pos = use_xy, n_pos
        self.net = nn.Sequential(nn.Linear(d + (n_pos if use_xy else 0), hid),
                                 nn.ReLU(), nn.Linear(hid, out))

    def pos_enc(self, xy):
        f = (2.0 ** torch.arange(self.n_pos // 4, device=xy.device)) * np.pi
        return torch.cat([torch.sin(xy[:, :1] * f), torch.cos(xy[:, :1] * f),
                          torch.sin(xy[:, 1:] * f), torch.cos(xy[:, 1:] * f)], 1)

    def forward(self, z, xy=None):
        if self.use_xy and xy is not None:
            z = torch.cat([z, self.pos_enc(xy)], 1)
        return self.net(z)


# -------------------------------------------------------------------- training
def _partners(idx, rng):
    k = idx.keys[rng.integers(len(idx.keys))]
    cand = idx.by_ot[k]
    a = cand[rng.integers(len(cand))]
    pos = a
    for _ in range(20):                 # positive MUST come from another image
        pos = cand[rng.integers(len(cand))]
        if idx.IMG[pos] != idx.IMG[a]:
            break
    o, t = k
    alt = idx.alt[(o, t)]
    nrm = idx.by_obj[o][idx.TYPE[idx.by_obj[o]] == 0]
    oth = idx.defect_idx[idx.OBJ[idx.defect_idx] != o]
    return (a, pos,
            alt[rng.integers(len(alt))] if len(alt) else a,
            nrm[rng.integers(len(nrm))] if len(nrm) else a,
            oth[rng.integers(len(oth))] if len(oth) else a)


def sample_batch(idx, bs, rng):
    if len(idx.keys) == 0:
        return None
    cols = list(zip(*[_partners(idx, rng) for _ in range(bs)]))
    return tuple(torch.as_tensor(np.array(c)) for c in cols)


def train_metric(idx, config, steps=2000, bs=48, lr=1e-3, margin=0.2, seed=0):
    """B2/B3: pull same defect morphology together, push the rest apart."""
    torch.manual_seed(seed)
    P = ProjXY(use_xy=(config == "B3")).to(DEV)
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Z, XY = idx.Z.to(DEV), idx.POS.to(DEV)
    for step in range(steps):
        b = sample_batch(idx, bs, rng)
        if b is None:
            break
        h = l2norm(P(Z[torch.cat(b).to(DEV)], XY[torch.cat(b).to(DEV)]))
        n = bs
        a, pos = h[:n], h[n:2 * n]
        d_ap = 1 - (a * pos).sum(1)               # anchor-positive distance
        loss = 0
        for ng in (h[2 * n:3 * n], h[3 * n:4 * n], h[4 * n:5 * n]):
            d_an = 1 - (a * ng).sum(1)            # anchor-negative distance
            # triplet: push negatives AWAY and positives TOGETHER.
            # Writing relu(margin + d_an) for the negative term, as this file
            # first did, is minimised by pulling the negatives IN and collapses
            # every patch onto one point (mean pairwise cos-dist went to 0.0000
            # and retrieval fell below raw in all 14 non-degenerate objects).
            loss = loss + F.relu(d_ap - d_an + margin)
        loss = loss.mean() / 3
        opt.zero_grad(); loss.backward(); opt.step()
    P.eval()
    return P


def train_binary(idx, steps=1500, bs=256, lr=1e-3, seed=0):
    """B1: the Gate 10A objective, so B1 is scored on the SAME retrieval metrics
    as everything else instead of only on anomaly accuracy."""
    torch.manual_seed(seed)
    P = ProjXY(use_xy=False).to(DEV)
    head = nn.Linear(128, 1).to(DEV)
    opt = torch.optim.AdamW(list(P.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    Z = idx.Z.to(DEV)
    y = torch.as_tensor(idx.TYPE != 0).float().to(DEV)
    negpool = np.where(~(idx.TYPE != 0))[0]
    for step in range(steps):
        sel = torch.as_tensor(np.concatenate([
            rng.choice(len(Z), bs // 2, replace=False),
            rng.choice(negpool, bs // 2, replace=False)])).to(DEV)
        loss = F.binary_cross_entropy_with_logits(
            head(l2norm(P(Z[sel]))).squeeze(-1), y[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    P.eval()
    return P


# ------------------------------------------------------------------ evaluation
def embed(P, z, xy=None):
    """P is None for B0 -- raw DINO with no projection at all."""
    z = z.to(DEV)
    if P is None:
        return l2norm(z)
    if xy is None:
        xy = torch.full((len(z), 2), 0.5)
    return l2norm(P(z, xy.to(DEV)))


@torch.no_grad()
def embed_all(P, idx, batch=20000):
    return torch.cat([embed(P, idx.Z[i:i + batch], idx.POS[i:i + batch]).cpu()
                      for i in range(0, len(idx.Z), batch)])


@torch.no_grad()
def retrieval(idx, E, n_query=150, seed=0):
    """Cross-image same-defect Recall@K.

    The gallery excludes the query's own image, so encoding image instance buys
    nothing -- the failure mode a within-image gallery would let slide.
    """
    rng = np.random.default_rng(seed)
    di = np.where(idx.defect_mask().numpy())[0]
    hits = {1: [], 5: [], 10: []}
    per_obj = []                     # long form: one row per object
    for oi in range(len(idx.classes)):
        g = di[idx.OBJ[di] == oi]
        if len(g) < 50:
            continue
        oh = {1: [], 5: [], 10: []}
        for i in rng.choice(g, size=min(n_query, len(g)), replace=False):
            gal = g[idx.IMG[g] != idx.IMG[i]]
            if len(gal) < 20:
                continue
            order = np.argsort(-(E[gal] @ E[i]).numpy())
            same = idx.TYPE[gal][order] == idx.TYPE[i]
            for k in hits:
                hits[k].append(same[:k].mean())
                oh[k].append(same[:k].mean())
        if oh[1]:
            per_obj.append({"object": idx.classes[oi],
                            **{f"r{k}": float(np.mean(oh[k])) * 100 for k in oh}})
    return ({"recall@1": float(np.mean(hits[1])) * 100,
             "recall@5": float(np.mean(hits[5])) * 100,
             "recall@10": float(np.mean(hits[10])) * 100,
             "knn10": float(np.mean(hits[10])) * 100,
             "n_query": len(hits[1])}, pd.DataFrame(per_obj))


@torch.no_grad()
def oracle_headroom(idx, P, seed=7):
    """Target-defect oracle IN THIS REPRESENTATION -- the number that mattered."""
    out = []
    for c in idx.classes:
        te, tr = idx.te[c], idx.tr[c]
        types = list(te["types"])
        gt = te["gt_frac"].reshape(len(types), -1)
        gh, gw = idx.grids[c]
        grid = torch.from_numpy(xy_grid(gh, gw))
        good = [i for i, t in enumerate(types) if t == "good"]
        bad = [i for i, t in enumerate(types) if t == "bad"]
        if not good or not bad:
            continue
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(bad)); h = len(bad) // 2
        sup = [bad[i] for i in p[:h]]
        ev = good + [bad[i] for i in p[h:]]
        y = np.array([0] * len(good) + [1] * (len(ev) - len(good)))
        nb = torch.from_numpy(
            tr["feats"][rng.choice(tr["feats"].shape[0],
                                   min(4, tr["feats"].shape[0]), False)]
            .reshape(-1, 384).astype(np.float32))
        Sn = [np.asarray(make_dist(nb.to(DEV), DEV)(
            torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)))
            for i in ev]
        base = roc_auc_score(y, [mean_top1p(s) for s in Sn]) * 100
        parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10] for i in sup
                 if (gt[i] > 0.10).sum()]
        if not parts:
            continue
        oe = embed(P, torch.from_numpy(np.concatenate(parts)))
        qe = [embed(P, torch.from_numpy(te["feats"][i].astype(np.float32)), grid)
              for i in ev]
        d = make_dist(oe, DEV)
        orc = roc_auc_score(y, [mean_top1p(Sn[k] - np.asarray(d(qe[k])))
                                for k in range(len(ev))]) * 100
        out.append({"object": c, "baseline": base, "oracle": orc,
                    "headroom": orc - base})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--objects", default="all")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    ckdir = os.path.join(RESULTS, "checkpoints_g10b")
    os.makedirs(ckdir, exist_ok=True)

    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files if a.objects == "all" else a.objects.split(","))

    rows, orc_rows, per_obj = [], [], []
    for cfg in CONFIGS:
        ck = os.path.join(ckdir, f"{cfg}.pt")
        if cfg == "B0":
            P = None
        elif a.eval_only and os.path.exists(ck):
            P = ProjXY(use_xy=(cfg == "B3")).to(DEV)
            P.load_state_dict(torch.load(ck, map_location=DEV)); P.eval()
        else:
            P = (train_binary(idx) if cfg == "B1"
                 else train_metric(idx, cfg, steps=a.steps))
            torch.save(P.state_dict(), ck)
        E = l2norm(idx.Z) if P is None else embed_all(P, idx)
        r, po = retrieval(idx, E)
        po["config"] = cfg
        per_obj.append(po)
        r["config"] = cfg
        rows.append(r)
        print(f"  {cfg}: recall@1 {r['recall@1']:.1f}  recall@5 {r['recall@5']:.1f}"
              f"  (n={r['n_query']})", flush=True)
        o = oracle_headroom(idx, P)
        o["config"] = cfg
        orc_rows.append(o)
        print(f"     oracle headroom median {o.headroom.median():+.2f}", flush=True)

    R = pd.DataFrame(rows)
    O = pd.concat(orc_rows, ignore_index=True)
    R.to_csv(os.path.join(METRICS, "gate10b_retrieval.csv"), index=False)
    O.to_csv(os.path.join(METRICS, "gate10b_oracle.csv"), index=False)
    pf = pd.concat(per_obj, ignore_index=True)
    pf.to_csv(os.path.join(METRICS, "gate10b_per_object.csv"))

    print("\n" + "=" * 90)
    print("GATE 10B-O -- structured defect diagnostic (MVTec v1, defect-type labels)")
    print("=" * 90)
    print(R.set_index("config")[["recall@1", "recall@5", "recall@10"]]
          .round(1).to_string())
    print("\n  cross-image: gallery EXCLUDES the query's own image")
    print("\n  per-object target-defect oracle headroom:")
    print(O.pivot_table(index="object", columns="config",
                        values="headroom").round(2).to_string())
    print("\n  summary:")
    print(O.groupby("config").headroom.agg(
        median="median", mean="mean",
        n_positive=lambda s: int((s > 0).sum())).round(2).to_string())
    pf.to_csv(os.path.join(METRICS, "gate10b_per_object.csv"), index=False)
    print("\n  PRE-REGISTERED: B2 > B0 on Recall@1 in >= 12/15 objects")
    for cfg in ["B1", "B2", "B3"]:
        w = pf.pivot(index="object", columns="config", values="r1")
        d = w[cfg] - w["B0"]
        print(f"    {cfg} - B0: better in {int((d > 0).sum())}/15 objects, "
              f"mean {d.mean():+.1f}")


if __name__ == "__main__":
    main()
