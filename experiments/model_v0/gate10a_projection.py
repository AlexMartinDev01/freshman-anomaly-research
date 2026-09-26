# -*- coding: utf-8 -*-
"""
Gate 10A -- Cross-Object Anomaly Semantic Projection.

Gate 9A/9B established that the learnable part of a (target, donor) relation is
the DONOR identity, never the target conditioning, and that a small conditional
MLP cannot turn external defects into target-specific support even with +147
AUROC of headroom available. The reading is that anomaly-relevant content is
present in external defects but is swamped by object/category identity in raw
DINO space.

So change the interface, not the capacity. Keep the normal branch completely
untouched -- Gate 6A showed that touching the normal representation invites
support overfitting -- and project ONLY the defect branch:

    z --P_theta--> h            (384 -> 256 -> 128)

    L_anom   BCE(anom_head(h), is_defect)      keep "this is an anomaly"
    L_obj    CE(obj_head(GRL(h)), object_id)   forget "this is fabric/bottle"

The GRL is what makes this different from the old hand-built whitening: nothing
is guessed about which directions carry identity, the split is learned from
source anomaly supervision.

Configs, pre-registered before the run:
    R0  raw DINO defect branch, no projection
    R1  projection trained with L_anom only
    R2  projection trained with L_anom + identity removal   <- the candidate

Success requires R2 > R1 AND R2 > baseline on a majority of eligible categories
AND mean oracle-gap closure > 0. Oracle headroom is reported for every config;
that is now a permanent discipline after the v1 testbed of gate 9B turned out to
have +0.49 of it.

The identity probe is a separate, mandatory check. If object identity drops but
anomaly information drops with it, this is the Gate 4 whitening failure again and
must be judged a failure regardless of the downstream number.

Usage: python experiments/model_v0/gate10a_projection.py --dataset v1 --folds all
"""
import argparse
import os
import sys
import time

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
AD2 = os.path.join(RESULTS, "cache")
METRICS = os.path.join(RESULTS, "metrics")
DEV = "cuda"
CONFIGS = ["R0", "R1", "R2"]


# --------------------------------------------------------------- gradient reversal
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x, lam):
    return _GRL.apply(x, lam)


class Proj(nn.Module):
    """384 -> 256 -> 128. Deliberately minimal; the interface is the variable."""

    def __init__(self, d_in=384, d_hid=256, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_out))

    def forward(self, z):
        return self.net(z)


# ------------------------------------------------------------------------- data
class Pool:
    """Per-class normal/defect patch pools."""

    def __init__(self, classes, root, cap_normal=4000, seed=0):
        self.classes = classes
        self.normal, self.defect, self.good_idx, self.bad_idx, self.te, self.tr = \
            {}, {}, {}, {}, {}, {}
        rng = np.random.default_rng(seed)
        for c in classes:
            tr = materialize(np.load(os.path.join(root, f"{c}_train.npz"),
                                     allow_pickle=True))
            te = materialize(np.load(os.path.join(root, f"{c}_test.npz"),
                                     allow_pickle=True))
            self.tr[c], self.te[c] = tr, te
            types = list(te["types"])
            gt = te["gt_frac"].reshape(len(types), -1)
            self.good_idx[c] = [i for i, t in enumerate(types) if t == "good"]
            self.bad_idx[c] = [i for i, t in enumerate(types) if t == "bad"]
            np_norm = tr["feats"].reshape(-1, 384).astype(np.float32)
            if len(np_norm) > cap_normal:
                np_norm = np_norm[rng.choice(len(np_norm), cap_normal, False)]
            self.normal[c] = torch.from_numpy(np_norm)
            parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                     for i in range(len(types))
                     if types[i] == "bad" and (gt[i] > 0.10).sum()]
            self.defect[c] = (torch.from_numpy(np.concatenate(parts)) if parts
                              else torch.zeros((0, 384)))
        print(f"  pool: {len(classes)} classes, "
              f"{sum(len(v) for v in self.defect.values())} defect patches",
              flush=True)

    def defect_of(self, c, idx):
        te = self.te[c]
        gt = te["gt_frac"].reshape(len(te["types"]), -1)
        parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                 for i in np.asarray(idx) if (gt[i] > 0.10).sum()]
        return (torch.from_numpy(np.concatenate(parts)) if parts
                else torch.zeros((0, 384)))

    def feats(self, c, idx):
        return torch.from_numpy(
            self.te[c]["feats"][np.asarray(idx)].astype(np.float32))

    def shot(self, c, shot, rng):
        n = self.tr[c]["feats"].shape[0]
        k = n if shot == -1 else min(shot, n)
        return torch.from_numpy(
            self.tr[c]["feats"][rng.choice(n, k, False)].reshape(-1, 384)
            .astype(np.float32))


def split_bad(pool, c, rng):
    b = pool.bad_idx[c]
    p = rng.permutation(len(b))
    h = len(b) // 2
    return [b[i] for i in p[:h]], [b[i] for i in p[h:]]


# ---------------------------------------------------------------------- training
def train_proj(pool, heldout_classes, config, steps=1500, bs=512, lr=1e-3,
               lam_obj=1.0, seed=0, log_every=300):
    """Train P_theta on the meta-train classes (never the held-out one)."""
    torch.manual_seed(seed)
    cls = list(heldout_classes)
    obj_of = {c: i for i, c in enumerate(cls)}
    P = Proj().to(DEV)
    anom = nn.Linear(128, 1).to(DEV)
    objh = nn.Linear(128, len(cls)).to(DEV)
    params = list(P.parameters()) + list(anom.parameters())
    if config == "R2":
        params += list(objh.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    for step in range(steps):
        # balanced batch: equal normal and defect, spread over sampled classes
        nb = bs // 2
        pick = [cls[i] for i in rng.integers(0, len(cls), size=min(8, len(cls)))]
        # keep z / y_anom / y_obj built in ONE order: the object head must see
        # normal and defect patches alike, and a separate concatenation silently
        # misaligned it (batch 512 vs target 256)
        zp, yp, op = [], [], []
        for c in pick:
            nm, df = pool.normal[c], pool.defect[c]
            if len(nm) == 0 or len(df) == 0:
                continue
            k = max(1, nb // len(pick))
            zp += [nm[torch.from_numpy(rng.choice(len(nm), k, False))],
                   df[torch.from_numpy(rng.choice(len(df), k, False))]]
            yp += [torch.zeros(k), torch.ones(k)]
            op += [torch.full((k,), obj_of[c], dtype=torch.long)] * 2
        if not zp:
            continue
        z = torch.cat(zp).to(DEV)
        y = torch.cat(yp).to(DEV)
        yo = torch.cat(op).to(DEV)

        h = P(z)
        l_anom = F.binary_cross_entropy_with_logits(anom(h).squeeze(-1), y)
        loss = l_anom
        l_obj = torch.zeros((), device=DEV)
        if config == "R2":
            l_obj = F.cross_entropy(objh(grl(h, lam_obj)), yo)
            loss = loss + l_obj
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"    step {step:>5}  anom {float(l_anom):.4f}  "
                  f"obj {float(l_obj):.4f}", flush=True)
    P.eval()
    return P


# -------------------------------------------------------------------- evaluation
@torch.no_grad()
def make_embed(P, config):
    """One projector per call. Applying the wrong one silently contaminates the
    other config's column, so the config is bound here rather than looked up."""
    if config == "R0" or P is None:
        return lambda X: l2norm(X.to(DEV))
    return lambda X: l2norm(P(X.to(DEV)))


@torch.no_grad()
def evaluate(pool, heldout, P, config, shots=(1, 2, 4), n_bank=256, seed=999):
    emb = make_embed(P, config)
    rows = []
    cls = [c for c in pool.classes if c != heldout]
    pool_cache = {c: pool.defect[c] for c in cls}
    for shot in shots:
        for rep in range(3):
            rng = np.random.default_rng(seed + shot * 100 + rep)
            sup_bad, _ = split_bad(pool, heldout, rng)
            ev = pool.good_idx[heldout] + [b for b in pool.bad_idx[heldout]
                                           if b not in set(sup_bad)]
            q_raw = pool.feats(heldout, ev)
            y = np.array([0] * len(pool.good_idx[heldout])
                         + [1] * (len(ev) - len(pool.good_idx[heldout])))
            nb_raw = pool.shot(heldout, shot, rng)
            Sn = [np.asarray(make_dist(nb_raw.to(DEV), DEV)(q_raw[i].to(DEV)))
                  for i in range(len(q_raw))]
            qq = [emb(q_raw[i]) for i in range(len(q_raw))]

            def auroc(Sd):
                sc = np.array([mean_top1p(Sn[i] - Sd[i]) for i in range(len(ev))])
                return roc_auc_score(y, sc) * 100

            row = {"heldout": heldout, "shot": shot, "rep": rep,
                   "baseline": auroc([np.zeros_like(Sn[0])] * len(ev)),
                   "n_sources": 0}
            # oracle UNDER THIS REPRESENTATION
            o_raw = pool.defect_of(heldout, sup_bad)
            if len(o_raw):
                d = make_dist(emb(o_raw), DEV)
                row["oracle"] = auroc([np.asarray(d(qq[i]))
                                       for i in range(len(q_raw))])
            per_src = []
            for c in cls:
                src = pool_cache[c]
                if len(src) == 0:
                    continue
                sel = rng.choice(len(src), size=min(n_bank, len(src)),
                                 replace=False)
                d = make_dist(emb(src[sel]), DEV)
                per_src.append(auroc([np.asarray(d(qq[i]))
                                      for i in range(len(q_raw))]))
            row["score"] = float(np.mean(per_src)) if per_src else np.nan
            row["n_sources"] = len(per_src)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ probes
def probes(pool, heldout_classes, P, n=4000, seed=0):
    """Object-ID and anomaly decodability from P_theta(z) vs raw z.

    Mandatory check: if identity falls but anomaly information falls with it,
    this is the Gate 4 whitening failure and the gate must fail regardless of
    the downstream AUROC.
    """
    rng = np.random.default_rng(seed)
    cls = list(heldout_classes)
    Z, Y_anom, Y_obj = [], [], []
    for i, c in enumerate(cls):
        nm, df = pool.normal[c], pool.defect[c]
        k = min(n, len(nm), max(len(df), 1) * 3)
        if len(nm) < 10 or len(df) < 10:
            continue
        Z.append(nm[rng.choice(len(nm), k, False)])
        Y_anom.append(np.zeros(k, dtype=np.int64))
        Y_obj.append(np.full(k, i, dtype=np.int64))
        kd = min(k, len(df))
        Z.append(df[rng.choice(len(df), kd, False)])
        Y_anom.append(np.ones(kd, dtype=np.int64))
        Y_obj.append(np.full(kd, i, dtype=np.int64))
    Z = torch.cat(Z).to(DEV)
    Y_anom = torch.cat([torch.as_tensor(a) for a in Y_anom]).to(DEV)
    Y_obj = torch.cat([torch.as_tensor(a) for a in Y_obj]).to(DEV)
    n_tr = int(len(Z) * 0.7)
    perm = torch.randperm(len(Z), device=DEV)
    tr, te = perm[:n_tr], perm[n_tr:]

    out = {}
    with torch.no_grad():          # frozen features: the probe must not
        feats = {"raw": l2norm(Z).detach(),        # backprop into P_theta, and
                 "proj": l2norm(P(Z)).detach()}    # the graph must survive
    for name, X in feats.items():                  # 400 separate backwards
        acc = {}
        for head, y in [("obj", Y_obj), ("anom", Y_anom)]:
            torch.manual_seed(0)
            lin = nn.Linear(X.shape[1], int(y.max()) + 1).to(DEV)
            opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-4)
            for _ in range(400):
                opt.zero_grad()
                F.cross_entropy(lin(X[tr]), y[tr]).backward()
                opt.step()
            with torch.no_grad():
                pr = lin(X[te]).argmax(1)
            acc[head] = float((pr == y[te]).float().mean())
        out[name] = acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v1", choices=["v1", "ad2"])
    ap.add_argument("--folds", default="all")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--probe-only", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    root = V1 if ds == "v1" else AD2
    os.makedirs(os.path.join(RESULTS, "checkpoints_g10a"), exist_ok=True)

    classes = sorted(f[:-len("_train.npz")] for f in os.listdir(root)
                     if f.endswith("_train.npz"))
    pool = Pool(classes, root)
    folds = classes if a.folds == "all" else a.folds.split(",")

    rows, prb = [], []
    for heldout in folds:
        meta = [c for c in classes if c != heldout]
        for seed in range(a.seeds):
            ck = os.path.join(RESULTS, "checkpoints_g10a",
                              f"{ds}_{heldout}_s{seed}.pt")
            Ps = {}
            for cfg in CONFIGS:
                if cfg == "R0":
                    Ps[cfg] = None
                    continue
                t0 = time.time()
                P = train_proj(pool, meta, cfg, steps=a.steps, seed=seed)
                torch.save(P.state_dict(), ck.replace(".pt", f"_{cfg}.pt"))
                print(f"  {heldout} {cfg} trained in {time.time()-t0:.0f}s",
                      flush=True)
                Ps[cfg] = P
            # one projector per config, each bound to its own representation
            merged = None
            for cfg in CONFIGS:
                df = evaluate(pool, heldout, Ps[cfg], cfg)
                df = df.rename(columns={"score": cfg, "oracle": f"oracle_{cfg}"})
                # baseline is config-independent: carry it only on the first
                # merge, otherwise the join duplicates the column
                keep = (["heldout", "shot", "rep", "baseline", cfg, f"oracle_{cfg}"]
                        if merged is None else
                        ["heldout", "shot", "rep", cfg, f"oracle_{cfg}"])
                df = df[keep]
                merged = df if merged is None else merged.merge(
                    df, on=["heldout", "shot", "rep"], how="left")
                print(f"  {heldout} {cfg} evaluated", flush=True)
            rows.append(merged)
            # probes on R1 and R2 projectors
            for cfg in ["R1", "R2"]:
                r = probes(pool, meta, Ps[cfg])
                prb.append({"heldout": heldout, "config": cfg,
                            "obj_raw": r["raw"]["obj"], "obj_proj": r["proj"]["obj"],
                            "anom_raw": r["raw"]["anom"],
                            "anom_proj": r["proj"]["anom"]})
                print(f"  {heldout} {cfg} probe: objID {r['raw']['obj']:.3f} -> "
                      f"{r['proj']['obj']:.3f} | anom {r['raw']['anom']:.3f} -> "
                      f"{r['proj']['anom']:.3f}", flush=True)
        pd.concat(rows, ignore_index=True).to_csv(
            os.path.join(METRICS, f"gate10a_{ds}_raw.csv"), index=False)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(METRICS, f"gate10a_{ds}.csv"), index=False)
    if prb:
        pd.DataFrame(prb).to_csv(
            os.path.join(METRICS, f"gate10a_{ds}_probes.csv"), index=False)
    print(f"\nwrote gate10a_{ds}.csv rows={len(out)}")


if __name__ == "__main__":
    main()
