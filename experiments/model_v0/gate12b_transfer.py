# -*- coding: utf-8 -*-
"""
Gate 12B -- Whitened Structured Transfer.

Gate 11's universal metric failed everywhere (79.6 vs raw 84.7) and the 15x15
matrix showed 209/210 source->target cells do not help. Gate 12's ablation now
shows the target-normal gain comes ENTIRELY from cross-dimensional covariance:
centring does nothing (-0.58), diagonal z-scoring does nothing (-0.22), full
whitening gives +2.36, and A3 - A2 = +2.58 (12/15).

That suggests Gate 11 failed for a mundane reason rather than a deep one: each
object's defect geometry lives in its own normal covariance frame, so a metric
fitted in one object's raw frame is misaligned in another's. If so, canonicalising
the frames first should make transfer work:

    W0  raw DINO                                         84.7
    W1  target whitening, no learning                    87.1
    W2  raw-space source structured metric (Gate 11)     79.6
    W3  whiten every object, then train a SHARED metric on the whitened
        source objects, evaluate on the whitened target
    W4  whitened target-supervised oracle -- the ceiling MOVES when the
        representation changes, so the old 92.5 must not be reused

Pre-registered BEFORE the run (the comparison that matters is W3 > W1, not
W3 > W0: whitening alone already buys 87.1):
    PASS         W3 - W1 > 0 in >= 10/15 targets and mean >= +1.0
    Strong PASS  >= 12/15 and recovery (mean W3 - mean W1)/(mean W4 - mean W1)
                 >= 25-30%, using the RATIO OF MEANS -- per-target ratios blow
                 up wherever the oracle gap is small
    FAIL         W3 <= W1 while W4 >> W1 -> covariance normalisation removes
                 part of the nuisance but the residual metric is still
                 object-dependent -> then, and only then, a conditional residual

Training health is now mandatory on every learned run, because two separate
zero-gradient traps have already produced confident but meaningless negatives:
    ||theta_final - theta_init||, gradient norm, active-triplet ratio
A run whose parameter update norm is ~0 is INVALID, not FAIL.

Usage: python experiments/model_v0/gate12b_transfer.py [--steps N]
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
from gate10b_structured import Index, ProjXY, embed, l2norm  # noqa: E402
from gate11_transfer import Sub  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g12b")
DEV = "cuda"
EPS_FRAC = 0.05


# ------------------------------------------------------------------ whitening
def whiten_all(idx, cap=20000, eps_frac=EPS_FRAC, seed=0):
    """Whitened, L2-normalised features for every object, computed ONCE.

    Whitening is per-object and fold-independent, so recomputing it inside the
    LOCO loop would be pure waste.
    """
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
        W = (V / np.sqrt(w)) @ V.T
        Zt[g] = l2norm((idx.Z[g] - torch.from_numpy(mu.astype(np.float32)))
                       @ torch.from_numpy(W.astype(np.float32)).T)
    return Zt


# -------------------------------------------------------------------- training
def train_shared(idx, Zs, src_objs, steps, lr=1e-3, seed=0):
    """Shared structured metric on the whitened source objects.

    softplus triplet, never relu(margin + ...): raw features already satisfy the
    hinge, so it sits at exactly zero with zero gradient -- that trap has now
    produced two invalid negatives in this project.
    """
    torch.manual_seed(seed)
    P = ProjXY(use_xy=False).to(DEV)
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    sub = Sub(idx, [idx.classes.index(c) for c in src_objs], "type")
    Zs_dev = Zs.to(DEV)
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
            oth = np.where(sub.OBJ == o)[0]
            oth = oth[sub.TYPE[oth] != sub.TYPE[a]]
            if not alt or not len(oth):
                continue
            cols.append((a, alt[rng.integers(len(alt))],
                         oth[rng.integers(len(oth))]))
        if not cols:
            break
        sel = torch.as_tensor(np.array(cols).T.ravel()).to(DEV)
        h = l2norm(P(Zs_dev[sel]))
        n = len(cols)
        a, pos, neg = h[:n], h[n:2 * n], h[2 * n:]
        d_ap = 1 - (a * pos).sum(1)
        d_an = 1 - (a * neg).sum(1)
        loss = F.softplus(d_ap - d_an).mean()
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(P.parameters(), 1e9))
        gnorm.append(gn)
        active.append(float((d_an > d_ap).float().mean()))
        opt.step()

    delta = sum(float((P.state_dict()[k] - init[k]).norm()) for k in init)
    P.eval()
    return P, {"grad_norm_mean": float(np.mean(gnorm)) if gnorm else 0.0,
               "param_update_norm": delta,
               "active_triplet_ratio": float(np.mean(active)) if active else 0.0}


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
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    V1 = os.path.join(RESULTS, "cache_v1")
    files = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                   if f.endswith("_train.npz"))
    idx = Index(files)
    E0 = l2norm(idx.Z)
    Zs = whiten_all(idx)                      # whitened features, computed once
    print("  whitened features built", flush=True)

    # W4: whitened target-supervised oracle -- ONE training, all objects
    ck4 = os.path.join(CKPT, "W4.pt")
    P4, h4 = train_shared(idx, Zs, files, a.steps)
    torch.save(P4.state_dict(), ck4)
    with torch.no_grad():
        E4 = l2norm(P4(Zs.to(DEV))).cpu()
    print(f"  W4 trained  health {h4}", flush=True)

    targets = files if a.targets == "all" else a.targets.split(",")
    rows = []
    for T in targets:
        oi = files.index(T)
        src = [c for c in files if c != T]
        # W0 / W1 / W4
        for cfg, E in [("W0", E0), ("W1", Zs), ("W4", E4)]:
            h = recall_one(idx, E, oi)
            h.update({"target": T, "config": cfg,
                      "grad_norm_mean": np.nan, "param_update_norm": np.nan,
                      "active_triplet_ratio": np.nan})
            rows.append(h)
        # W3: shared metric on whitened sources
        ck = os.path.join(CKPT, f"{T}_W3.pt")
        P, hh = train_shared(idx, Zs, src, a.steps)
        torch.save(P.state_dict(), ck)
        gidx = idx.defect_idx[idx.OBJ[idx.defect_idx] == oi]   # CPU index: Zs is
        with torch.no_grad():                                 # on CPU here
            Em = l2norm(P(Zs[torch.as_tensor(gidx)].to(DEV))).cpu()
        # P maps 384 -> 128, so the result cannot be spliced into the 384-dim Zs.
        # recall_one only ever indexes rows inside gidx, so a zero-filled
        # index-space tensor is enough.
        Ew = torch.zeros(len(idx.Z), Em.shape[1])
        Ew[gidx] = Em
        h = recall_one(idx, Ew, oi)
        h.update({"target": T, "config": "W3", **hh})
        rows.append(h)
        print(f"  {T:<12} W1 {rows[-2]['recall@1']:5.1f}  W3 {h['recall@1']:5.1f}"
              f"  (update_norm {hh['param_update_norm']:.2f}, "
              f"active {hh['active_triplet_ratio']:.2f})", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate12b_transfer.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate12b_transfer.csv"), index=False)
    P = R.pivot_table(index="target", columns="config", values="recall@1")
    H = R.dropna(subset=["param_update_norm"])

    print("\n" + "=" * 100)
    print("GATE 12B -- whitened structured transfer (cross-image Recall@1)")
    print("=" * 100)
    print(P.round(1).to_string())
    means = P.mean()
    print("\n  means: " + "  ".join(f"{c} {means[c]:.1f}" for c in
                                    ["W0", "W1", "W2", "W3", "W4"] if c in means))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED  (W3 > W1 is the comparison that matters)")
    print("=" * 100)
    d = P["W3"] - P["W1"]
    rec = (means["W3"] - means["W1"]) / (means["W4"] - means["W1"]) * 100
    print(f"  W3 - W1: mean {d.mean():+.2f}  better {int((d > 0).sum())}/{len(d)}")
    print(f"  PASS needs >= 10/15 and mean >= +1.0 -> "
          f"{'PASS' if (int((d > 0).sum()) >= 10 and d.mean() >= 1.0) else 'FAIL'}")
    print(f"  Strong PASS recovery (ratio of means) = {rec:+.1f}%  "
          f"(needs >= 25-30%)")
    print(f"  W3 - W0: mean {(P['W3'] - P['W0']).mean():+.2f}")

    print("\n  health (INVALID if param_update_norm ~ 0):")
    for cfg in ["W3", "W4"]:
        s = H[H.config == cfg]
        if len(s):
            print(f"    {cfg}: update_norm mean {s.param_update_norm.mean():.3f} "
                  f"min {s.param_update_norm.min():.3f}  "
                  f"|grad| {s.grad_norm_mean.mean():.4f}  "
                  f"active-triplet {s.active_triplet_ratio.mean():.3f}")


if __name__ == "__main__":
    main()
