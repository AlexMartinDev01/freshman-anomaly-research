# -*- coding: utf-8 -*-
"""
Gate 9A -- Learned Target-Conditioned Defect Selector.

Trained on MVTec AD v1 (target, bank) episodes, evaluated leave-one-class-out
within v1 and then transferred to MVTec AD 2, whose classes never appear in
training. Frozen DINOv2, frozen normal branch, raw fusion, beta = 1 -- the ONLY
new component is the selector.

  z_t = proj(h_target)      normal-appearance summary of the target
  z_b = proj(h_bank)        defect-bank summary
  s   = MLP([z_t, z_b, z_t*z_b, scalars])   predicted utility

The elementwise product is the "high-order interaction" the whole question turns
on: Gate 8 showed neither the target's normal manifold nor the target's defect
manifold explains which bank helps, so if anything is learnable it has to live in
how the two descriptions interact.

Trained with a pairwise ranking loss over banks within one episode, because the
decision is "which bank ranks highest", not "what is the AUROC number".

Three variants are run, and the ablation is the point:
    target_only  h_t alone            -> can a target predict its best bank?
    bank_only    h_b alone            -> is one bank globally good?
    interaction  both + product       -> the learnable-relation hypothesis

Usage: python experiments/model_v0/gate9a_selector.py [--epochs N] [--seeds N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
DEV = "cuda"
VARIANTS = ["target_only", "bank_only", "interaction"]


class Selector(nn.Module):
    def __init__(self, d_t, d_b, variant, proj=32, hidden=64):
        super().__init__()
        self.variant = variant
        self.proj_t = nn.Linear(d_t, proj) if variant != "bank_only" else None
        self.proj_b = nn.Linear(d_b, proj) if variant != "target_only" else None
        if variant == "target_only":
            d_in = proj
        elif variant == "bank_only":
            d_in = proj
        else:
            # concat + elementwise product: the product is what lets the model
            # express "this bank is good FOR THIS target" rather than
            # "this bank is good" or "this target likes big banks"
            d_in = proj * 3
        self.head = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, h_t, h_b):
        if self.variant == "target_only":
            return self.head(torch.tanh(self.proj_t(h_t))).squeeze(-1)
        if self.variant == "bank_only":
            return self.head(torch.tanh(self.proj_b(h_b))).squeeze(-1)
        zt = torch.tanh(self.proj_t(h_t))
        zb = torch.tanh(self.proj_b(h_b))
        return self.head(torch.cat([zt, zb, zt * zb], dim=-1)).squeeze(-1)


def load(ds):
    p = os.path.join(METRICS, f"selector_{ds}.npz")
    z = np.load(p, allow_pickle=True)
    meta = pd.DataFrame({
        "target": [m[0] for m in z["meta"]], "bank": [m[1] for m in z["meta"]],
        "shot": [m[2] for m in z["meta"]], "seed": [m[3] for m in z["meta"]],
        "size": [m[4] for m in z["meta"]], "baseline": [m[5] for m in z["meta"]]})
    return z["h_t"], z["h_b"], z["utility"], meta


def episode_id(meta):
    # meta["size"] not meta.size: `.size` is a DataFrame attribute (row count)
    # and would silently shadow the column
    return (meta.target + "|" + meta.shot.astype(str) + "|"
            + meta.seed.astype(str) + "|" + meta["size"].astype(str))


def make_pairs(util, groups):
    """Precompute (i, j) index pairs with util[i] > util[j] inside one episode.

    Rebuilding these with a Python loop each epoch dominated the runtime; the
    pair set depends only on the labels, so it is fixed for the whole fit.
    """
    ii, jj = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        ui = util[idx]
        a, b = np.meshgrid(np.arange(len(idx)), np.arange(len(idx)),
                           indexing="ij")
        m = ui[a] > ui[b]
        ii.append(idx[a[m]])
        jj.append(idx[b[m]])
    if not ii:
        return None, None
    return (torch.as_tensor(np.concatenate(ii), device=DEV),
            torch.as_tensor(np.concatenate(jj), device=DEV))


def rank_loss(scores, ii, jj, margin=0.0):
    """Hinge ranking loss: scores[ii] should exceed scores[jj]."""
    return torch.clamp(margin - (scores[ii] - scores[jj]), min=0).mean()


def fit(tr, va, variant, epochs=400, lr=1e-3, wd=1e-4, seed=0, verbose=False):
    torch.manual_seed(seed)
    h_t, h_b, u, meta = tr
    d_t, d_b = h_t.shape[1], h_b.shape[1]
    model = Selector(d_t, d_b, variant).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(h_t, dtype=torch.float32, device=DEV)
    Xb = torch.as_tensor(h_b, dtype=torch.float32, device=DEV)
    U = np.asarray(u, dtype=np.float32)
    ii, jj = make_pairs(U, episode_id(meta).to_numpy())

    tva, bva, uva, mva = va
    Xt_va = torch.as_tensor(tva, dtype=torch.float32, device=DEV)
    Xb_va = torch.as_tensor(bva, dtype=torch.float32, device=DEV)
    iiv, jjv = make_pairs(np.asarray(uva, np.float32),
                          episode_id(mva).to_numpy())

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        s = model(Xt, Xb)
        loss = rank_loss(s, ii, jj)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            sva = model(Xt_va, Xb_va)
            vloss = float(rank_loss(sva, iiv, jjv))
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def pick(model, h_t, h_b, meta, k=1):
    """Rank banks within each episode; return the chosen row indices per episode."""
    with torch.no_grad():
        s = model(torch.as_tensor(h_t, dtype=torch.float32, device=DEV),
                  torch.as_tensor(h_b, dtype=torch.float32, device=DEV)).cpu().numpy()
    out = {}
    for g, idx in pd.Series(range(len(s))).groupby(episode_id(meta).to_numpy()):
        order = np.argsort(-s[idx.to_numpy()])
        out[g] = idx.to_numpy()[order[:k]]
    return out, s


def evaluate(model, h_t, h_b, u, meta):
    """Per-episode achieved utility, against baseline / random / oracle."""
    chosen, _ = pick(model, h_t, h_b, meta, k=1)
    eid = episode_id(meta).to_numpy()
    rows = []
    for g, idx in pd.Series(range(len(u))).groupby(eid):
        i = idx.to_numpy()
        rows.append({
            "episode": g, "target": meta.target.iloc[i[0]],
            "shot": meta.shot.iloc[i[0]], "seed": meta.seed.iloc[i[0]],
            "size": meta["size"].iloc[i[0]], "baseline": meta.baseline.iloc[i[0]],
            "learned_top1": u[chosen[g][0]],
            "mean_all": float(np.mean(u[i])),      # random-bank expectation
            "median_all": float(np.median(u[i])),
            "oracle_best": float(np.max(u[i])),
            "best_bank": meta.bank.iloc[i[np.argmax(u[i])]],
            "picked_bank": meta.bank.iloc[chosen[g][0]],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--holdout", default=None,
                    help="single target class to hold out (default: all)")
    ap.add_argument("--dataset", default="v1", choices=["v1", "ad2"],
                    help="ad2 = leave-one-object-out WITHIN the AD2 regime, "
                         "where external banks actually have headroom to help. "
                         "v1 is near its ceiling (baseline 96, oracle 97), so a "
                         "v1-trained selector may simply never see a case where "
                         "choosing well matters.")
    args = ap.parse_args()
    ds = args.dataset

    h_t, h_b, u, meta = load(ds)
    tgt = np.array(meta.target)
    all_classes = sorted(set(tgt))
    print(f"{ds} table: {len(u)} rows, {len(all_classes)} classes, "
          f"{meta.groupby(['target','bank']).ngroups} (target,bank) pairs")

    # standardise on the whole v1 table: v1 is the TRAINING dataset, so its
    # statistics are legitimate to use; AD2 never contributes a scaling factor
    mu_t, sd_t = h_t.mean(0), h_t.std(0) + 1e-6
    mu_b, sd_b = h_b.mean(0), h_b.std(0) + 1e-6
    Ht = ((h_t - mu_t) / sd_t).astype(np.float32)
    Hb = ((h_b - mu_b) / sd_b).astype(np.float32)

    holdouts = [args.holdout] if args.holdout else all_classes
    all_out = []
    for T in holdouts:
        te_m = tgt == T
        tr_m = ~te_m
        # validation classes: two training targets held out for early stopping
        others = [c for c in all_classes if c != T]
        val_cls = set(others[::max(1, len(others) // 3)][:2])
        vm = np.array([c in val_cls for c in tgt]) & tr_m
        fit_m = tr_m & ~vm
        tr = (Ht[fit_m], Hb[fit_m], u[fit_m], meta[fit_m].reset_index(drop=True))
        va = (Ht[vm], Hb[vm], u[vm], meta[vm].reset_index(drop=True))
        te = (Ht[te_m], Hb[te_m], u[te_m], meta[te_m].reset_index(drop=True))

        for v in VARIANTS:
            for sd_ in range(args.seeds):
                m, vl = fit(tr, va, v, epochs=args.epochs, seed=sd_)
                df = evaluate(m, *te)
                df["variant"] = v
                df["seed"] = sd_
                df["heldout"] = T
                all_out.append(df)
        print(f"  held out {T:<14} done", flush=True)

    out = pd.concat(all_out, ignore_index=True)
    out.to_csv(os.path.join(METRICS, f"gate9a_loco_{ds}.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"GATE 9A -- leave-one-class-out on {ds} (per-episode utility)")
    print("=" * 100)
    for v in VARIANTS:
        s = out[out.variant == v]
        g = s.groupby("episode").agg(
            base=("baseline", "first"), learned=("learned_top1", "mean"),
            rand=("mean_all", "mean"), oracle=("oracle_best", "mean"))
        # aggregate per held-out class so classes do not get double counted
        per = s.groupby(["heldout", "variant"]).agg(
            base=("baseline", "mean"), learned=("learned_top1", "mean"),
            rand=("mean_all", "mean"), oracle=("oracle_best", "mean")).round(1)
        print(f"\n  --- {v} ---")
        print(per.to_string())
        d = (per.learned - per.rand)
        print(f"  learned - random : mean {d.mean():+.1f}  "
              f"({int((d > 0).sum())}/{len(d)} classes better)")
        gap = (per.oracle - per.rand)
        closed = (d / gap.replace(0, np.nan) * 100)
        print(f"  oracle gap closed: mean {closed.mean():.1f}%")
        print(f"  learned beats baseline: {int((per.learned > per.base).sum())}"
              f"/{len(per)}")


if __name__ == "__main__":
    main()
