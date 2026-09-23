# -*- coding: utf-8 -*-
"""
Model v0 shared library: data split, adapter training, evaluation.

Protocol note (important): the adapter is trained ONLY on train/good, split into
a BANK half (the memory bank) and a TAILQ half (normal queries whose upper-tail
1-NN distance is penalised). test_public -- good *and* bad -- is never seen
during training; test_public/good is used only to *report* the tail statistics
afterwards. That keeps the diagnostic metric honest and avoids test leakage.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman")
from models import (ResidualAdapter, tail_loss, preserve_loss, mean_dist_loss,
                    separation_loss)
from models.tail_adapter import l2norm, nn_distances

RESULTS = r"E:\work\freshman\results\model_v0"
CACHE = os.path.join(RESULTS, "cache")
PATCH = 14


# --------------------------------------------------------------------------- data
def load_cache(obj):
    tr = np.load(os.path.join(CACHE, f"{obj}_train.npz"), allow_pickle=True)
    te = np.load(os.path.join(CACHE, f"{obj}_test.npz"), allow_pickle=True)
    return tr, te


def split_train(n, bank_frac=0.8, seed=0):
    """Random split of train/good into bank images and tail-query images.

    Random rather than a sorted prefix: train/good is ordered by scene id, so a
    prefix would hand the bank a biased slice of the scene space.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_bank = int(round(n * bank_frac))
    return np.sort(perm[:n_bank]), np.sort(perm[n_bank:])


def mean_top1p(d):
    d = np.asarray(d).ravel()
    k = int(len(d) * 0.01)
    return float(d.max()) if k == 0 else float(np.sort(d)[-k:].mean())


# ----------------------------------------------------------------------- training
def train_adapter(tailq, bank, cfg, device="cuda", log=None, defect=None):
    """Train one adapter. `tailq`/`bank` are (N, P, D) raw frozen features.

    `defect` (optional, flat (Nd, D)) turns on the DEFECT-SUPERVISED probe: a
    hinge that pushes every defect patch above the normal tail. Real few-shot
    detection has no defect labels, so this config exists only to answer whether
    the tail can be suppressed *at all* without destroying defect evidence.

    Queries are sampled as a fixed number of PATCHES rather than a fixed number
    of images: patch count per image ranges from 1216 (fabric) to 4096
    (sheet_metal), so image-based batching would make the step's memory profile
    depend on the object.
    """
    torch.manual_seed(cfg.get("seed", 0))
    gen = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))
    adapter = ResidualAdapter(bank.shape[-1], cfg.get("hidden", 128)).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.get("lr", 2e-3))

    bank_flat = bank.reshape(-1, bank.shape[-1]).to(device)
    if bank_flat.shape[0] > cfg.get("bank_patches", 8192):
        sel = torch.randperm(bank_flat.shape[0], generator=gen, device=device)
        bank_flat = bank_flat[sel[:cfg["bank_patches"]]]
    tailq_flat = tailq.reshape(-1, tailq.shape[-1]).to(device)

    n_q = tailq_flat.shape[0]
    q_patches = min(cfg.get("query_patches", 6144), n_q)
    obj = cfg.get("objective", "tail")     # 'tail' | 'mean'
    alpha = cfg.get("alpha", 0.01)
    lam = cfg.get("lam_preserve", 1.0)
    use_pres = cfg.get("preserve", True)

    hist = []
    for step in range(cfg.get("steps", 600)):
        idx = torch.randint(0, n_q, (q_patches,), generator=gen, device=device)
        z = tailq_flat[idx]
        z_new = adapter(z)
        bank_new = adapter(bank_flat)
        d = nn_distances(z_new, bank_new)
        base = tail_loss(d, alpha) if obj == "tail" else mean_dist_loss(d)
        loss = base
        pres = torch.zeros((), device=device)
        if use_pres:
            pres = preserve_loss(z_new, z)
            loss = loss + lam * pres
        sep = torch.zeros((), device=device)
        lam_def = cfg.get("lam_defect", 0.0)
        if defect is not None and lam_def > 0:
            nd = defect.shape[0]
            d_idx = torch.randint(0, nd, (min(q_patches, nd),), generator=gen,
                                  device=device)
            d_def = nn_distances(adapter(defect[d_idx]), bank_new)
            sep = separation_loss(d, d_def, cfg.get("defect_margin", 0.05))
            loss = loss + lam_def * sep
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == cfg.get("steps", 600) - 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "base": float(base.detach()),
                   "preserve": float(pres.detach()),
                   "separation": float(sep.detach())}
            hist.append(rec)
            if log:
                log(rec)
    return adapter, hist


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def patch_dists_all(adapter, feats, bank_new, device="cuda", q_chunk=4096,
                    b_chunk=16384):
    """Per-patch 1-NN distance for every image, chunked over both sides.

    The bank can be ~700k patches (a full train/good set), so neither the
    (query x bank) matrix nor a single matmul can be materialised whole.
    """
    n, p, dim = feats.shape
    # L2-normalise both sides: the 1-NN distance is a COSINE distance, and the
    # raw DINOv2 dot products are ~1e3, which would silently turn "distance"
    # into an unnormalised inner product (AUROC is rank-based, so it would not
    # even look wrong).
    bank_h = l2norm(bank_new.float()).half()
    out = np.empty((n, p), dtype=np.float32)
    for i in range(n):
        z = l2norm(adapter(feats[i].to(device)).float()).half()
        best = torch.full((p,), -2.0, device=device, dtype=torch.half)
        for j in range(0, bank_h.shape[0], b_chunk):
            b = bank_h[j:j + b_chunk]
            for k in range(0, p, min(q_chunk, p)):
                sim = z[k:k + q_chunk] @ b.T
                best[k:k + q_chunk] = torch.maximum(
                    best[k:k + q_chunk], sim.max(dim=1).values)
        out[i] = (1.0 - best.float()).cpu().numpy()
    return out


def tail_stats(normal_d, defect_d, quantiles=(95, 99, 99.5)):
    """The mechanism metrics the diagnosis says to watch.

    normal_d / defect_d are flat arrays of per-patch 1-NN distances to the
    normal bank, from test_public good / defect patches respectively.
    """
    st = {}
    for q in quantiles:
        thr = float(np.percentile(normal_d, q))
        st[f"normal_p{q}"] = thr
        st[f"frac_defect_below_p{q}"] = float((defect_d < thr).mean())
    st["normal_mean"] = float(normal_d.mean())
    st["defect_mean"] = float(defect_d.mean())
    st["defect_p50"] = float(np.median(defect_d))
    st["defect_p95"] = float(np.percentile(defect_d, 95))
    # how far the typical defect sits above the normal p99, in p99 units
    st["defect_normal_margin"] = st["defect_mean"] - st["normal_p99"]
    st["margin_ratio"] = (st["defect_p50"] / st["normal_p99"]
                          if st["normal_p99"] > 0 else np.nan)
    return st
