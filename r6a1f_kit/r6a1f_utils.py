# -*- coding: utf-8 -*-
"""R6-A1F helpers -- dense 12-block causal propagation trace.

Everything here obeys two contracts inherited from R6-A1E:

  1. `valid` masks are ALWAYS flat (N,) bool. A1E shipped a crash because a
     2-D (H,W) score map met a flat mask at `topmean`. `assert_flat` enforces it.
  2. Never call the wrapper's `extract_features` (it `.squeeze()`s and silently
     drops rank for a batch of 1). Go through `get_intermediate_layers`.

Layer index convention, fixed by the project's pre-existing pin drop
(`cache_multilayer.py`): mid=5 = 50% depth, midlate=8 = 75%, final=11 = 100%.
`early` is defined by arithmetic continuation as 25% depth = block 2. It is a
fixed label, NOT the same thing as `onset_layer`, which is a result.
"""
from __future__ import annotations

import numpy as np
import torch

LAYER_INDEX = list(range(12))
LAYER_NAME = {2: "early", 5: "mid", 8: "midlate", 11: "final"}
# A1E cache layer name -> block index, for the parity gate
A1E_CACHE_BLOCK = {"mid": 5, "midlate": 8, "final": 11}

EXPECT_N = {"screw": 1024, "cable": 1024, "bottle": 1024,
            "macaroni2": 1536, "chewinggum": 1216, "pcb2": 1312}

IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def assert_flat(v, name="valid"):
    v = np.asarray(v)
    if v.ndim != 1:
        raise AssertionError(f"{name} must be flat 1-D, got shape {v.shape}")
    return v.astype(bool)


def _l2n_last(a):
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def extract_dense(model, tensors, batch_size=4):
    """One forward per image yields every block.

    Returns
        outs : list of (12, N, 384) float32 arrays, one per input image
        x0s  : list of (N, 384) float32 patch_embed outputs (pre-block-0 token
               space) -- the perturbation actually delivered to the transformer,
               which is a better covariate than raw pixel L2.
    """
    outs, x0s = [], []
    for s in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[s:s + batch_size], dim=0).to(model.device)
        with torch.inference_mode():
            x0 = model.model.patch_embed(batch)
            z = model.model.get_intermediate_layers(batch, n=LAYER_INDEX)
        for i in range(batch.shape[0]):
            outs.append(np.stack([z[l][i].float().cpu().numpy()
                                  for l in range(len(LAYER_INDEX))], axis=0))
            x0s.append(x0[i].float().cpu().numpy())
        del batch, z, x0
    return outs, x0s


def dense_drift_full(raw12, cf12):
    """(12, N) per-token drift over ALL tokens.

    Drift is a property of the token: the radius only selects which tokens get
    averaged. Compute once per cell and derive every radius and ring by masking.
    """
    r = _l2n_last(np.asarray(raw12, np.float32))
    c = _l2n_last(np.asarray(cf12, np.float32))
    return 1.0 - np.einsum("lnd,lnd->ln", r, c)


def dense_rel_l2(raw12, cf12, valid):
    """(12,) ||dF|| / ||F_raw|| over the valid tokens.

    `get_intermediate_layers` applies the final LayerNorm to every selected
    block, which deliberately removes magnitude. A pure cosine curve can hide a
    large residual-stream magnitude change, so report this next to it.
    """
    v = assert_flat(valid)
    d = np.asarray(raw12, np.float32)[:, v, :]
    c = np.asarray(cf12, np.float32)[:, v, :]
    num = np.linalg.norm(c - d, axis=2).mean(axis=1)
    den = np.clip(np.linalg.norm(d, axis=2).mean(axis=1), 1e-12, None)
    return (num / den).astype(np.float64)


def topmean_rows(per_token, valid, alpha):
    """(12,) top-alpha mean of per-token drift -- the SAME functional as the
    image score, so the mechanism and the outcome are read with one functional.
    """
    v = assert_flat(valid)
    if not v.any():
        return np.full(per_token.shape[0], np.nan)
    out = np.empty(per_token.shape[0], np.float64)
    for l in range(per_token.shape[0]):
        x = per_token[l][v]
        k = min(max(1, int(round(alpha * len(x)))), len(x))
        out[l] = np.partition(x, len(x) - k)[-k:].mean()
    return out


def input_delta_pixel(raw_t, cf_t, pixmask):
    """Mean |raw - donor| over the replaced pixels, in denormalized [0,1] units.

    This is the protocol-mandated 'raw input difference'.
    """
    m = torch.from_numpy(np.asarray(pixmask, bool))
    if not bool(m.any()):
        return 0.0
    d = (raw_t - cf_t).abs()[:, m].cpu().numpy()          # (3, npx)
    return float((d * IMAGENET_STD).mean())


def input_delta_token(x0_raw, x0_cf, patchmask):
    """Mean ||d x0|| over the replaced patch tokens (patch_embed output space)."""
    m = np.asarray(patchmask, bool).reshape(-1)
    if not m.any():
        return 0.0
    d = np.asarray(x0_cf, np.float32)[m] - np.asarray(x0_raw, np.float32)[m]
    return float(np.linalg.norm(d, axis=1).mean())


# ---------------------------------------------------------------- statistics

def fit_linear(delta, drift, min_n=30, min_rel_sd=0.05):
    """OLS drift ~ a + b*delta with degeneracy guards.

    Returns None when the fit is not identifiable: too few points, or the
    covariate has too little spread for a line to mean anything (e.g. r=0 with a
    two-patch defect). Callers must emit a flag rather than a number.
    """
    delta = np.asarray(delta, float)
    drift = np.asarray(drift, float)
    n = len(delta)
    if n < min_n:
        return None
    mu = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1))
    if not np.isfinite(sd) or sd < min_rel_sd * max(abs(mu), 1e-12):
        return None
    A = np.vstack([np.ones(n), delta]).T
    coef, *_ = np.linalg.lstsq(A, drift, rcond=None)
    pred = A @ coef
    ss_res = float(((drift - pred) ** 2).sum())
    ss_tot = float(((drift - drift.mean()) ** 2).sum())
    return dict(a=float(coef[0]), b=float(coef[1]),
                r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                n=n, sd_delta=sd, mean_delta=mu)


def resid_from_fit(fit, delta_bad, drift_bad):
    return np.asarray(drift_bad, float) - (fit["a"] + fit["b"] * np.asarray(delta_bad, float))


def permutation_null(delta, drift, cluster, is_bad, n_perm=2000, seed=0):
    """Within-cluster label permutation.

    Under H0 -- the counterfactual's far-field consequence is a fixed function of
    the delivered input change, identical for bad and good images -- the mean
    residual is 0. Permuting WHICH series in each (bad_image, donor, radius)
    cluster is labelled 'bad' preserves the cluster structure, the delta
    distribution, the mask, the donor and the sample sizes; it destroys only the
    label. One-sided p for 'bad images drift more than magnitude explains'.
    """
    delta = np.asarray(delta, float)
    drift = np.asarray(drift, float)
    is_bad = np.asarray(is_bad, bool)
    cluster = np.asarray(cluster)

    # primary statistic on the observed labels (primary fit spec, with guards)
    fit0 = fit_linear(delta[~is_bad], drift[~is_bad])
    if fit0 is None:
        return dict(obs=np.nan, p=np.nan, n_perm=0)
    obs = float(np.mean(resid_from_fit(fit0, delta[is_bad], drift[is_bad])))

    # index lists per cluster so a permutation only reshuffles inside a cluster
    order = np.argsort(cluster, kind="stable")
    cl_sorted = cluster[order]
    bounds = np.flatnonzero(np.r_[True, cl_sorted[1:] != cl_sorted[:-1], True])
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    flat = np.concatenate(groups)
    sizes = np.array([len(g) for g in groups])
    offs = np.r_[0, np.cumsum(sizes)[:-1]]
    C, n = len(groups), len(delta)

    # vectorised null: draw one "bad" per cluster per permutation with a single
    # RNG call, then closed-form OLS for every permutation at once. Same null
    # distribution as the naive per-permutation loop, ~1000x faster.
    rng = np.random.default_rng(seed)
    pos = (rng.random((n_perm, C)) * sizes).astype(np.int64)
    hit = flat[offs + pos]                                  # (n_perm, C)
    labf = np.zeros((n_perm, n), np.float64)
    np.put_along_axis(labf, hit, 1.0, axis=1)
    goodf = 1.0 - labf
    ng = float(n - C)

    gx = goodf @ delta
    gxx = goodf @ (delta * delta)
    gy = goodf @ drift
    gxy = goodf @ (delta * drift)
    denom = gxx - gx * gx / ng
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (gxy - gx * gy / ng) / denom
        a = (gy - b * gx) / ng
        rb = drift[hit] - (a[:, None] + b[:, None] * delta[hit])
        null = rb.mean(axis=1)
    null = null[np.isfinite(null)]
    if not len(null):
        return dict(obs=obs, p=np.nan, n_perm=0)
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    p_lo = float((1 + (null <= obs).sum()) / (1 + len(null)))   # lower tail
    return dict(obs=obs, p=p, p_lower=p_lo, n_perm=int(len(null)),
                null_mean=float(np.mean(null)), null_sd=float(np.std(null)))


def cluster_bootstrap(delta, drift, cluster, is_bad, n_boot=2000, seed=0):
    """Bootstrap the mean residual by resampling bad images (clusters)."""
    delta = np.asarray(delta, float)
    drift = np.asarray(drift, float)
    is_bad = np.asarray(is_bad, bool)
    cluster = np.asarray(cluster)
    fit = fit_linear(delta[~is_bad], drift[~is_bad])
    if fit is None:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_clusters=0)
    resid = resid_from_fit(fit, delta[is_bad], drift[is_bad])
    cl = cluster[is_bad]
    keys, inv = np.unique(cl, return_inverse=True)
    nk = len(keys)
    # per-cluster means, then a vectorised bootstrap over clusters
    sums = np.bincount(inv, weights=resid, minlength=nk)
    cnts = np.bincount(inv, minlength=nk).astype(float)
    cmean = sums / np.clip(cnts, 1, None)
    rng = np.random.default_rng(seed)
    pick = rng.integers(nk, size=(n_boot, nk))
    out = cmean[pick].mean(axis=1)
    return dict(mean=float(resid.mean()), lo=float(np.quantile(out, .025)),
                hi=float(np.quantile(out, .975)), n_clusters=nk)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan, np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    # two-sided t approximation
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    from scipy.stats import t as tdist
    return r, float(2 * tdist.sf(abs(t), n - 2))
