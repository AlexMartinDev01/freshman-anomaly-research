# -*- coding: utf-8 -*-
"""
Pseudo-defect sources for Proxy-E.

Config E showed that defect-aware separation supervision works: the adapter
simultaneously suppressed the normal tail and *raised* the defect response. But
E consumed real defect patches from half the test_public bad images, which
few-shot detection does not have.

Proxy-E keeps the adapter, the tail objective and the separation objective
IDENTICAL and replaces only the defect source. Any change in the outcome is
therefore attributable to the information, not to the machinery.

  F1 texture    local noise / blur / colour shift inside a blob mask
  F2 structural local content replacement (CutPaste), rotation, erase
  F3 feature    perturb normal patch features along directions OUTSIDE the
                normal subspace, magnitude calibrated to the normal tail

F3 encodes the diagnosis directly: legitimate normal variation moves a patch
*along* the normal manifold, so compress those directions and preserve the
residual. F1/F2 are the cheap image-space baselines that must be ruled out
first.

Everything below is generated from train/good only.

Usage: python experiments/model_v0/proxy_defects.py [obj ...]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model
from ad2_pipeline import AD2_ROOT, list_png
from tail_calib import CACHE, RESULTS, load_cache, split_train

PROXY_DIR = os.path.join(RESULTS, "proxy")
OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
N_AUG_IMGS = 80          # train/good images to corrupt per object
SEED = 0
COVER_LO, COVER_HI = 0.01, 0.05      # blob area as a fraction of the image
PATCH_MIN_COVER = 0.5                # a patch counts as pseudo-defect at >=50%
# Calibration target, as a multiple of the normal p99 distance. This must be
# BELOW the separation hinge threshold (normal_p99 + margin), or the hinge is
# satisfied at step 0, its gradient is zero and the proxy degenerates into
# config D. It is also the honest target: real defects in the failing
# categories sit at ~0.5x the normal p99 (wallplugs defect_mean 0.119 vs
# normal_p99 0.242) -- they are not far outliers, they are indistinguishable.
F3_TARGET_MULT = 0.6
F3_ALPHAS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
F3_N_PATCHES = 4000
LR_K = (5, 20)         # rank-k variants of the off-manifold proxy (F4)
RANK_K = (1, 3, 5, 10, 20, 40, 70, 120)   # Probe 1: rank scan (extended to
                                          # reach ~0.99 explained variance)
PROTO_K = (1, 2, 4, 8, 16, 32, 64)   # Probe 2: prototype count
DIAG_BANK = 20000      # bank subsample used for the proxy-distance report


# ------------------------------------------------------------------ masks
def blob_mask(h, w, rng):
    """A smooth, localised blob covering COVER_LO..COVER_HI of the image."""
    target = rng.uniform(COVER_LO, COVER_HI) * h * w
    m = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(1, 4)):
        ax = int(rng.integers(8, max(9, w // 6)))
        ay = int(rng.integers(8, max(9, h // 6)))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.ellipse(m, (cx, cy), (ax, ay), int(rng.integers(0, 180)), 0, 360, 1, -1)
        if (m > 0).sum() >= target:
            break
    m = cv2.GaussianBlur(m * 255, (0, 0), 5)
    return (m > 127).astype(np.uint8)


# ------------------------------------------------------------- F1: texture
def texture_proxy(img, rng):
    h, w = img.shape[:2]
    m = blob_mask(h, w, rng).astype(bool)
    kind = rng.integers(0, 4)
    out = img.copy()
    if kind == 0:                                    # gaussian noise
        noise = rng.normal(0, 28, img.shape)
        out[m] = np.clip(img[m].astype(np.float32) + noise[m], 0, 255)
    elif kind == 1:                                  # blur
        out[m] = cv2.GaussianBlur(img, (0, 0), 4)[m]
    elif kind == 2:                                  # per-channel colour shift
        g = rng.uniform(0.7, 1.3, 3)
        out[m] = np.clip(img[m].astype(np.float32) * g, 0, 255)
    else:                                            # texture from elsewhere
        sy, sx = rng.integers(0, h), rng.integers(0, w)
        sh, sw = img.shape[0], img.shape[1]
        src = np.roll(np.roll(img, -sy, axis=0), -sx, axis=1)[:sh, :sw]
        out[m] = src[m]
    return out.astype(np.uint8), m.astype(np.float32)


# ---------------------------------------------------------- F2: structural
def structural_proxy(img, rng, donor=None):
    h, w = img.shape[:2]
    kind = int(rng.integers(0, 4))
    ph = int(rng.integers(h // 8, h // 3))
    # a 90/270 rotation swaps the region's dimensions, so that branch needs a
    # square region or the rotated block will not fit back into the hole
    pw = ph if kind == 2 else int(rng.integers(w // 8, w // 3))
    y0 = int(rng.integers(0, max(1, h - ph)))
    x0 = int(rng.integers(0, max(1, w - pw)))
    m = np.zeros((h, w), np.float32)
    m[y0:y0 + ph, x0:x0 + pw] = 1.0
    out = img.copy()
    if kind == 0 and donor is not None:              # CutPaste from another image
        dh, dw = donor.shape[:2]
        sy = int(rng.integers(0, max(1, dh - ph)))
        sx = int(rng.integers(0, max(1, dw - pw)))
        out[y0:y0 + ph, x0:x0 + pw] = donor[sy:sy + ph, sx:sx + pw]
    elif kind == 1:                                  # CutPaste from the same image
        sy = int(rng.integers(0, max(1, h - ph)))
        sx = int(rng.integers(0, max(1, w - pw)))
        out[y0:y0 + ph, x0:x0 + pw] = img[sy:sy + ph, sx:sx + pw]
    elif kind == 2:                                  # rotate the region
        out[y0:y0 + ph, x0:x0 + pw] = cv2.rotate(
            img[y0:y0 + ph, x0:x0 + pw], int(rng.integers(0, 3)))
    else:                                            # erase
        out[y0:y0 + ph, x0:x0 + pw] = img.reshape(-1, 3).mean(axis=0)
    return out.astype(np.uint8), m


# --------------------------------------------------------------- helpers
def collect(model, img_path, kind, rng, donor_path=None):
    """Run the corrupted image through frozen DINOv2 and keep masked patches."""
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "texture":
        aug, m = texture_proxy(img, rng)
    else:
        donor = (cv2.cvtColor(cv2.imread(donor_path, cv2.IMREAD_COLOR),
                              cv2.COLOR_BGR2RGB) if donor_path else None)
        aug, m = structural_proxy(img, rng, donor)
    # ad2_pipeline.extract_patch_features() reads from a path; the backbone's
    # prepare_image() accepts an array, which is what we need for an in-memory
    # corrupted image. `aug` is already RGB.
    tensor, grid = model.prepare_image(aug)
    feats = model.extract_features(tensor)
    gh, gw = grid
    ph, pw = aug.shape[0] // gh, aug.shape[1] // gw
    mm = cv2.resize(m, (gw * pw, gh * ph), interpolation=cv2.INTER_AREA)
    cover = mm.reshape(gh, ph, gw, pw).mean(axis=(1, 3))
    keep = cover >= PATCH_MIN_COVER
    return feats.reshape(gh, gw, -1)[keep]


def build_image_proxy(model, obj, kind):
    d = os.path.join(AD2_ROOT, obj, "train", "good")
    files = list_png(d)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=min(N_AUG_IMGS, len(files)), replace=False)
    chunks = []
    for i in pick:
        other = files[rng.integers(0, len(files))] if len(files) > 1 else None
        donor_path = None if kind == "texture" else os.path.join(d, other)
        chunks.append(collect(model, os.path.join(d, files[i]), kind, rng,
                              donor_path))
    X = np.concatenate([c for c in chunks if len(c)], axis=0)
    print(f"  {obj}/{kind}: {len(X)} pseudo-defect patches from "
          f"{len(pick)} corrupted images")
    return X


# ------------------------------------------------------------- F3: feature
def feature_proxy(obj, rng, n_bank_for_cal=20000):
    """Off-manifold perturbations of normal patches.

    The normal subspace is fit on the BANK patches; the perturbations are
    applied to TAILQ patches so the two never share a source image.
    """
    tr, _ = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    bank = tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32)
    tailq = tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    b = norm(bank)
    mu = b.mean(axis=0)
    U, S, Vt = np.linalg.svd(b - mu, full_matrices=False)
    var = S ** 2
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), 0.90) + 1)
    V = Vt[:k]                                   # normal subspace basis (k, 384)

    # target magnitude: 1.5x the normal p99 distance of tailq patches
    cal_bank = b[rng.choice(len(b), size=min(n_bank_for_cal, len(b)), replace=False)]
    q = norm(tailq)
    d0 = 1.0 - (q[:2000] @ cal_bank.T).max(axis=1)
    target = F3_TARGET_MULT * float(np.percentile(d0, 99))

    # random unit directions in the orthogonal complement of the normal subspace
    sel = rng.choice(len(q), size=min(F3_N_PATCHES, len(q)), replace=False)
    z = q[sel]
    r = rng.normal(size=(len(z), 384)).astype(np.float32)
    r -= (r @ V.T) @ V                           # project out the normal subspace
    r /= (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)

    best, best_alpha, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        zp = norm(z + a * r)
        d = float((1.0 - (zp @ cal_bank.T).max(axis=1)).mean())
        if abs(d - target) < abs(best_d - target):
            best, best_alpha, best_d = a * r, a, d
    zp = norm(z + best)
    print(f"  {obj}/feature: normal subspace k={k}/384, normal_p99={target/F3_TARGET_MULT:.3f}, "
          f"target={target:.3f}, alpha={best_alpha} -> mean d={best_d:.3f}, "
          f"{len(zp)} patches")
    return zp.astype(np.float16)


def lowrank_proxy(obj, base="feature", k=5):
    """Rank-k reconstruction of an existing pseudo-defect set.

    A minimal intervention for testing the coherence hypothesis: the patches
    keep their own mean and their top-k directions, but the set is forced to
    span k dimensions instead of hundreds. Real defect sets need only
    k90 = 5..40 dimensions, so if dimensionality is what matters, this should
    recover some of the oracle's behaviour; if it does not, the missing
    ingredient is semantic rather than geometric.
    """
    X = np.load(os.path.join(PROXY_DIR, f"{obj}_{base}.npy")).astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    Xk = (Xc @ V.T) @ V + mu
    print(f"  {obj}/lowrank(k={k}) from '{base}': "
          f"{len(Xk)} patches, k90 forced to <= {k}, "
          f"mean|shift|={np.linalg.norm(Xk - X, axis=1).mean():.3f}")
    return Xk.astype(np.float16)


def defect_direction(source, obj_split_seed=SEED):
    """Mean defect direction in `source`'s feature space, unit-normalised.

    Defect direction := mean(defect patches) - mean(training normals). Using a
    direction rather than absolute donor patches is what makes the cross-object
    test meaningful: donor patches placed verbatim sit far outside the
    separation hinge (measured: 0% below it) so the constraint would be vacuous.
    A text prior would also supply a direction, not a set of points.
    """
    tr, te = load_cache(source)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(obj_split_seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    D = np.concatenate([te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                        for i in sup])
    N = tr["feats"].reshape(-1, tr["feats"].shape[-1]).astype(np.float32)
    N = N[np.random.default_rng(obj_split_seed).choice(
        len(N), size=min(20000, len(N)), replace=False)]
    v = D.mean(axis=0) - N.mean(axis=0)
    print(f"    direction from '{source}': n_defect={len(D)}, "
          f"n_normal={len(N)}, |v|={np.linalg.norm(v):.3f}")
    return v / (np.linalg.norm(v) + 1e-8)


def direction_proxy(obj, source, tag):
    """Apply a real defect direction from `source` to obj's own normal patches.

    Magnitude is calibrated the same way as F3 (0.6x the normal p99), so the
    only thing that differs from F3 is *which* direction is used: a real
    defect direction instead of a random off-manifold one.
    """
    v = defect_direction(source)
    tr, _ = load_cache(obj)
    _, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, SEED)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bank = norm(tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32))
    rng = np.random.default_rng(SEED)
    b = bank[rng.choice(len(bank), size=min(20000, len(bank)), replace=False)]
    q = norm(tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32))
    d0 = 1.0 - (q[rng.choice(len(q), size=min(2000, len(q)), replace=False)]
                @ b.T).max(axis=1)
    target = F3_TARGET_MULT * float(np.percentile(d0, 99))
    sel = rng.choice(len(q), size=min(F3_N_PATCHES, len(q)), replace=False)
    z = q[sel]
    best, best_a, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        zp = norm(z + a * v)
        d = float((1.0 - (zp @ b.T).max(axis=1)).mean())
        if abs(d - target) < abs(best_d - target):
            best, best_a, best_d = a * v, a, d
    print(f"    {obj}/{tag}: target d={target:.3f}, alpha={best_a} -> "
          f"mean d={best_d:.3f}, n={len(z)}")
    return norm(z + best).astype(np.float16)


def _real_defect_set(obj, seed=SEED):
    """The same supervision-half real defect patches config E uses."""
    tr, te = load_cache(obj)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    return np.concatenate([te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                           for i in sup])


def rank_scan_proxy(obj, k):
    """Real defect set compressed to rank k by PCA reconstruction.

    Config E keeps the whole set; this asks how many dimensions of it are
    actually needed. Rank-1 here is NOT the same test as G2: G2 applied a mean
    *direction* to normal patches (a line through the normal cloud), whereas
    this is the real defect cloud flattened onto its own top-k axes.
    """
    X = _real_defect_set(obj)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    Xk = (Xc @ Vt[:k].T) @ Vt[:k] + mu
    evr = float((S[:k] ** 2).sum() / (S ** 2).sum())
    print(f"    {obj}/rank{k}: n={len(Xk)}, explained var={evr:.3f}, "
          f"mean|shift|={np.linalg.norm(Xk - X, axis=1).mean():.3f}")
    return Xk.astype(np.float16)


def proto_scan_proxy(obj, K):
    """Real defect set replaced by K k-means prototypes.

    Each patch is replaced by its cluster centroid, so the set keeps its size
    (and therefore its weight in the loss) but carries only K distinct
    vectors. Separates "needs many dimensions" from "needs many modes".
    """
    from sklearn.cluster import KMeans
    X = _real_defect_set(obj)
    K = min(K, len(X))
    if K == 1:
        lab = np.zeros(len(X), dtype=int)
    else:
        lab = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit_predict(X)
    C = np.stack([X[lab == c].mean(axis=0) for c in range(K)])
    resid = float(np.linalg.norm(X - C[lab], axis=1).mean())
    print(f"    {obj}/proto{K}: n={len(X)}, {K} prototypes, "
          f"mean|shift|={resid:.3f}")
    return C[lab].astype(np.float16)


def pooled_subspace_proxy(obj, donors, k=20):
    """Target normals perturbed along a POOLED cross-object defect subspace.

    G1 transferred a single mean direction and failed. This transfers a k-dim
    subspace built from several donors' defect deviations, which is the
    "multiple visual defect modes" hypothesis. Magnitude is calibrated the same
    way as F3 so the hinge stays active (raw donor patches sit far outside it).
    """
    rows = []
    for d in donors:
        tr_d, te_d = load_cache(d)
        gt = te_d["gt_frac"].reshape(len(te_d["types"]), -1)
        bad = np.where(te_d["types"] == "bad")[0]
        perm = np.random.default_rng(SEED).permutation(len(bad))
        sup = bad[perm[:len(bad) // 2]]
        D = np.concatenate([te_d["feats"][i].astype(np.float32)[gt[i] > 0.10]
                            for i in sup])
        N = tr_d["feats"].reshape(-1, tr_d["feats"].shape[-1]).astype(np.float32)
        N = N[np.random.default_rng(SEED).choice(
            len(N), size=min(10000, len(N)), replace=False)]
        rows.append(D - N.mean(axis=0, keepdims=True))
    A = np.concatenate(rows)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:k]                                   # pooled defect directions
    print(f"    {obj}/pooled_k{k}: donors={donors}, "
          f"{len(A)} deviations -> {V.shape[0]} directions")

    tr, _ = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bank = norm(tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32))
    rng = np.random.default_rng(SEED)
    b = bank[rng.choice(len(bank), size=min(20000, len(bank)), replace=False)]
    q = norm(tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32))
    d0 = 1.0 - (q[rng.choice(len(q), size=min(2000, len(q)), replace=False)]
                @ b.T).max(axis=1)
    target = F3_TARGET_MULT * float(np.percentile(d0, 99))
    sel = rng.choice(len(q), size=min(F3_N_PATCHES, len(q)), replace=False)
    z = q[sel]
    # random mixtures of the pooled defect directions
    w = rng.normal(size=(len(z), V.shape[0])).astype(np.float32)
    v = norm(w @ V)
    best, best_a, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        zp = norm(z + a * v)
        d = float((1.0 - (zp @ b.T).max(axis=1)).mean())
        if abs(d - target) < abs(best_d - target):
            best, best_a, best_d = a * v, a, d
    print(f"      target d={target:.3f}, alpha={best_a} -> mean d={best_d:.3f}")
    return norm(z + best).astype(np.float16)


def residual_band_proxy(obj, lo=20, hi=45):
    """Gate 3: perturb normal patches along normal-PCA dims [lo, hi).

    This is the ONLY direction source whose overlap with the real defect
    subspace beat the sham null on part of the objects (wallplugs 0.166 vs
    0.114; can 0.091 vs 0.074; sheet_metal and vial lost). Gate 3 asks whether
    that marginal geometric signal has any causal value once it is fed to the
    same separation loss F1-F5 used.

    METHODOLOGICAL CAVEAT, keep it attached to any result: the band [20, 45)
    was selected by looking at real-defect overlap. So a success here licenses
    "this normal-derived subspace has intervention value", NOT "normal data
    alone can find it". Automatically choosing the band without defects is a
    separate, unsolved problem.
    """
    tr, _ = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    rng = np.random.default_rng(SEED)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bank = tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32)
    b = norm(bank[rng.choice(len(bank), size=min(20000, len(bank)),
                             replace=False)])
    Xn = bank - bank.mean(axis=0, keepdims=True)
    C = (Xn.T @ Xn) / (len(Xn) - 1)
    _, V = np.linalg.eigh(C)                      # ascending eigenvalues
    V = V[:, ::-1]                                # high variance first
    band = V[:, lo:hi]                            # (384, hi-lo)
    q = norm(tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32))
    cal = b[rng.choice(len(b), size=min(20000, len(b)), replace=False)]
    d0 = 1.0 - (q[rng.choice(len(q), size=min(2000, len(q)), replace=False)]
                @ cal.T).max(axis=1)
    target = F3_TARGET_MULT * float(np.percentile(d0, 99))
    sel = rng.choice(len(q), size=min(F3_N_PATCHES, len(q)), replace=False)
    z = q[sel]
    w = rng.normal(size=(len(z), band.shape[1])).astype(np.float32)
    v = norm(w @ band.T)
    best, best_a, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        zp = norm(z + a * v)
        d = float((1.0 - (zp @ cal.T).max(axis=1)).mean())
        if abs(d - target) < abs(best_d - target):
            best, best_a, best_d = a * v, a, d
    print(f"    {obj}/band[{lo}:{hi}]: {band.shape[1]} dirs, target d="
          f"{target:.3f}, alpha={best_a} -> mean d={best_d:.3f}, n={len(z)}")
    return norm(z + best).astype(np.float16)


def _normal_cov(obj, n_max=40000, seed=SEED, one_shot=False):
    """Target/source normal covariance (regularised), optionally 1-shot only."""
    tr, _ = load_cache(obj)
    if one_shot:
        # the actual few-shot constraint: a single reference image
        X = tr["feats"][0].astype(np.float32)
    else:
        bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, seed)
        X = tr["feats"][bank_idx].reshape(-1, tr["feats"].shape[-1]).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        X = X[rng.choice(len(X), size=n_max, replace=False)]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / max(len(Xc) - 1, 1)


def _sqrtm(C, inv=False, eps_rel=1e-2):
    """Regularised C^(+-1/2). Small eigenvalues are clamped, not inverted raw."""
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps_rel * max(w.mean(), 1e-12))
    p = -0.5 if inv else 0.5
    return (V * (w ** p)) @ V.T


def _source_residuals(src, seed=SEED):
    """Per-image (mean defect) - (mean clean) in the source's own space.

    This is the "object-relative anomaly residual": subtracting the same
    image's clean patches removes source scene, lighting and object identity,
    leaving what an anomaly looks like relative to that object's own normal.
    """
    tr, te = load_cache(src)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    out = []
    for i in sup:
        X = te["feats"][i].astype(np.float32)
        d, c = gt[i] > 0.10, gt[i] == 0.0
        if d.sum() >= 1 and c.sum() >= 1:
            out.append(X[d].mean(axis=0) - X[c].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, 384), np.float32)


def transport_proxy(obj, sources, mode, one_shot_target=False):
    """Gate 4: external anomaly residuals, transported into the target.

    X0 raw          source residuals used as-is (the known failure baseline)
    X4 whiten       each source whitened by ITS OWN normal covariance, so the
                    source's object identity is divided out
    X5 whiten+rec   then recoloured by the TARGET's normal covariance, i.e.
                    re-expressed in the target's coordinate system
    X6              as X5 but the target covariance comes from a single
                    reference image -- the honest few-shot constraint

    All four apply the result to the target's own normal patches with the same
    magnitude calibration as F3, so the only thing that varies is the direction
    source. No target defect patch is used at any point.
    """
    rs = []
    for s in sources:
        R = _source_residuals(s)
        if len(R) == 0:
            continue
        if mode != "raw":
            R = R @ _sqrtm(_normal_cov(s), inv=True)
        rs.append(R)
        print(f"      source '{s}': {len(R)} residuals")
    if not rs:
        return None
    R = np.concatenate(rs)
    if mode in ("whiten_rec", "whiten_rec_1shot"):
        R = R @ _sqrtm(_normal_cov(obj, one_shot=one_shot_target))

    tr, _ = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bank = tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32)
    rng = np.random.default_rng(SEED)
    b = norm(bank[rng.choice(len(bank), size=min(20000, len(bank)),
                             replace=False)])
    q = norm(tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32))
    cal = b[rng.choice(len(b), size=min(20000, len(b)), replace=False)]
    d0 = 1.0 - (q[rng.choice(len(q), size=min(2000, len(q)), replace=False)]
                @ cal.T).max(axis=1)
    target = F3_TARGET_MULT * float(np.percentile(d0, 99))
    sel = rng.choice(len(q), size=min(F3_N_PATCHES, len(q)), replace=False)
    z = q[sel]
    # only ~130 source residuals are available across all donors, so the
    # transported directions are resampled with replacement.
    # v MUST be unit-normalised: z is unit-norm, and raw source residuals have
    # norm ~30, so without this even alpha=0.05 swamps z entirely -- the alpha
    # search then cannot reach the calibration target and every pseudo-defect
    # lands far above the hinge (measured: 0-2% below it, i.e. a silent
    # degeneracy into config D).
    v = R[rng.choice(len(R), size=len(z), replace=True)]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    best, best_a, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        zp = norm(z + a * v)
        d = float((1.0 - (zp @ cal.T).max(axis=1)).mean())
        if abs(d - target) < abs(best_d - target):
            best, best_a, best_d = a * v, a, d
    print(f"    {obj}/{mode}: target d={target:.3f}, alpha={best_a} -> "
          f"mean d={best_d:.3f}, n={len(z)}")
    return norm(z + best).astype(np.float16)


def gate5a_proxy(obj, mode, k=40):
    """Gate 5A: is the defect rank-40 SUBSPACE sufficient, or do the
    coefficients inside it matter too?

    R40 (the positive control) keeps both the true subspace U40 and the true
    coefficients c_i = U40^T (z_i - mu). So its success does NOT license
    "predicting U_target is enough" -- that was an untested inference. This
    builder holds U40 fixed at the target's true defect basis and varies ONLY
    the coefficient matrix:

        H1  isotropic          c ~ N(0, I), total variance matched
        H2  covariance-matched c ~ N(0, Sigma_c) with Sigma_c the real
                               coefficient covariance -- same subspace, same
                               second-order shape, no real per-sample values
        H3  source coefficients c = R_source @ U40^T, i.e. external anomaly
                               residuals projected into the TARGET's true
                               basis. X0 failed without the correct basis;
                               this asks whether the basis was the whole story.

    All four use the same construction as R40 (absolute points mu + c @ U40),
    so the only thing that differs is where the coefficients come from.
    """
    D = _real_defect_set(obj)
    mu = D.mean(axis=0, keepdims=True)
    U = top_subspace_local(D, k)              # (k, 384)
    C_real = (D - mu) @ U.T                   # (n, k)
    n = len(D)
    rng = np.random.default_rng(SEED)
    if mode == "H1":
        scale = float(np.sqrt((C_real ** 2).sum(axis=1).mean()))
        C = rng.normal(size=(n, k)) * (scale / np.sqrt(k))
    elif mode == "H2":
        Sc = np.cov(C_real.T) + 1e-6 * np.eye(k)
        try:
            L = np.linalg.cholesky(Sc)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Sc)
            L = V * np.sqrt(np.maximum(w, 1e-8))
        C = rng.normal(size=(n, k)) @ L.T
    elif mode == "H0b":
        C = C_real
    elif mode == "H3":
        others = [o for o in OBJECTS if o != obj]
        R = np.concatenate([_source_residuals(s) for s in others])
        C = R @ U.T
        C = C[rng.choice(len(C), size=n, replace=True)]
    else:
        raise ValueError(mode)
    # Construction: z_normal + alpha * (U40 @ c), with alpha calibrated to the
    # same 0.6 x normal-p99 the other probes use.
    #
    # (z = mu + c @ U, the R40 construction, cannot be calibrated on can: its
    #  mean defect patch mu_D already sits 0.240 from the bank while R40's
    #  cloud averages 0.127, so shrinking about mu_D converges to 0.240 -- above
    #  the hinge threshold -- and no scale factor reaches the target. Perturbing
    #  normal patches is calibratable for every object because alpha -> 0 gives
    #  the normal distance, and it also makes these rows directly comparable to
    #  F3 / Q1 / X0, which share the construction.)
    tr0, _ = load_cache(obj)
    bi0, tq0 = split_train(tr0["feats"].shape[0], 0.8, SEED)
    _nb = norm_rows(tr0["feats"][bi0].reshape(-1, 384).astype(np.float32))
    _r = np.random.default_rng(SEED)
    _nb = _nb[_r.choice(len(_nb), size=min(20000, len(_nb)), replace=False)]
    _q = norm_rows(tr0["feats"][tq0].reshape(-1, 384).astype(np.float32))
    _q = _q[_r.choice(len(_q), size=min(2000, len(_q)), replace=False)]
    _tgt = F3_TARGET_MULT * float(
        np.percentile(1.0 - (_q @ _nb.T).max(axis=1), 99))
    sel = _r.choice(len(_q), size=min(F3_N_PATCHES, len(_q)), replace=False)
    z0 = _q[sel]
    dirs = C[_r.choice(len(C), size=len(z0), replace=True)] @ U
    dirs = norm_rows(dirs)
    best, best_a, best_d = None, None, np.inf
    for a in F3_ALPHAS:
        d = float((1.0 - (norm_rows(z0 + a * dirs) @ _nb.T).max(axis=1)).mean())
        if abs(d - _tgt) < abs(best_d - _tgt):
            best, best_a, best_d = a * dirs, a, d
    dev = float(np.linalg.norm(C, axis=1).mean())
    dev_real = float(np.linalg.norm(C_real, axis=1).mean())
    print(f"    {obj}/{mode}: k={U.shape[0]}, n={len(z0)}, "
          f"mean|coef|={dev:.2f} (real={dev_real:.2f}), "
          f"alpha={best_a} -> mean_d {best_d:.3f} (target {_tgt:.3f})")
    return norm_rows(z0 + best).astype(np.float16)


def _unused_absolute_construction(obj, mode, mu, C, C_real, U):
    Z = mu + C @ U
    Z_real = mu + C_real @ U
    # Amplitude calibration. Matching coefficient norms is NOT enough to make
    # the hinge engage: real defect coefficients land at mean 1-NN distance
    # ~0.134 from the bank while covariance-matched Gaussian coefficients with
    # the SAME norm land at ~0.249 -- so a rebuild would sit above the hinge,
    # carry zero separation gradient and silently degenerate into config D.
    # Rescaling the deviations so both sets sit at the same mean distance keeps
    # the comparison about coefficient SHAPE and nothing else.
    tr, _ = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, SEED)
    b = norm_rows(tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32))
    rng2 = np.random.default_rng(SEED)
    b = b[rng2.choice(len(b), size=min(20000, len(b)), replace=False)]

    def mean_nn_dist(X):
        return float((1.0 - (norm_rows(X) @ b.T).max(axis=1)).mean())

    # Iterative, not a single linear rescale: 1-NN cosine distance is a highly
    # nonlinear function of the deviation magnitude, so one shot at
    # target/current badly overshoots (measured on can: a 3.6x shrink moved the
    # distance only 0.462 -> 0.282, leaving the hinge dead).
    d_real = mean_nn_dist(Z_real)
    d_syn = mean_nn_dist(Z)
    for _ in range(12):
        if d_syn < 1e-9 or abs(d_syn - d_real) < 1e-3 * max(d_real, 1e-6):
            break
        Z = mu + (Z - mu) * (d_real / d_syn)
        d_syn = mean_nn_dist(Z)
    dev = float(np.linalg.norm(C, axis=1).mean())
    dev_real = float(np.linalg.norm(C_real, axis=1).mean())
    print(f"    {obj}/{mode}: k={U.shape[0]}, n={n}, mean|coef|={dev:.2f} "
          f"(real={dev_real:.2f}), mean_d calibrated {d_syn:.3f} "
          f"vs real {d_real:.3f}")
    return Z.astype(np.float16)


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def top_subspace_local(X, k):
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].T.copy()


def gate5c_proxy(obj, mode):
    """Gate 5C: factorise the defect cloud into anchor + internal structure.

        z_defect = mu_d + r        r = centred rank-40 residual (real coeffs)

    Gate 5A was confounded: R40 (mu_d + r) succeeded while H0b -- same basis,
    same real coefficients, but anchored on NORMAL patches -- failed, so the
    only thing that differed was the placement of the cloud. This builder moves
    one factor at a time.

      J1 / lam0.00   mu_normal + r          anchor swapped, internal shape kept
      lam0.25/0.50/0.75   mu(lam) + r       Euclidean centroid interpolation
      J5             mu_defect only         centre with matched tiny jitter
      J2             mu_defect + U40 x isotropic coeffs
      J3             mu_defect + U40 x covariance-matched coeffs
      (R40 = mu_defect + r is the existing lambda=1 positive control)

    Every variant keeps the pseudo-point COUNT equal to the real defect set, and
    everything is L2-normalised downstream by the detector exactly as normal
    features are, so no variant gets a norm advantage. The interpolation is
    Euclidean, and l2norms of the interpolated clouds are reported so a pure
    norm artefact can be ruled out.
    """
    D = _real_defect_set(obj)
    n = len(D)
    mu_d = D.mean(axis=0, keepdims=True)
    U = top_subspace_local(D, 40)
    C_real = (D - mu_d) @ U.T
    r = C_real @ U                                  # (n, 384) centred residual
    tr, _ = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, SEED)
    mu_n = tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32).mean(
        axis=0, keepdims=True)
    rng = np.random.default_rng(SEED)

    if mode == "J5":
        # centre alone; jitter is 1% of the real residual scale, only so the
        # set is not a single duplicated point
        Z = mu_d + 0.01 * float(np.linalg.norm(r, axis=1).mean()) * \
            rng.normal(size=(n, 384)).astype(np.float32)
    elif mode == "J2":
        scale = float(np.sqrt((C_real ** 2).sum(axis=1).mean()))
        C = rng.normal(size=(n, 40)).astype(np.float32) * (scale / np.sqrt(40))
        Z = mu_d + C @ U
    elif mode == "J3":
        Sc = np.cov(C_real.T) + 1e-6 * np.eye(40)
        try:
            L = np.linalg.cholesky(Sc)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Sc)
            L = V * np.sqrt(np.maximum(w, 1e-8))
        Z = mu_d + (rng.normal(size=(n, 40)).astype(np.float32) @ L.T) @ U
    elif mode.startswith("lam"):
        lam = float(mode[3:])
        mu = (1.0 - lam) * mu_n + lam * mu_d
        Z = mu + r
    else:
        raise ValueError(mode)
    print(f"    {obj}/{mode}: n={n}, |Z| mean={np.linalg.norm(Z,axis=1).mean():.2f} "
          f"(mu_d |.|={np.linalg.norm(mu_d):.2f}, mu_n |.|={np.linalg.norm(mu_n):.2f})")
    return Z.astype(np.float16)


def truth_variants(obj, donor="can"):
    """Oracle-variant defect sets, used to bound how transferable the signal is.

    These DO use real test_public defect patches, so they are oracles, not
    candidate methods. They answer the question that has to precede any
    investment in CLIP/VLM text priors:

      truth_cross  another object's real defects. A text description carries
                   strictly less than this, so if cross-object defects do not
                   transfer, no semantic prior will either.
      truth_mean   rank-1 summary of THIS object's own defects (the mean
                   direction, repeated). A text prior is also essentially one
                   direction, so this says whether a single direction could
                   suffice at all.

    Both use the same supervision/eval split as config E.
    """
    PRIMARY_GT = 0.10
    out = {}
    for tag, source in (("truth_cross", donor), ("truth_mean", obj)):
        tr, te = load_cache(source)
        gt = te["gt_frac"].reshape(len(te["types"]), -1)
        bad = np.where(te["types"] == "bad")[0]
        perm = np.random.default_rng(SEED).permutation(len(bad))
        sup = bad[perm[:len(bad) // 2]]
        X = np.concatenate([te["feats"][i].astype(np.float32)[gt[i] > PRIMARY_GT]
                            for i in sup])
        if tag == "truth_mean":
            X = np.repeat(X.mean(axis=0, keepdims=True), len(X), axis=0)
        out[tag] = X.astype(np.float16)
        print(f"  {obj}/{tag}: n={len(X)} from '{source}' "
              f"({'rank-1 mean' if tag == 'truth_mean' else 'all patches'})")
    return out


def report_distances(obj):
    """How far are the pseudo-defects from the bank, versus the normal p99?

    If a proxy's patches already sit above normal_p99 + margin (0.05), the
    separation hinge is satisfied at step 0 and carries no gradient -- that
    config would silently reduce to config D. This is the check that catches it
    before wasting a run.
    """
    tr, _ = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, SEED)
    rng = np.random.default_rng(SEED)
    norm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    bank = tr["feats"][bank_idx].reshape(-1, 384).astype(np.float32)
    sub = bank[rng.choice(len(bank), size=min(DIAG_BANK, len(bank)),
                          replace=False)]
    b = norm(sub)
    q = norm(tr["feats"][tailq_idx].reshape(-1, 384).astype(np.float32))
    q = q[rng.choice(len(q), size=min(4000, len(q)), replace=False)]
    d_norm = 1.0 - (q @ b.T).max(axis=1)
    p99 = float(np.percentile(d_norm, 99))
    thr = p99 + 0.05
    print(f"  --- {obj}: normal p99={p99:.3f}, hinge threshold={thr:.3f} ---")
    for kind in ("texture", "structural", "feature", "lowrank5", "lowrank20",
                 "crossdir", "selfdir", "rank1", "rank5", "rank20",
                 "proto1", "proto16", "pooled", "band2045",
                 "X0raw", "X4white", "X5recol", "X6recol1",
                 "H1iso", "H2cov", "H3src",
                 "J1", "J5", "J2", "J3", "rank40"):
        p = os.path.join(PROXY_DIR, f"{obj}_{kind}.npy")
        if not os.path.exists(p):
            continue
        X = np.load(p).astype(np.float32)
        X = X[rng.choice(len(X), size=min(4000, len(X)), replace=False)]
        d = 1.0 - (norm(X) @ b.T).max(axis=1)
        frac_active = float((d < thr).mean())
        flag = "" if frac_active > 0.2 else "   <-- WARNING: hinge mostly inactive"
        print(f"      {kind:<11} mean_d={d.mean():.3f} median={np.median(d):.3f} "
              f"| below hinge: {100*frac_active:.0f}%{flag}")


def main():
    objects = sys.argv[1:] or OBJECTS
    os.makedirs(PROXY_DIR, exist_ok=True)
    need_image = [o for o in objects
                  if not all(os.path.exists(os.path.join(PROXY_DIR, f"{o}_{k}.npy"))
                             for k in ("texture", "structural"))]
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448) \
        if need_image else None
    for obj in objects:
        print(f"=== {obj} ===", flush=True)
        for kind in ("texture", "structural"):
            out = os.path.join(PROXY_DIR, f"{obj}_{kind}.npy")
            if os.path.exists(out):
                print(f"  {obj}/{kind}: cached, skip")
                continue
            np.save(out, build_image_proxy(model, obj, kind))
        out = os.path.join(PROXY_DIR, f"{obj}_feature.npy")
        if os.path.exists(out):
            print(f"  {obj}/feature: cached, skip")
        else:
            np.save(out, feature_proxy(obj, np.random.default_rng(SEED)))
        # rank-k variants, derived from the cached off-manifold set so the ONLY
        # thing that changes between F3 and F4 is the dimensionality
        for k in LR_K:
            o = os.path.join(PROXY_DIR, f"{obj}_lowrank{k}.npy")
            if os.path.exists(o):
                print(f"  {obj}/lowrank{k}: cached, skip")
            else:
                np.save(o, lowrank_proxy(obj, "feature", k))
        donor = "can" if obj != "can" else "wallplugs"
        for tag, src in (("crossdir", donor), ("selfdir", obj)):
            o = os.path.join(PROXY_DIR, f"{obj}_{tag}.npy")
            if not os.path.exists(o):
                np.save(o, direction_proxy(obj, src, tag))
        # ---- Phase 3C probes 1 & 2: how much of the real defect set is needed
        for k in RANK_K:
            o = os.path.join(PROXY_DIR, f"{obj}_rank{k}.npy")
            if not os.path.exists(o):
                np.save(o, rank_scan_proxy(obj, k))
        for K in PROTO_K:
            o = os.path.join(PROXY_DIR, f"{obj}_proto{K}.npy")
            if not os.path.exists(o):
                np.save(o, proto_scan_proxy(obj, K))
        # ---- Gate 5C: anchor vs internal structure of the defect cloud
        for tag, mode in (("J1", "lam0.00"), ("Jl25", "lam0.25"),
                          ("Jl50", "lam0.50"), ("Jl75", "lam0.75"),
                          ("J5", "J5"), ("J2", "J2"), ("J3", "J3")):
            o = os.path.join(PROXY_DIR, f"{obj}_{tag}.npy")
            if os.path.exists(o):
                print(f"  {obj}/{tag}: cached, skip")
                continue
            np.save(o, gate5c_proxy(obj, mode))
        # ---- Gate 5A: is the rank-40 SUBSPACE sufficient, or do the
        # coefficients inside it matter? R40 keeps both, so it never tested this.
        for tag, mode in (("H0b", "H0b"), ("H1iso", "H1"),
                              ("H2cov", "H2"), ("H3src", "H3")):
            o = os.path.join(PROXY_DIR, f"{obj}_{tag}.npy")
            if os.path.exists(o):
                print(f"  {obj}/{tag}: cached, skip")
                continue
            np.save(o, gate5a_proxy(obj, mode))
        # ---- Gate 4: external anomaly residual transport (leave-one-out)
        others4 = [o for o in OBJECTS if o != obj]
        for tag, mode, osh in (("X0raw", "raw", False),
                               ("X4white", "whiten", False),
                               ("X5recol", "whiten_rec", False),
                               ("X6recol1", "whiten_rec_1shot", True)):
            o = os.path.join(PROXY_DIR, f"{obj}_{tag}.npy")
            if os.path.exists(o):
                print(f"  {obj}/{tag}: cached, skip")
                continue
            print(f"  {obj}/{tag} (sources={others4}):")
            r = transport_proxy(obj, others4, mode, osh)
            if r is not None:
                np.save(o, r)
        # ---- Gate 3: normal residual band [20, 45)
        o = os.path.join(PROXY_DIR, f"{obj}_band2045.npy")
        if not os.path.exists(o):
            np.save(o, residual_band_proxy(obj, 20, 45))
        # ---- probe 3: pooled cross-object defect subspace
        others = [o for o in OBJECTS if o != obj]
        o = os.path.join(PROXY_DIR, f"{obj}_pooled.npy")
        if not os.path.exists(o):
            np.save(o, pooled_subspace_proxy(obj, others, k=20))
        report_distances(obj)


if __name__ == "__main__":
    main()
