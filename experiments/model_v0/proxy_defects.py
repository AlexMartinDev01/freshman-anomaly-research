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
                 "proto1", "proto16", "pooled", "band2045"):
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
