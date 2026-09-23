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
    for kind in ("texture", "structural", "feature", "lowrank5", "lowrank20"):
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
        report_distances(obj)


if __name__ == "__main__":
    main()
