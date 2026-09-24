# -*- coding: utf-8 -*-
"""
Phase 3C identifiability gates.

Before asking "which normal-derived subspace best approximates the defect
subspace", two questions have to be settled first. Both are pure geometry on
cached features -- no training, no loss, no adapter.

GATE 1 -- is the target even well defined?
    Split the real defect patches in half at random, build the rank-k subspace
    of each half, and measure how well they agree. Controls:
        same object, different splits   (should be high if the target is stable)
        different objects               (should be low if it is object-specific)
        random rank-k subspaces         (the floor)
    If same-object agreement is not clearly above both controls, then there is
    no stable "defect subspace" to estimate and the whole direction is void.
    Note this is prior to any normal-data question: an unstable target cannot be
    predicted by anything.

GATE 2 -- does normal geometry point at it at all?
    PCA the target's normal (bank) features, then ask where the defect
    subspace's energy sits in that eigenbasis. The natural readout is
    ENRICHMENT: (share of defect energy in a normal-PCA window) / (window's
    share of dimensions). 1.0 means the defect subspace is indifferent to
    normal geometry -- i.e. it looks like a random subspace with respect to
    normal covariance, and no amount of normal-only spectrum work will find it.
    A stable, large enrichment in the LOW-variance tail would be the first real
    evidence that normal data constrains the defect direction.

Overlap metric: mean squared principal cosine, trace(P_U P_V)/k.
    1.0 = identical subspaces, k/d = what two random rank-k subspaces give.

Usage: python experiments/model_v0/subspace_gates.py [obj ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import load_cache, split_train  # noqa: E402

OBJECTS = ["wallplugs", "sheet_metal", "vial", "can"]
RANK = 40
N_SPLITS = 10
SEED = 0


# ------------------------------------------------------------------ helpers
def defect_patches(obj, seed=SEED):
    """All real defect patches from the supervision half (same as config E)."""
    tr, te = load_cache(obj)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    return np.concatenate([te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                           for i in sup])


def defect_deltas(obj, seed=SEED):
    """Within-image (mean defect patch) - (mean clean patch), per bad image.

    The raw-defect-patch subspace is dominated by *which image* a patch came
    from -- scene and lighting variation, i.e. exactly the normal high-variance
    directions. Differencing against the clean patches of the SAME image
    cancels that, leaving the direction that actually separates defect from
    normal. Without this control, Gate 2 measures scene variation and calls it
    the defect subspace.
    """
    tr, te = load_cache(obj)
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


def sham_deltas(obj, seed=SEED):
    """Same construction as defect_deltas but with the defect mask displaced.

    DINOv2 features are strongly anisotropic, so ANY perturbation projects
    heavily onto the top principal directions. If a sham region -- same
    construction, no defect -- shows the same Gate-2 enrichment as the real
    defect region, then that enrichment is a property of the feature geometry
    rather than evidence that normal covariance points at the defect
    direction. This control decides whether Gate 2 says anything at all.
    """
    tr, te = load_cache(obj)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    rng = np.random.default_rng(seed + 77)
    out = []
    for i in sup:
        X = te["feats"][i].astype(np.float32)
        d, c = gt[i] > 0.10, gt[i] == 0.0
        n = int(d.sum())
        if n < 1 or c.sum() < n + 1:
            continue
        idx = np.where(c)[0]
        pick = rng.choice(len(idx), size=n, replace=False)
        out.append(X[idx[pick]].mean(axis=0) - X[idx].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, 384), np.float32)


def normal_patches(obj, cap=60000, seed=SEED):
    """Bank-split normal patches (target's own train/good)."""
    tr, _ = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, seed)
    X = tr["feats"][bank_idx].reshape(-1, tr["feats"].shape[-1]).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(X) > cap:
        X = X[rng.choice(len(X), size=cap, replace=False)]
    return X


def top_subspace(X, k):
    """Orthonormal (k, d) basis of the top-k principal directions of X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economical: eigendecompose the d x d covariance instead of SVD of X
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].T.copy()


def random_subspace(d, k, rng):
    return np.linalg.qr(rng.normal(size=(d, k)))[0].T.copy()


def overlap(U, V):
    """Mean squared principal cosine between two (k, d) orthonormal subspaces."""
    s = np.linalg.svd(U @ V.T, compute_uv=False)
    return float((s ** 2).mean())


# ------------------------------------------------------------------- gate 1
def gate1(objects, getter=None):
    print("=" * 96)
    print("GATE 1 -- is the rank-{} defect subspace a stable target?".format(RANK))
    print("=" * 96)
    rng = np.random.default_rng(SEED)
    d = None
    sub = {}
    print(f"\n{'object':<12}{'n_patch':>9}{'k':>5}{'same-obj (mean+-sd)':>24}"
          f"{'random floor':>14}")
    for o in objects:
        X = (getter or defect_patches)(o)
        d = X.shape[1]
        k = min(RANK, len(X) // 2 - 1, d)
        # same-object, different random halves
        vals = []
        for i in range(N_SPLITS):
            p = np.random.default_rng(1000 + i).permutation(len(X))
            A, B = X[p[:len(X) // 2]], X[p[len(X) // 2:]]
            try:
                vals.append(overlap(top_subspace(A, k), top_subspace(B, k)))
            except np.linalg.LinAlgError:
                pass
        sub[o] = (top_subspace(X, k), k)
        floor = k / d
        print(f"{o:<12}{len(X):>9}{k:>5}{np.mean(vals):>16.3f} +- "
              f"{np.std(vals):<5.3f}{floor:>14.3f}")
    print(f"\n  random floor = k/d (what two unrelated rank-k subspaces score)")

    print(f"\n{'pair (cross-object)':<28}{'overlap':>10}{'random floor':>14}")
    objs = list(sub)
    cross = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            k = min(sub[a][1], sub[b][1])
            ov = overlap(sub[a][0][:k], sub[b][0][:k])
            cross.append(ov)
            print(f"{a + ' vs ' + b:<28}{ov:>10.3f}{k / d:>14.3f}")
    print(f"\n  mean cross-object overlap = {np.mean(cross):.3f}")

    print("\n  VERDICT: same-object must clearly exceed BOTH the random floor "
          "and cross-object.")
    print("           If it does not, there is no stable target to estimate.")
    return sub


# ------------------------------------------------------------------- gate 2
def gate2(objects, sub, getter=None):
    print("\n" + "=" * 96)
    print("GATE 2 -- where does the defect subspace sit in the NORMAL eigenspectrum?")
    print("=" * 96)
    print("\n  enrichment = (defect energy share in a normal-PCA window) / "
          "(window's share of dims)")
    print("  1.0 = indifferent to normal geometry (a random subspace would "
          "score ~1.0)")
    windows = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 384)]
    hdr = "".join(f"{f'PCA[{a}:{b}]':>13}" for a, b in windows)
    print(f"\n{'object':<12}{'k':>5}{hdr}{'  |  max enr':>12}")
    for o in objects:
        if o not in sub:
            continue
        U, k = sub[o]
        Xn = normal_patches(o)
        d = Xn.shape[1]
        C = ((Xn - Xn.mean(0, keepdims=True)).T @ (Xn - Xn.mean(0, keepdims=True))) \
            / (len(Xn) - 1)
        w, V = np.linalg.eigh(C)
        V = V[:, ::-1]                      # eigenvectors, high variance first
        # energy of each normal eigenvector inside the defect subspace
        e = np.einsum("ij,jk->i", V.T, U.T)  # (d,) projection magnitudes
        e = e ** 2
        e = e / (e.sum() + 1e-12)
        enr = []
        for a, b in windows:
            b = min(b, d)
            share_dims = (b - a) / d
            enr.append(e[a:b].sum() / share_dims if share_dims > 0 else np.nan)
        print(f"{o:<12}{k:>5}" + "".join(f"{v:>13.2f}" for v in enr)
              + f"{max(enr):>12.2f}")
    print("\n  Control: a RANDOM rank-k subspace scores ~1.0 in every window.")
    print("  Enrichment concentrated in the low-variance tail (rightmost "
          "windows) would")
    print("  mean normal geometry does constrain the defect direction.")
    print("  Flat ~1.0 everywhere means normal covariance carries NO signal "
          "about it.")


def gate2b(objects, k=25):
    """Corrected Gate 2: candidate normal-only subspaces against a SHAM null.

    The first version asked whether the defect subspace's energy sits in
    particular normal-PCA windows. It is invalid: DINOv2 features are so
    anisotropic that a random clean region scores *higher* enrichment than the
    real defect (13.15 vs 6.50 on wallplugs). The null must therefore be
    anisotropy-matched.

    Here each normal-only candidate subspace N is scored by
        overlap(N, D_real)  vs  overlap(N, D_sham)
    where D_sham is built by the identical construction with the defect mask
    replaced by a random clean region. A candidate that beats the sham null is
    picking up defect structure; one that does not is just picking up the
    feature geometry every perturbation shares.
    """
    print("\n" + "=" * 96)
    print("GATE 2 (corrected) -- normal-only candidates vs an anisotropy-matched "
          "sham null")
    print("=" * 96)
    print(f"\n  rank k={k}; random-subspace floor would be k/384 = {k/384:.3f}")
    print(f"{'object':<12}{'candidate':<22}{'overlap(D_real)':>17}"
          f"{'overlap(D_sham)':>18}{'random':>9}{'beats sham?':>13}")
    for o in objects:
        D = defect_deltas(o)
        S = sham_deltas(o)
        kk = min(k, (len(D) - 1) // 2, (len(S) - 1) // 2)
        Ud, Us = top_subspace(D, kk), top_subspace(S, kk)
        Xn = normal_patches(o)
        Dn = Xn - Xn.mean(0, keepdims=True)
        C = (Dn.T @ Dn) / (len(Dn) - 1)
        w, V = np.linalg.eigh(C)
        V = V[:, ::-1]
        # high 1NN-distance normal patches (the "tail") for the tail candidate
        rng = np.random.default_rng(SEED)
        cen = Xn.mean(0, keepdims=True)
        sub = Xn[rng.choice(len(Xn), size=min(20000, len(Xn)), replace=False)]
        d1 = 1.0 - ((Xn - cen) / (np.linalg.norm(Xn - cen, axis=1, keepdims=True)
                                  + 1e-8)) @ \
            ((sub - cen) / (np.linalg.norm(sub - cen, axis=1, keepdims=True)
                            + 1e-8)).T
        tail = Xn[np.argsort(d1.max(axis=1))[-max(kk * 20, 500):]]
        cands = {
            "normal top-k": V[:, :kk].T,
            "normal bottom-k": V[:, -kk:].T,
            "normal residual k (20:)": V[:, 20:20 + kk].T,
            "normal tail patches": top_subspace(tail, kk),
        }
        for name, N in cands.items():
            orl, osh = overlap(N, Ud), overlap(N, Us)
            print(f"{o:<12}{name:<22}{orl:>17.3f}{osh:>18.3f}"
                  f"{kk/384:>9.3f}{('YES' if orl > osh else 'no'):>13}")
    print("\n  A candidate worth pursuing must beat the sham on most objects. "
          "Beating only")
    print("  the random floor is not enough -- the sham already does that.")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    objects = argv or OBJECTS
    mode = ("delta" if "--delta" in sys.argv else
            "sham" if "--sham" in sys.argv else "raw")
    if mode == "delta":
        print("MODE: within-image defect-minus-clean deltas "
              "(scene/lighting cancelled)")
        getter = defect_deltas
    elif mode == "sham":
        print("MODE: SHAM -- same construction, random clean region instead of "
              "the defect")
        getter = sham_deltas
    else:
        print("MODE: raw defect patches (subspace may reflect scene/lighting)")
        getter = defect_patches
    if mode == "raw":
        sub = gate1(objects, getter)
        gate2(objects, sub, getter)
    else:
        gate2b(objects)


if __name__ == "__main__":
    main()
