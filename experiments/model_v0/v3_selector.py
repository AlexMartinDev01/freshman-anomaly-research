# -*- coding: utf-8 -*-
"""
V3 Stage 2 -- support-only calibration and the label-free trigger.

Implements docs/MODEL_V3_PREREGISTRATION.md sections 2.2.2 - 2.2.4 exactly as
frozen.  Nothing here is tunable; q, tau_hot, tau_gap and the window rule are
module constants.

    d_cal(p) = min over q in B_k, q not in N_3x3(p) of d(p, q)

`N_3x3(p)` is the 3x3 spatial neighbourhood of p *within the same support
image*: neighbouring patches are highly correlated, and letting a patch match
its own neighbours would drive the pseudo-normal score to ~0 and destroy the
reference distribution.  Matching non-local patches of the same image stays
allowed, which is what makes k = 1 work.

The calibration touches ONLY the k images this (object, shot, split) drew.
That is the point of the section: with all-train-images calibration a 1-shot
run would really be "1-shot bank + many-shot normal calibration".

Both the calibration map and the test map M0 use the SAME agg_all3 recipe --
per-layer 1-NN distance to the bank, per-layer z-score, then average -- with
each side's own pooled statistics.  Anything else would read tau_hot off a
different quantity than the one it is compared against.  Making the test map
from raw features instead of bank distances is a silent, total break of that
comparison, so it is asserted against in the smoke test.

Usage: python experiments/model_v0/v3_selector.py --objects bottle,screw --shot 1
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, VISA, load_cached  # noqa: E402

LAYERS = ["mid", "midlate", "final"]
WIN, Q_CAND, Q_HOT, Q_GAP = 3, 1.0 / 3.0, 0.99, 1.0 / 3.0
MVTEC_TR = os.path.join(RESULTS, "cache_ml_img")
MVTEC_TE = os.path.join(RESULTS, "cache_ml")
VISA_C = os.path.join(RESULTS, "cache_visa")


def _ensure_grids(d):
    """Some caches store no `grids` member.  Infer a square grid, but only if
    every image really has the same patch count -- otherwise a single inferred
    grid would silently mislabel the grid positions the trigger depends on."""
    if "grids" in d:
        return d
    offs = d["offsets"]
    n = int(offs[1] - offs[0])
    s = int(round(np.sqrt(n)))
    if s * s != n:
        raise RuntimeError(f"cannot infer a square grid from {n} patches")
    if any(int(offs[i + 1] - offs[i]) != n for i in range(len(offs) - 1)):
        raise RuntimeError("patch count varies across images; one inferred "
                           "grid is invalid -- store the real grids instead")
    d["grids"] = np.tile(np.array([s, s], dtype=np.int32), (len(offs) - 1, 1))
    return d


def load_split(obj, which):
    vis = not os.path.isdir(os.path.join(V1, obj))
    src = (VISA_C if vis else (MVTEC_TR if which == "train" else MVTEC_TE))
    return vis, {l: _ensure_grids(load_cached(src, l, f"{obj}_{which}"))
                 for l in LAYERS}


def window_stats(S):
    """hot(W), gap(W) for every fully-inside 3x3 window, vectorised.

    Frozen rule (2.2.3): cand = top 1/3 of the window by score (3 of 9),
    ctx = the rest, gap = median(cand) - median(ctx), hot = max(W).
    Window (r, c) of the result corresponds to S[r:r+3, c:c+3].
    """
    gh, gw = S.shape
    if gh < WIN or gw < WIN:
        return None, None
    W = sliding_window_view(S, (WIN, WIN)).reshape(gh - WIN + 1, gw - WIN + 1, WIN * WIN)
    srt = np.sort(W, axis=-1)
    k = int(round(Q_CAND * WIN * WIN))               # 3 of 9
    hot = W.max(axis=-1)
    gap = np.median(srt[:, :, -k:], axis=-1) - np.median(srt[:, :, :-k], axis=-1)
    return hot, gap


def _zscore_then_mean(dists):
    """agg_all3: per-layer z-score on the pooled values, then average."""
    Z = []
    for l in LAYERS:
        a = np.concatenate([m.ravel() for m in dists[l]])
        mu, sd = a.mean(), a.std() + 1e-12
        Z.append([(m - mu) / sd for m in dists[l]])
    return [sum(zs) / len(LAYERS) for zs in zip(*Z)]


def _bank(tr, idx, layer):
    offs = tr[layer]["offsets"]
    return l2n(torch.from_numpy(np.concatenate(
        [tr[layer]["feats"][offs[i]:offs[i + 1]] for i in idx]
    ).astype(np.float32)).to(DEV))


def _dist_maps(te, layer, bank, subset=None):
    offs = te[layer]["offsets"]
    n = len(te[layer]["names"]) if subset is None else subset
    out = []
    for i in range(n):
        q = torch.from_numpy(
            te[layer]["feats"][offs[i]:offs[i + 1]].astype(np.float32)).to(DEV)
        with torch.no_grad():
            out.append(nn_dist(l2n(q), bank).cpu().numpy())
    return out


def d_cal_maps(tr, idx):
    """Pseudo-normal agg_all3 maps, one per support image, using ONLY those k
    images as the bank.  The exclusion is applied per layer before z-scoring."""
    any_l = LAYERS[0]
    grids = tr[any_l]["grids"]
    gset = {tuple(int(v) for v in grids[i]) for i in idx}
    assert len(gset) == 1, f"support images have differing grids: {gset}"
    grid = gset.pop()
    gh, gw = grid

    offs = [0]
    for i in idx:
        offs.append(offs[-1] + gh * gw)
    offs = np.asarray(offs, dtype=np.int64)

    # row p of image m: which bank columns must be masked out
    rows = np.arange(gh * gw)
    rr, cc = rows // gw, rows % gw
    per_layer = {}
    for l in LAYERS:
        feats = np.concatenate([tr[l]["feats"][tr[l]["offsets"][i]:
                                                tr[l]["offsets"][i + 1]]
                                for i in idx])
        B = l2n(torch.from_numpy(feats.astype(np.float32)).to(DEV))
        maps = []
        for m in range(len(idx)):
            base = m * gh * gw
            Q = l2n(torch.from_numpy(feats[base:base + gh * gw].astype(np.float32)).to(DEV))
            D = (1.0 - Q @ B.T).cpu().numpy()
            allowed = np.ones_like(D, dtype=bool)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r2, c2 = rr + dr, cc + dc
                    ok = (r2 >= 0) & (r2 < gh) & (c2 >= 0) & (c2 < gw)
                    allowed[np.where(ok)[0], base + (r2[ok] * gw + c2[ok])] = False
            if not allowed.any(axis=1).all():
                # tiny-grid guard from the pre-registration: fall back to
                # excluding only the patch itself
                allowed[:] = True
                allowed[rows, base + rows] = False
            maps.append(np.where(allowed, D, np.inf).min(axis=1)
                        .reshape(gh, gw).astype(np.float32))
        per_layer[l] = maps
    return _zscore_then_mean(per_layer), grid


def calibrate(obj, shot, split, tr, verbose=False):
    """tau_hot / tau_gap from the k support images ONLY (section 2.2.2)."""
    idx = draw_images(tr[LAYERS[0]]["offsets"], shot, split, obj)
    maps, grid = d_cal_maps(tr, idx)
    hot, gap = [], []
    for m in maps:
        h, g = window_stats(m)
        if h is not None:
            hot.append(h.ravel())
            gap.append(g.ravel())
    hot, gap = np.concatenate(hot), np.concatenate(gap)
    tau_hot = float(np.quantile(hot, Q_HOT))
    m = hot >= tau_hot
    tau_gap = float(np.quantile(gap[m], Q_GAP))
    if verbose:
        print(f"      calib: {len(idx)} support img, grid {grid}, "
              f"{hot.size} windows, {int(m.sum())} nominally hot")
    return tau_hot, tau_gap, grid


def test_maps(obj, tr, te, shot, split, subset=None):
    """M0 for every test image -- identical construction to
    phase_m1_rescue.run(), so the trigger is applied to the project's map."""
    idx = draw_images(tr[LAYERS[0]]["offsets"], shot, split, obj)
    dists = {l: _dist_maps(te, l, _bank(tr, idx, l), subset) for l in LAYERS}
    return _zscore_then_mean(dists)


BUDGET_FRAC, N_MIN, N_MAX = 0.5, 5, 33


def budget(grid672):
    """B = half of the full-672 token count for this image (addendum 1, A1.2)."""
    return BUDGET_FRAC * grid672[0] * grid672[1]


def allocate(m, B):
    """Frozen crop-budget rule (addendum 1, A1.2).

    Truncate to K_max = floor(B/25) FIRST: without it the budget is not bounded
    once m > floor(B/25) -- at B=1152, m=60, even a 5x5 crop overspends
    (60*25 = 1500 > 1152), so "G2 holds by construction" would be false.

    Then take the largest ODD side n in [5, 33] with m'*n^2 <= B.
    """
    if B < N_MIN * N_MIN or m == 0:
        return 0, 0
    K = int(B // (N_MIN * N_MIN))
    m2 = min(m, K)
    n = 0
    for q in range(N_MAX, N_MIN - 1, -1):
        if q % 2 == 1 and m2 * q * q <= B:
            n = q
            break
    assert n >= N_MIN, f"no admissible crop side for m'={m2}, B={B}"
    assert m2 * n * n <= B, f"budget violated: {m2}*{n}^2 > {B}"
    return m2, n


def rank_truncate(hot, gap, rows, cols, K):
    """Keep K windows: hot desc, then gap asc, then (row, col) lexicographic."""
    return np.lexsort((cols, rows, gap, -hot))[:K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-table", action="store_true",
                    help="verify the frozen crop-budget table (addendum 1 A1.2)")
    ap.add_argument("--objects", default="")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.budget_table:
        B = 1152                     # 900x900 MVTec: 672 grid 48x48 = 2304
        print(f"  B = {B},  K_max = floor(B/25) = {int(B // 25)}")
        print(f"  {'m':>4} {'m_prime':>8} {'n':>3} {'m_prime*n^2':>12} {'<=B':>5}")
        for m in (1, 2, 4, 6, 10, 20, 46, 47, 60, 100):
            m2, n = allocate(m, B)
            print(f"  {m:>4} {m2:>8} {n:>3} {m2 * n * n:>12} "
                  f"{str(m2 * n * n <= B):>5}")
        bad = [m for m in range(0, 500) if (lambda t: t[0] * t[1] ** 2 > B)(
            allocate(m, B))]
        print(f"  exhaustive m in [0, 500): budget violations = {len(bad)}")
        assert not bad, f"frozen invariant violated at m={bad[:10]}"
        # and across the image sizes actually in play
        for g in ((48, 48), (48, 62)):          # MVTec 900x900; VisA pcb2
            Bg = budget(g)
            bad = [m for m in range(0, 800) if (lambda t: t[0] * t[1] ** 2 > Bg)(
                allocate(m, Bg))]
            print(f"  grid {g}: B={Bg:.0f}, K_max={int(Bg // 25)}, "
                  f"violations = {len(bad)}")
            assert not bad
        return

    for obj in a.objects.split(","):
        _, tr = load_split(obj, "train")
        _, te = load_split(obj, "test")
        names = [str(x) for x in te[LAYERS[0]]["names"]]
        types = [str(x) for x in te[LAYERS[0]]["types"]]
        for sp in range(a.splits):
            t0 = time.time()
            tau_hot, tau_gap, grid = calibrate(obj, a.shot, sp, tr,
                                               verbose=True)
            S = test_maps(obj, tr, te, a.shot, sp,
                          subset=(a.limit or None))
            n_fire = n_wtot = n_ph = n_pg = 0
            by = {}
            teg = te[LAYERS[0]]["grids"]
            for i, m in enumerate(S):
                # test_maps returns flat vectors; the window rule needs the grid
                m = m.reshape(int(teg[i][0]), int(teg[i][1]))
                h, g = window_stats(m)
                f = (h >= tau_hot) & (g <= tau_gap)
                ty = types[i]
                by.setdefault(ty, [0, 0])
                by[ty][0] += int(f.sum())
                by[ty][1] += int(f.size)
                n_fire += int(f.sum())
                n_wtot += int(f.size)
                n_ph += int((h >= tau_hot).sum())
                n_pg += int((g <= tau_gap).sum())
            print(f"  {obj:<12} {a.shot}-shot s{sp}  tau_hot {tau_hot:+.4f} "
                  f"tau_gap {tau_gap:+.4f}  grid {grid}  TEST trigger "
                  f"{n_fire}/{n_wtot} = {100.0 * n_fire / max(n_wtot, 1):.3f}% "
                  f"({time.time() - t0:.0f}s)", flush=True)
            for ty, (f, nw) in sorted(by.items()):
                print(f"      {ty:<6} {f}/{nw} = {100.0 * f / max(nw, 1):.3f}%",
                      flush=True)
            # Marginal rates: which condition is doing the suppressing?
            # The nominal 0.33% = 1% x 1/3 assumes the calibration
            # distribution transfers to test images.  If `hot` alone already
            # fires far below 1%, tau_hot sits too high for real normal data
            # -- the 3x3 exclusion inflates d_cal relative to a genuine
            # bank lookup, which section 2.2.4 flags as a known mismatch.
            print(f"      marginal  P(hot>=tau_hot) = "
                  f"{100.0 * n_ph / max(n_wtot, 1):.4f}%   "
                  f"P(gap<=tau_gap) = {100.0 * n_pg / max(n_wtot, 1):.4f}%",
                  flush=True)


if __name__ == "__main__":
    main()
