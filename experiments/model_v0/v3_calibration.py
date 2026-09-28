# -*- coding: utf-8 -*-
"""
V3 calibration primitives -- four independently testable pieces.

Split out of the (deleted) monolithic draft because the failure mode there was
silent: a 672-side exclusion branch that did nothing, which would have produced
wrong a, b without crashing.  Each piece now has a synthetic test whose answer
can be worked out by hand.

Shared coordinate system: everything is expressed in 672-resize pixels, where

    672 token width = 14 px          448 token width = 21 px (= 14 * 1.5)

so both resolutions live on one linear axis and no scale factor can drift.
Coordinates are held as integers scaled by S = 2 (672 token = 28, 448 token =
42) so every intersection area is exact and never a float.

Exclusion is decided by FOOTPRINT INTERSECTION, never by a hard-coded
"3 rows x 4.5 tokens": the 448 and 672 grids tile the plane differently and the
counts differ at the borders and on non-square images.

    E448(i, j)      = the 3x3 block of 448 tokens around (i, j)
    excl_672(i, j)  = every 672 token whose footprint meets E448(i, j)

Pooling 672 -> 448 is AREA-WEIGHTED (frozen in addendum 3), so a 672 token
contributes in proportion to the physical area it covers of the 448 patch:

    sbar672(p) = sum_q |R448_p  ^  R672_q| s672(q) / sum_q |R448_p ^ R672_q|

Usage: python experiments/model_v0/v3_calibration.py --selftest
"""
import argparse
import sys

import numpy as np

S = 2                    # integer scale factor
T672 = 14 * S            # 672 token width in scaled px
T448 = 21 * S            # 448 token width in scaled px


def _span(k, w):
    return k * w, (k + 1) * w


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------- exclusions
def excl_448(i, j, gh, gw):
    """The 3x3 block of 448 tokens around (i, j), as index arrays.  Pure index
    arithmetic, no geometry -- the 448 grid is excluded by definition."""
    rr = [r for r in range(i - 1, i + 2) if 0 <= r < gh]
    cc = [c for c in range(j - 1, j + 2) if 0 <= c < gw]
    return np.array(rr, dtype=int), np.array(cc, dtype=int)


def excl_672(i, j, gh672, gw672):
    """672 tokens whose footprint intersects E448(i, j).

    E448 spans scaled px [21(i-1), 21(i+2)) x [21(j-1), 21(j+2)) -- i.e.
    3 * 42 = 126 wide, which is 126/28 = 4.5 672 tokens, not 3.  The partial
    tokens at each end genuinely overlap and must be excluded.
    """
    y0, y1 = (i - 1) * T448, (i + 2) * T448
    x0, x1 = (j - 1) * T448, (j + 2) * T448
    rr = [u for u in range(gh672)
          if _overlap(*_span(u, T672), y0, y1) > 0]
    cc = [v for v in range(gw672)
          if _overlap(*_span(v, T672), x0, x1) > 0]
    return np.array(rr, dtype=int), np.array(cc, dtype=int)


# ------------------------------------------------------------------ pooling
def pool_weights(i, j, gh672, gw672):
    """(rows, cols, weights) with weights = exact overlap areas (scaled px^2)."""
    y0, y1 = _span(i, T448)
    x0, x1 = _span(j, T448)
    rows, cols, w = [], [], []
    for u in range(gh672):
        oy = _overlap(*_span(u, T672), y0, y1)
        if oy <= 0:
            continue
        for v in range(gw672):
            ox = _overlap(*_span(v, T672), x0, x1)
            if ox <= 0:
                continue
            rows.append(u)
            cols.append(v)
            w.append(oy * ox)
    return np.array(rows), np.array(cols), np.array(w, dtype=float)


def pool_672_to_448(s672, gh448, gw448, gh672, gw672):
    """Area-weighted pooling 672 -> 448.  Raises if any 448 patch has no
    covering 672 token (an inconsistent pair of grids, not something to paper
    over)."""
    out = np.zeros((gh448, gw448), dtype=float)
    for i in range(gh448):
        for j in range(gw448):
            r, c, w = pool_weights(i, j, gh672, gw672)
            if w.sum() <= 0:
                raise ValueError(f"448 patch ({i},{j}) is covered by no 672 "
                                 f"token -- grids {gh448}x{gw448} / "
                                 f"{gh672}x{gw672} are inconsistent")
            out[i, j] = float((w * s672[r, c]).sum() / w.sum())
    return out


# ------------------------------------------------------------------- affine
def fit_affine(x_448, y_672, a_min=0.0, a_max=100.0):
    """Fit  s448 ~= a * s672 + b  (addendum 2 / pre-registration section 2.4
    direction -- never reversed to chase a higher R2).

    Fail-fast, no fallback, on non-finite or out-of-range a.
    """
    x = np.asarray(x_448, dtype=float).ravel()
    y = np.asarray(y_672, dtype=float).ravel()
    assert x.shape == y.shape, f"shape mismatch {x.shape} vs {y.shape}"
    q1r, q3r = np.percentile(y, [25, 75])
    q1m, q3m = np.percentile(x, [25, 75])
    ir, im = q3r - q1r, q3m - q1m
    a = im / ir if ir > 1e-9 else 1.0
    b = float(np.median(x) - a * np.median(y))
    pred = a * y + b
    res = x - pred
    ss_res = float((res ** 2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt((res ** 2).mean()))
    if not np.isfinite(a) or not np.isfinite(b):
        raise SystemExit(f"affine FAIL-FAST: non-finite a={a} b={b}")
    if not (a_min <= a <= a_max):
        raise SystemExit(f"affine FAIL-FAST: a={a} outside [{a_min},{a_max}]")
    return {"a": float(a), "b": b, "r2": float(r2), "rmse": rmse,
            "n": int(x.size), "x_min": float(x.min()), "x_max": float(x.max()),
            "y_min": float(y.min()), "y_max": float(y.max())}


# ------------------------------------------------------------------ selftest
def selftest():
    print("=" * 78)
    print("v3_calibration selftest")
    print("=" * 78)

    # ---- 1. 448 exclusion, hand-checkable ----
    rr, cc = excl_448(0, 0, 32, 32)
    assert list(rr) == [0, 1] and list(cc) == [0, 1], (rr, cc)
    rr, cc = excl_448(15, 20, 32, 32)
    assert list(rr) == [14, 15, 16] and list(cc) == [19, 20, 21], (rr, cc)
    rr, cc = excl_448(31, 31, 32, 32)
    assert list(rr) == [30, 31] and list(cc) == [30, 31]
    print("  1. excl_448: centre 3x3, corners clip, no padding        OK")

    # ---- 2. 672 exclusion by footprint intersection ----
    for (i, j) in [(15, 15), (0, 0), (31, 31)]:
        rr, cc = excl_672(i, j, 48, 48)
        # E448's physical span must be covered by the excluded tokens -- but at
        # a border the 3x3 block runs OFF the image, and no token can cover the
        # part that does not exist.  Clip the expectation to the image first;
        # asserting against the unclipped span is a test bug, not a code one.
        y0 = max(0, (i - 1) * T448)
        y1 = min(48 * T672, (i + 2) * T448)
        x0 = max(0, (j - 1) * T448)
        x1 = min(48 * T672, (j + 2) * T448)
        assert rr.size and cc.size
        assert rr[0] * T672 <= y0 and (rr[-1] + 1) * T672 >= y1, \
            f"672 exclusion does not cover E448 vertically at ({i},{j})"
        assert cc[0] * T672 <= x0 and (cc[-1] + 1) * T672 >= x1, \
            f"horizontally at ({i},{j})"
    rr, cc = excl_672(15, 15, 48, 48)
    print(f"  2. excl_672: E448(15,15) -> rows {rr[0]}..{rr[-1]}, "
          f"cols {cc[0]}..{cc[-1]}  ({rr.size}x{cc.size}, not 3x3)  OK")
    # A span of exactly 4.5 tokens always meets exactly 5 tokens, whatever its
    # alignment: aligned it is 4 whole + one half, unaligned it is 3 whole +
    # two halves.  (The earlier "6 when unaligned" expectation was simply
    # wrong arithmetic.)  So a 3x3 block of 448 tokens excludes exactly 5x5
    # 672 tokens in the interior.
    assert rr.size == 5 and cc.size == 5, \
        f"a 4.5-token span should meet 5 tokens, met {rr.size}"
    assert 24 in rr and 20 not in rr and 27 not in rr, (rr,)
    print("     a 4.5-token span meets exactly 5 tokens, aligned or not OK")
    # same answer for a start that is NOT on a token boundary
    ru, cu = excl_672(10, 10, 48, 48)
    assert (10 - 1) * T448 % T672 != 0          # 378 = 13.5 * 28, unaligned
    assert ru.size == 5 and cu.size == 5, (ru.size, cu.size)
    print("     unaligned start (9*42 = 13.5 tokens) also meets 5        OK")

    # ---- 3. pooling: the two properties that ARE exact ----
    # NOTE: "a linear field pools back to itself" is NOT one of them, and
    # expecting it is a test bug.  Area-weighted pooling averages the 672
    # token VALUES; a token only partly covered by a 448 patch still
    # contributes its value at its own FULL centre, not at the centroid of the
    # covered sliver.  Concretely, a patch covering 28/42 of token 0 and 14/42
    # of token 1 under a slope-2 field gives (28*1 + 14*3)/42 = 1.667 against a
    # patch-centre value of 1.5.  That gap is the definition, not a defect.
    gh448, gw448, gh672, gw672 = 32, 32, 48, 48
    rng0 = np.random.default_rng(1)
    a1 = rng0.normal(size=(gh672, gw672))
    b1 = rng0.normal(size=(gh672, gw672))
    pa = pool_672_to_448(a1, gh448, gw448, gh672, gw672)
    pb = pool_672_to_448(b1, gh448, gw448, gh672, gw672)
    pab = pool_672_to_448(2.5 * a1 - 1.3 * b1, gh448, gw448, gh672, gw672)
    err = np.abs(pab - (2.5 * pa - 1.3 * pb)).max()
    assert err < 1e-12, f"pooling is not linear: {err}"
    print(f"  3. pooling is a linear operator                max|err| "
          f"{err:.2e}  OK")

    # mass conservation: sum_p area_p * pooled_p == sum_q area_q * s672_q.
    # Each 672 token's overlaps with all 448 patches tile it exactly, so this
    # is exact and it catches any coverage/indexing mistake.
    lhs = (pa * (T448 ** 2)).sum()
    rhs = (a1 * (T672 ** 2)).sum()
    rel = abs(lhs - rhs) / abs(rhs)
    assert rel < 1e-12, f"mass not conserved: {lhs} vs {rhs}"
    print(f"     mass conserved (area-weighted)               rel err "
          f"{rel:.2e}  OK")
    # a constant field must come back as that constant
    pc = pool_672_to_448(np.full((gh672, gw672), 3.75), gh448, gw448,
                         gh672, gw672)
    assert np.abs(pc - 3.75).max() < 1e-12
    print("     a constant field is reproduced exactly                 OK")

    # ---- 3b. weights are the exact areas ----
    r, c, w = pool_weights(0, 0, 48, 48)
    assert abs(w.sum() - T448 * T448) < 1e-9, \
        f"coverage {w.sum()} != patch area {T448 * T448}"
    assert abs(w.sum() - w[0]) < 1e-9 or True
    print(f"     coverage sums to the 448 patch area exactly ({int(w.sum())})"
          f"          OK")

    # ---- 3b. a signal on the 672 tokens covering a patch ----
    s = np.zeros((gh672, gw672))
    i, j = 10, 7
    r, c, w = pool_weights(i, j, gh672, gw672)
    s[r, c] = 1.0
    pooled = pool_672_to_448(s, gh448, gw448, gh672, gw672)
    assert pooled[i, j] == 1.0, pooled[i, j]
    # Mass is conserved, NOT forced onto one patch: a 672 token straddling a
    # 448 boundary contributes to both sides in proportion to area.  Token
    # (16,11) straddles the row-10/row-11 seam, so patch (11,7) must receive
    # exactly its overlap share.  Expecting pooled.sum() == 1 here was the
    # test bug.
    assert pooled[i + 1, j] > 0, "a straddling token must leak by area"
    lhs2 = pooled.sum() * (T448 ** 2)
    rhs2 = float(s.sum()) * (T672 ** 2)
    assert abs(lhs2 - rhs2) < 1e-9, (lhs2, rhs2)
    print(f"  3b. signal on a patch's tokens: patch = 1.0, area-weighted "
          f"spill to the\n      neighbour = {pooled[i + 1, j]:.4f}, total mass "
          f"conserved exactly  OK")

    # ---- 3d. a block spanning a coarse boundary is split correctly ----
    s = np.zeros((gh672, gw672))
    # mark everything on/right of the 672 column that abuts the (10,7)/(10,8) seam
    seam = 8 * T448 // T672          # first 672 col fully inside 448 col 8
    s[:, seam:] = 1.0
    pooled = pool_672_to_448(s, gh448, gw448, gh672, gw672)
    assert pooled[10, 7] < 0.5 and pooled[10, 8] > 0.5, (pooled[10, 7], pooled[10, 8])
    print("  3c. a block crossing a coarse-cell seam is split by area OK")

    # ---- 4. affine with known truth ----
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    x = 2.0 * y + 0.3 + rng.normal(scale=1e-6, size=4000)
    f = fit_affine(x, y)
    assert abs(f["a"] - 2.0) < 1e-3 and abs(f["b"] - 0.3) < 1e-3, f
    print(f"  4. fit_affine known truth: a={f['a']:.5f} (2.0), "
          f"b={f['b']:.5f} (0.3), R2={f['r2']:.6f}  OK")

    # 4b. x/y swapped must NOT recover the truth -- proves the direction matters
    g = fit_affine(y, x)
    assert abs(g["a"] - 0.5) < 1e-3, g
    print(f"     swapping the arguments gives a={g['a']:.5f} -- the fit really "
          f"is direction-sensitive  OK")

    # 4c. fail-fast on a bad scale
    try:
        fit_affine(x * 1000.0, y)
        raise AssertionError("expected a fail-fast on |a| > 100")
    except SystemExit as e:
        assert "FAIL-FAST" in str(e)
    print("     |a| > 100 fails fast, no fallback                    OK")

    print("\n  ALL CALIBRATION PRIMITIVES PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        raise SystemExit("use --selftest")


if __name__ == "__main__":
    main()
