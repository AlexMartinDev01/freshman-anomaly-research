# -*- coding: utf-8 -*-
"""
Model V3 -- A3 orchestration (Stage 2 trigger -> Stage 3 crops -> Stage 4 map).

ORCHESTRATION ONLY.  Every piece of numerics is imported from a module whose
primitives already pass synthetic, hand-checkable tests:

    v3_selector     calibration, trigger, budget allocation
    v3_calibration  exclusion, pooling, affine fit
    v3_refine       crop geometry, 672 bank guard, writeback
    v3_check_affine_real  the fitted a, b for a (object, shot, split)

Nothing here re-implements exclusion, pooling, affine, budget or writeback.
The A3/A4/A5 arms differ in exactly one value, `selected_positions`
(see `choose`), so a later difference between arms cannot come from geometry.

Usage: python experiments/model_v0/v3_run.py --objects bottle,transistor
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate_r2_layer_confirm import draw_images, l2n  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_calibration as cal  # noqa: E402
import v3_check_affine_real as aff  # noqa: E402
import v3_check_scale as cs  # noqa: E402


def pool_crop_to_448(crop, r0, c0, gh448, gw448, gh672, gw672):
    """Area-pool a 672-token crop map onto the 448 patches it covers.

    Reuses pool_weights with the GLOBAL grid size, then admits only the crop's
    own token range -- passing the crop extent as if it were the grid size
    would silently truncate the token range and mis-weight the edge patches.
    """
    n = crop.shape[0]
    i0, j0 = cal.containing_448(r0, c0)
    i1, j1 = cal.containing_448(r0 + n - 1, c0 + n - 1)
    out = {}
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if not (0 <= i < gh448 and 0 <= j < gw448):
                continue
            r, c, w = cal.pool_weights(i, j, gh672, gw672)
            m = (r >= r0) & (r < r0 + n) & (c >= c0) & (c < c0 + n)
            if not m.any() or w[m].sum() <= 0:
                continue
            out[(i, j)] = float((w[m] * crop[r[m] - r0, c[m] - c0]).sum()
                                / w[m].sum())
    return out


def run_arm(obj, shot, split, mode, model, resize, totensor, norm,
            c448, c672, te448, tau_hot, tau_gap, a_, b_, seed_i=0, gt=None):
    """A0 -> trigger -> budget -> crops -> 672 score -> affine -> writeback.

    `mode` in {a3, a4, a5} changes ONLY where the refined positions come from:

        a3  the selector, from the fully-triggered (hot AND low-gap) set
        a4  uniformly random from the LARGER hot-only pool   (frozen A1.3)
        a5  GT oracle, ranked within that same hot pool

    crop size, budget, 672 bank, scoring, affine and writeback are literally
    the same code for all three, so a later difference between arms cannot be
    a geometry artefact.  `gt` is the per-448-patch GT fraction, used by a5
    ONLY -- it never enters the a3/a4 path.
    """
    idx = draw_images(c448[rf.LAYERS[0]]["offsets"], shot, split, obj)
    tro = c448[rf.LAYERS[0]]["offsets"]
    banks = {}
    for l in rf.LAYERS:
        o6 = c672[l]["offsets"]
        banks[l] = l2n(torch.from_numpy(np.concatenate(
            [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
        ).astype(np.float32)).to(rf.DEV))

    S = sel.test_maps(obj, c448, te448, shot, split)
    teg = te448[rf.LAYERS[0]]["grids"]
    import cv2
    from PIL import Image
    names = [str(x) for x in te448[rf.LAYERS[0]]["names"]]
    root = rf.V1 if os.path.isdir(os.path.join(rf.V1, obj)) else rf.VISA

    recs = []
    for i, m0 in enumerate(S):
        gh4, gw4 = int(teg[i][0]), int(teg[i][1])
        M0 = m0.reshape(gh4, gw4).astype(np.float64)
        h, g = sel.window_stats(M0)
        fire = (h >= tau_hot) & (g <= tau_gap)
        m = int(fire.sum())
        rows, cols = np.mgrid[0:h.shape[0], 0:h.shape[1]]
        grid672 = (int(np.ceil(gh4 * 1.5)), int(np.ceil(gw4 * 1.5)))
        B = sel.budget(grid672)
        m2, n = sel.allocate(m, B)
        pool = np.where((h >= tau_hot).ravel())[0]
        cand = np.where(fire.ravel())[0]
        # runtime guard: the selector's set is always a subset of the hot pool
        assert np.isin(cand, pool).all(), "selector set escaped the hot pool"
        # a5 oracle signal: mean GT fraction over the window's patches
        gtwin = None
        if mode == "a5":
            gf = gt[i].reshape(gh4, gw4)
            gtwin = np.array([[gf[r:r + 3, c:c + 3].mean()
                               for c in range(g.shape[1])]
                              for r in range(g.shape[0])]).ravel()
        use = cand if mode == "a3" else pool
        pos = rf.choose(mode, use, m2, h.ravel(), g.ravel(),
                        rows.ravel(), cols.ravel(), obj, seed_i=seed_i,
                        gt=gtwin)
        # a4/a5 draw from the hot pool, so they always have enough candidates
        assert len(pos) == m2
        assert m2 * n * n <= B, f"budget violated {m2}*{n}^2 > {B}"

        # ---- m = 0: A3 must be BIT-IDENTICAL to A0, not merely close ----
        if m2 == 0:
            A3 = M0.copy()
            recs.append(dict(image=names[i], m=m, m2=0, n=0, T=0, B=B,
                             cap_hit=int(m > int(B // 25)), a=a_, b=b_,
                             shape=A3.shape, pos=pos,
                             identical=bool(np.array_equal(A3, M0))))
            continue

        src = os.path.join(root, obj, "test", *names[i].split("/"))
        big = rf.resized_grid(cv2.imread(src, cv2.IMREAD_COLOR), resize)
        gh6, gw6 = big.size[1] // rf.STRIDE, big.size[0] // rf.STRIDE

        acc = {}
        for p in pos:
            r448, c448_ = int(rows.ravel()[p]), int(cols.ravel()[p])
            r0, c0 = rf.window_672(r448, c448_, n, gh6, gw6)
            feats = rf.crop_feats(model, big, r0, c0, n, norm, totensor)
            cmap = rf.agg3(feats, [banks[l] for l in rf.LAYERS]).reshape(n, n)
            for k, v in pool_crop_to_448(cmap, r0, c0, gh4, gw4,
                                         gh6, gw6).items():
                acc.setdefault(k, []).append(v)
        A3 = M0.copy()
        for (r, c), vs in acc.items():
            A3[r, c] = a_ * float(np.mean(vs)) + b_

        assert A3.shape == M0.shape, "writeback changed the map shape"
        assert np.isfinite(A3).all(), "non-finite final map"
        recs.append(dict(image=names[i], m=m, m2=m2, n=n, T=m2 * n * n, B=B,
                         cap_hit=int(m > int(B // 25)), a=a_, b=b_,
                         shape=A3.shape, pos=pos, integrated=len(pos),
                         map=A3))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="bottle,transistor")
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--split", type=int, default=0)
    a = ap.parse_args()
    print("A3/A4/A5 orchestration -- engineering verification, NO detection "
          "metric read")
    model, resize, totensor, norm = cs.build()
    N_SEEDS = rf.N_SEEDS
    for obj in a.objects.split(","):
        t0 = time.time()
        _, c448 = sel.load_split(obj, "train")
        _, te448 = sel.load_split(obj, "test")
        c672 = rf.load_672(obj, "train")
        tau_hot, tau_gap, _ = sel.calibrate(obj, a.shot, a.split, c448)
        r = aff.run_case(obj, a.shot)
        a_, b_ = r["fit"]["a"], r["fit"]["b"]

        # per-448-patch GT fraction (a5 ONLY; never touches a3/a4)
        o = te448[rf.LAYERS[0]]["offsets"]
        gtf = te448[rf.LAYERS[0]]["gt_frac"]
        gr = te448[rf.LAYERS[0]]["grids"]
        gt = [np.asarray(gtf[o[i]:o[i + 1]]).reshape(int(gr[i][0]),
                                                     int(gr[i][1]))
              for i in range(len(gr))]

        arms = {"a3": run_arm(obj, a.shot, a.split, "a3", model, resize,
                              totensor, norm, c448, c672, te448, tau_hot,
                              tau_gap, a_, b_)}
        for s in range(N_SEEDS):
            arms[f"a4s{s}"] = run_arm(obj, a.shot, a.split, "a4", model,
                                      resize, totensor, norm, c448, c672,
                                      te448, tau_hot, tau_gap, a_, b_, seed_i=s)
        arms["a5"] = run_arm(obj, a.shot, a.split, "a5", model, resize,
                             totensor, norm, c448, c672, te448, tau_hot,
                             tau_gap, a_, b_, gt=gt)

        n = len(arms["a3"])
        nzero = sum(1 for x in arms["a3"] if x["m2"] == 0)
        ident = sum(1 for x in arms["a3"] if x.get("identical"))
        integ = [x for x in arms["a3"] if x["m2"] > 0]
        Ts = np.array([[x["T"] for x in arms[k]] for k in arms])
        Bs = np.array([x["B"] for x in arms["a3"]])
        print(f"\n  {obj} k={a.shot} s{a.split}  ({time.time() - t0:.0f}s)")
        print(f"    images {n}   m=0 and bit-identical to A0: {ident}/{n}"
              f"   refined: {len(integ)}")
        print(f"    tau_hot {tau_hot:+.4f}  tau_gap {tau_gap:+.4f}  "
              f"affine a={a_:.5f} b={b_:.5f}   arms {len(arms)}")
        if integ:
            nn = sorted(set(x["n"] for x in integ))
            print(f"    refined: m {min(x['m'] for x in integ)}.."
                  f"{max(x['m'] for x in integ)}, m' "
                  f"{min(x['m2'] for x in integ)}..{max(x['m2'] for x in integ)}"
                  f", n {nn}, cap_hit {sum(x['cap_hit'] for x in integ)}"
                  f"/{len(integ)}")
        # ---- orchestration-level section 10-F: budget equality per image ----
        same = (Ts == Ts[0:1]).all()
        print(f"    BUDGET EQUALITY (m', n, T) across A3 + {N_SEEDS} A4 seeds "
              f"+ A5: {'OK' if same else 'VIOLATION'}")
        assert same, "budget differs across arms at orchestration level"
        assert (Ts <= Bs).all(), "budget exceeded"
        # m=0 must be bit-identical in EVERY arm
        for k, rec in arms.items():
            for idx_, x in enumerate(rec):
                if x["m2"] == 0:
                    m0 = arms["a3"][idx_]
                    assert x.get("identical") and m0.get("identical"), \
                        f"{k}: an m=0 image is not bit-identical to A0"
        ndiff = sum(1 for i in range(len(integ))
                    if not np.array_equal(np.sort(arms["a3"][i]["pos"]),
                                          np.sort(arms["a4s0"][i]["pos"])))
        print(f"    G3 diagnostic: A3 positions differ from A4(seed 0) on "
              f"{ndiff}/{len(integ)} refined images (A3==A4 is expected "
              f"sometimes, but\n      high agreement everywhere would mean the "
              f"random arm is not random)")
        print(f"    util (T/B): mean {float((Ts / Bs).mean()):.4f}  "
              f"max {float((Ts / Bs).max()):.4f}")
    print("\n  A3/A4/A5 ORCHESTRATION OK")


if __name__ == "__main__":
    main()
