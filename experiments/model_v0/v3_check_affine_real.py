# -*- coding: utf-8 -*-
"""
V3 real-data affine health check -- protocol and numerics ONLY.

Answers one question: do the primitives that pass on synthetic data behave
correctly on real DINO features?  It reads NO detection metric, and by design
cannot: nothing here computes an anomaly score.

Per (object, shot, split) it walks the frozen path

    S_k -> s448_cal -> s672_cal -> Pool_672->448 -> fit(a, b)

and reports health.  Hard conditions are exactly the frozen ones -- a > 0,
|a| <= 100, finite -- plus protocol correctness.  R2 / Pearson / Spearman /
RMSE are DIAGNOSTICS ONLY: a low R2 is not a failure and must never trigger a
switch to isotonic / quantile / piecewise / log calibration, nor dropping
outliers or selecting high-R2 objects.

PAIRING IS BY EXPLICIT KEY (image_id, row448, col448), never by flatten order.
Equal lengths prove nothing about correspondence; a mis-paired fit would still
produce finite, plausible-looking a and b.

Usage: python experiments/model_v0/v3_check_affine_real.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import draw_images  # noqa: E402
from phase_m1_rescue import V1, VISA, load_cached  # noqa: E402
from v3_calibration import (excl_448, excl_for_672, pool_672_to_448,  # noqa: E402
                            cal_scores_from_D, fit_affine)
import v3_refine as rf  # noqa: E402

LAYERS = ["mid", "midlate", "final"]
CASES = [("bottle", 1), ("bottle", 8), ("pcb2", 1), ("pcb2", 8)]
SPLIT = 0
C448_TR = os.path.join(RESULTS, "cache_ml_img")
C448_TR_VISA = os.path.join(RESULTS, "cache_visa")
C672_TR = rf.C672


def _dists(feats, offs, which):
    o0, o1 = int(offs[which]), int(offs[which + 1])
    Q = torch.from_numpy(feats[o0:o1].astype(np.float32))
    B = torch.from_numpy(feats.astype(np.float32))
    Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-12)
    B = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    return (1.0 - Q @ B.T).numpy()


def _pmap(cache, pfx, layer, obj, which):
    o = cache[layer]["offsets"]
    return np.concatenate([cache[layer]["feats"][o[i]:o[i + 1]]
                           for i in which])


def run_case(obj, shot):
    # reuse the already-verified loader (it infers `grids` for caches that do
    # not store them, refusing when patch counts vary across images)
    import v3_selector as sel
    vis, c448 = sel.load_split(obj, "train")
    c672 = rf.load_672(obj, "train")

    idx = draw_images(c448[LAYERS[0]]["offsets"], shot, SPLIT, obj)
    assert len(idx) == shot, f"drew {len(idx)} support images for k={shot}"

    g4 = {tuple(int(v) for v in c448[LAYERS[0]]["grids"][i]) for i in idx}
    g6 = {tuple(int(v) for v in c672[LAYERS[0]]["grids"][i]) for i in idx}
    if len(g4) != 1 or len(g6) != 1:
        raise SystemExit(f"STOP: support images have differing grids "
                         f"448={g4} 672={g6} -- per-image geometry needed")
    gh4, gw4 = g4.pop()
    gh6, gw6 = g6.pop()
    n4, n6 = gh4 * gw4, gh6 * gw6

    off4 = np.arange(0, shot * n4 + 1, n4, dtype=np.int64)
    off6 = np.arange(0, shot * n6 + 1, n6, dtype=np.int64)

    # ---- per image, per layer: guard maps at both resolutions ----
    z4, z6 = {}, {}
    for l in LAYERS:
        f4 = np.concatenate([c448[l]["feats"][c448[l]["offsets"][i]:
                                              c448[l]["offsets"][i + 1]]
                             for i in idx])
        f6 = np.concatenate([c672[l]["feats"][c672[l]["offsets"][i]:
                                              c672[l]["offsets"][i + 1]]
                             for i in idx])
        a4, a6 = [], []
        for m in range(shot):
            m4 = cal_scores_from_D(_dists(f4, off4, m), gh4, gw4,
                                   int(off4[m]), excl_448, f"{obj}/k{shot}/448")
            m6 = cal_scores_from_D(_dists(f6, off6, m), gh6, gw6,
                                   int(off6[m]), excl_for_672,
                                   f"{obj}/k{shot}/672")
            a4.append(m4)
            a6.append(pool_672_to_448(m6, gh4, gw4, gh6, gw6))
        z4[l], z6[l] = a4, a6

    # agg_all3-style: z-score each layer over that image's patches, then mean
    S4 = np.mean([[(m - m.mean()) / (m.std() + 1e-12) for m in z4[l]]
                  for l in LAYERS], axis=0)
    S6 = np.mean([[(m - m.mean()) / (m.std() + 1e-12) for m in z6[l]]
                  for l in LAYERS], axis=0)

    # ---- explicit pairing key: (support image id, row448, col448) ----
    rec = {}
    for m in range(shot):
        for r in range(gh4):
            for c in range(gw4):
                rec[(int(idx[m]), r, c)] = (S4[m][r, c], S6[m][r, c])
    keys = sorted(rec.keys())
    x = np.array([rec[k][0] for k in keys])      # 448 side
    y = np.array([rec[k][1] for k in keys])      # pooled-672 side
    assert x.size == y.size == shot * n4, (x.size, y.size, shot * n4)

    fit = fit_affine(x, y)
    pear = float(np.corrcoef(x, y)[0, 1])
    from scipy.stats import spearmanr
    spr = float(spearmanr(x, y).statistic)
    return dict(obj=obj, shot=shot, vis=vis, idx=sorted(int(i) for i in idx),
                pairs=x.size, x=x, y=y, fit=fit, pear=pear, spr=spr,
                grid4=(gh4, gw4), grid6=(gh6, gw6))


def main():
    print("=" * 100)
    print("V3 real-data affine health check -- protocol/numerics only, no "
          "detection metric")
    print("=" * 100)
    for obj, shot in CASES:
        r = run_case(obj, shot)
        f = r["fit"]
        x, y = r["x"], r["y"]
        print(f"\n  {obj}  k={shot}  split={SPLIT}")
        print(f"    support_ids   {r['idx']}   (n={len(r['idx'])})")
        print(f"    grids         448 {r['grid4']}   672 {r['grid6']}")
        print(f"    N_pairs       {r['pairs']}  = {len(r['idx'])} x "
              f"{r['pairs'] // len(r['idx'])}  (explicit (img,row,col) keys)")
        q = lambda v: (f"min {v.min():+.3f}  med {np.median(v):+.3f}  "
                       f"mean {v.mean():+.3f}  p95 {np.percentile(v, 95):+.3f}  "
                       f"max {v.max():+.3f}")
        print(f"    s448_cal      {q(x)}")
        print(f"    s672_pooled   {q(y)}")
        print(f"    affine        a {f['a']:+.5f}   b {f['b']:+.5f}   "
              f"R2 {f['r2']:+.4f}   RMSE {f['rmse']:.5f}")
        print(f"    corr          Pearson {r['pear']:+.4f}   "
              f"Spearman {r['spr']:+.4f}")
        ok = (np.isfinite([f['a'], f['b']]).all() and 0 < f['a'] <= 100)
        print(f"    health        finite {bool(np.isfinite(x).all() and np.isfinite(y).all())}"
              f"   a>0 {f['a'] > 0}   |a|<=100 {abs(f['a']) <= 100}   "
              f"support-only ids={r['idx']}   -> "
              f"{'OK' if ok else 'STOP: implementation failure'}")
        print(f"    NOTE          R2 is diagnostic only; a low value is NOT a "
              f"failure and\n                  must not trigger a different "
              f"calibration.")


if __name__ == "__main__":
    main()
