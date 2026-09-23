# -*- coding: utf-8 -*-
"""
Phase 3C readout: how much of the real defect set does config E actually need?

Three probes, all on the same held-out half as E:
  rank scan       PCA reconstruction of the real defect set at rank k
  prototype scan  each defect patch replaced by one of K k-means centroids
  pooled transfer multi-donor defect subspace applied to the target's normals

Reading it correctly requires three corrections that a naive mean-over-objects
table gets wrong:

  1. E does not help everywhere. E - A_half is +13.8 (wallplugs), +5.6
     (sheet_metal), 0.0 (vial), -1.8 (can). A "gap closure" ratio with a
     negative denominator silently flips sign, so closure is only computed on
     the objects where E actually improves, and the rest are reported raw.
  2. Rank is not information. rank-20 keeps only 0.81-0.86 of the defect set's
     variance, so it is not a fair "low-dimensional" test. The reported x-axis
     is explained variance, recomputed here from the cache.
  3. A config whose pseudo-defects all start above the hinge threshold has zero
     separation gradient and silently degenerates into config D. Those cells
     are flagged, not read as failures.

Usage: python analysis/model_v0/scan_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import load_cache  # noqa: E402

RESULTS = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
RANKS = [1, 3, 5, 10, 20, 40, 70, 120]
PROTOS = [1, 2, 4, 8, 16, 32, 64]
# Hinge-inactive cells found by proxy_defects.report_distances(): the single
# prototype already sits above normal_p99 + margin for these objects.
HINGE_DEAD = {"P1": {"can", "sheet_metal"}}


def explained_variance(obj, k):
    """Fraction of the real defect set's variance kept at rank k."""
    tr, te = load_cache(obj)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(0).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    X = np.concatenate([te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                        for i in sup])
    s = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False) ** 2
    k = min(k, len(s))
    return float(s[:k].sum() / s.sum())


def main():
    df = pd.read_csv(RESULTS)
    base = df[df.config == "A_half"].set_index("object")
    ceil = df[df.config == "E"].set_index("object")
    gain = (ceil.img_AUROC - base.img_AUROC).round(1)
    helps = [o for o in OBJECTS if gain.get(o, 0) > 1.0]

    print("=" * 96)
    print("E's gain over A_half, per object "
          "(>1 means E actually helps; the scans can only be read there)")
    print("=" * 96)
    print("  " + "  ".join(f"{o}={gain[o]:+.1f}" for o in OBJECTS
                           if o in gain.index))
    print(f"  readable objects: {helps}")

    def table(title, cfgs, xlab, xval, note):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        for o in helps:
            print(f"\n--- {o} ---")
            print(f"{'config':<8}{xlab:>12}{'img_AUROC':>11}{'defect<p99':>12}"
                  f"{'defect_mean':>13}{'closure(img)':>14}")
            a = base.loc[o]
            e = ceil.loc[o]
            for c in ["A_half"] + cfgs + ["E"]:
                r = df[(df.object == o) & (df.config == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = xval(c, o)
                clo = ((r.img_AUROC - a.img_AUROC) / (e.img_AUROC - a.img_AUROC)
                       if c not in ("A_half", "E") else np.nan)
                flag = ""
                if c in HINGE_DEAD and o in HINGE_DEAD[c]:
                    flag = "  <- hinge inactive, degenerates to D"
                    clo = np.nan
                print(f"{c:<8}{('-' if x is None else f'{x:.3f}'):>12}"
                      f"{r.img_AUROC:>11.1f}{r.frac_defect_below_p99:>12.3f}"
                      f"{r.defect_mean:>13.3f}"
                      f"{('' if np.isnan(clo) else f'{clo:+.2f}'):>14}{flag}")
        print(f"\n  {note}")

    table("PROBE 1 -- rank scan of the REAL defect set",
          [f"R{k}" for k in RANKS], "expvar",
          lambda c, o: (None if c == "A_half" else
                        (1.0 if c == "E" else
                         explained_variance(o, int(c[1:]))))
          if c.startswith("R") or c == "E" else None,
          "knee at rank ~40 (expvar ~0.94): the information IS compressible, "
          "but only to a ~40-dim CONTINUOUS subspace -- rank <=20 fails. "
          "Read this against PROBE 2: prototypes fail at the same budget.")

    table("PROBE 2 -- prototype count of the REAL defect set",
          [f"P{K}" for K in PROTOS], "K", lambda c, o: None,
          "saturates near half of E even at K=64, while PROBE 1 recovers E at "
          "rank ~40 -> the structure is a CONTINUOUS ~40-dim subspace, not a "
          "discrete set of defect modes")

    print("\n" + "=" * 96)
    print("PROBE 3 -- pooled cross-object transfer")
    print("=" * 96)
    for o in helps:
        a, e = base.loc[o], ceil.loc[o]
        print(f"\n--- {o} ---")
        for c in ["A_half", "G1", "G2", "X1", "E"]:
            r = df[(df.object == o) & (df.config == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {c:<8} img_AUROC={r.img_AUROC:>6.1f}  "
                  f"defect<p99={r.frac_defect_below_p99:.3f}  "
                  f"defect_mean={r.defect_mean:.3f}")
    print("\n  X1 << E on every readable object -> the defect information does "
          "not transfer across object categories")


if __name__ == "__main__":
    main()
