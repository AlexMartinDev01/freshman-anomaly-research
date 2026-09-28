# -*- coding: utf-8 -*-
"""
Phase R step 5 -- the downstream error matrix: agg_all3-kNN vs SubspaceAD-B.

Reads `metrics/gate_r2_subspacead_b.csv`, where both columns were computed on
the SAME k-shot images with the SAME evaluator, and lays out a 15 objects x
{1,2,4,8}-shot matrix of

    delta = SubspaceAD - agg_all3_kNN

for image AUROC, pixel AUROC and AUPRO, then sorts every (object, shot) cell
into the four cases that decide what the next modelling question is:

    SA_both        SubspaceAD wins image AND pixel -> normal-subspace modelling
                   is the real direction.
    COMPLEMENT     one side wins image, the other wins localisation -> the two
                   scoring mechanisms are complementary, and an oracle fusion
                   is worth one bounded test.
    KNN_HARD_LOC   our kNN wins localisation on the objects where BOTH methods
                   are weak (the tile / transistor regime) -> analyse what
                   multi-level nearest-neighbour captures that a PCA residual
                   does not. This is the candidate new-method wedge.
    BOTH_FAIL      both methods land below a fixed AUPRO floor -> that is the
                   gap the next model should attack, independent of scoring.

The AUPRO floor is fixed here rather than tuned: 70.0, which is roughly the
midpoint between the worst (tile/transistor ~50-65) and the best (~95)
baselines and is stated before looking at the numbers.

Usage: python experiments/model_v0/gate_r2_error_matrix.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
AUPRO_FLOOR = 70.0
METS = ["img_AUROC", "px_AUROC", "AUPRO"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gate_r2_subspacead_b")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(METRICS, f"{a.tag}.csv"))
    print(f"loaded {len(df)} cells, aug values: {sorted(df['aug'].unique())}")

    for aug in sorted(df["aug"].unique()):
        S = df[df["aug"] == aug].copy()
        print("\n" + "=" * 100)
        print(f"ERROR MATRIX  aug_count={aug}   "
              f"delta = SubspaceAD - agg_all3_kNN  (paired, same shots, same evaluator)")
        print("=" * 100)
        for met in METS:
            S[f"d_{met}"] = S[f"sa_{met}"] - S[f"knn_{met}"]
        # only the delta columns exist by that name; the raw values are sa_*/knn_*
        piv = S.pivot_table(index="object", columns="shot",
                            values=[f"d_{m}" for m in METS])
        print("\n  mean delta per object x shot (rows=object, cols=shot):")
        for met in METS:
            print(f"\n   {met}  [knn mean {S['knn_' + met].mean():.2f} -> "
                  f"SA mean {S['sa_' + met].mean():.2f}, "
                  f"delta {S['d_' + met].mean():+.2f}, "
                  f"SA better {int((S['d_' + met] > 0).sum())}/{len(S)}]")
            print(piv[f"d_{met}"].round(1).to_string())

        # ---- classify cells ----
        print("\n  " + "-" * 96)
        print("  CASE ASSIGNMENT (per object, averaged over shots and seeds)")
        print("  " + "-" * 96)
        g = S.groupby("object").mean(numeric_only=True)
        rows = []
        for obj, r in g.iterrows():
            d_img, d_px, d_pr = r.d_img_AUROC, r.d_px_AUROC, r.d_AUPRO
            weak = min(r.knn_AUPRO, r.sa_AUPRO) < AUPRO_FLOOR
            if d_img > 0 and d_px > 0 and d_pr > 0:
                case = "SA_both"
            elif d_img < 0 and d_pr < 0:
                case = "KNN_HARD_LOC" if weak else "KNN_wins"
            elif d_img < 0 < d_pr:
                case = "COMPLEMENT"
            elif d_pr < 0 < d_img:
                case = "COMPLEMENT"
            else:
                case = "TIE"
            if weak and d_pr <= 0:
                case += " +BOTH_FAIL"
            rows.append({"object": obj, "case": case,
                         "knn_AUPRO": r.knn_AUPRO, "sa_AUPRO": r.sa_AUPRO,
                         "d_AUPRO": d_pr, "d_img": d_img, "d_px": d_px,
                         "knn_img": r.knn_img_AUROC, "sa_img": r.sa_img_AUROC})
        c = pd.DataFrame(rows).sort_values(["case", "d_AUPRO"])
        print(c.round(2).to_string(index=False))
        print("\n  case counts:")
        print(c["case"].value_counts().to_string())

        # ---- where do both fail? ----
        print(f"\n  objects where BOTH are weak (min(AUPRO) < {AUPRO_FLOOR}):")
        w = c[c[["knn_AUPRO", "sa_AUPRO"]].min(axis=1) < AUPRO_FLOOR]
        if len(w):
            for _, r in w.iterrows():
                print(f"    {r['object']:<12} knn {r['knn_AUPRO']:5.1f}   "
                      f"SA {r['sa_AUPRO']:5.1f}   delta {r['d_AUPRO']:+5.1f}")
        else:
            print("    (none)")

    print("\n  marginals over all cells:")
    for met in METS:
        d = df[f"sa_{met}"] - df[f"knn_{met}"]
        print(f"    {met:<10} knn {df['knn_' + met].mean():6.2f}   "
              f"SA {df['sa_' + met].mean():6.2f}   delta {d.mean():+6.2f}   "
              f"SA better {int((d > 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
