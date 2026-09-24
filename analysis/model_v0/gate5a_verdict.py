# -*- coding: utf-8 -*-
"""
Gate 5A verdict: is the defect rank-40 SUBSPACE sufficient, or do the
coefficients inside it carry information the subspace does not?

Why this gate exists: R40 (the earlier positive control) keeps U40 AND the real
coefficients c_i = U40^T (z_i - mu). Reading its success as "predicting
U_target is enough" was an untested inference. This gate holds U40 fixed at the
target's true defect basis and varies ONLY the coefficient source.

Configs (all in the same perturbation construction, alpha calibrated to
0.6 x normal-p99, hinges verified active at 99-100%):
    H0b  real coefficients            in-construction positive control
    H1   isotropic c ~ N(0, I)        the subspace alone, no shape
    H2   c ~ N(0, Sigma_c)            subspace + real second-order shape
    H3   c = R_source @ U40^T         external anomaly residuals projected
                                      into the CORRECT target basis

Reference rows: A_half (floor), E (full real defect set), R40 (absolute-point
positive control).

Decision table, fixed before reading the numbers:
    H1 ~ H0b            -> only U_target needs predicting
    H1 x, H2 ~ H0b      -> need U_target + covariance
    H2 x, H0b ~ E       -> need the full conditional defect distribution
    H3 ~ H0b            -> external coefficient distribution is reusable
    H3 x                -> the anomaly distribution is itself object-specific

Usage: python analysis/model_v0/gate5a_verdict.py
"""
import sys

import numpy as np
import pandas as pd

CSV = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
COLLAPSED = ["can", "wallplugs"]
HEALTHY = ["vial", "sheet_metal"]
DEFECT_FLOOR = 0.80
HEALTH_TOL = 2.0
LABEL = {"A_half": "floor (frozen DINOv2)", "E": "ORACLE (full real defects)",
         "R40": "R40 (absolute-point control)", "H0b": "H0b real coefficients",
         "H1": "H1 isotropic coeffs", "H2": "H2 cov-matched coeffs",
         "H3": "H3 external coeffs"}
ORDER = ["A_half", "E", "R40", "H0b", "H1", "H2", "H3"]


def main():
    df = pd.read_csv(CSV)
    have = [c for c in ORDER if c in set(df.config)]
    if not any(c in set(df.config) for c in ("H0b", "H1", "H2", "H3")):
        sys.exit("no Gate 5A configs found - run run_oracle.py --configs "
                 "H0b H1 H2 H3 --eval-half first")
    base = df[df.config == "A_half"].set_index("object")

    print("=" * 104)
    print("GATE 5A -- fixed true rank-40 basis, coefficient source varied")
    print("=" * 104)
    for o in base.index:
        print(f"\n--- {o} ---")
        print(f"  {'config':<30}{'img_AUROC':>11}{'AUPRO':>9}{'normal_p99':>12}"
              f"{'defect_mean':>13}{'defect<p99':>12}")
        for c in have:
            r = df[(df.object == o) & (df.config == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {LABEL.get(c, c):<30}{r.img_AUROC:>11.1f}"
                  f"{r['AUPRO@0.05']:>9.1f}{r.normal_p99:>12.3f}"
                  f"{r.defect_mean:>13.3f}{r.frac_defect_below_p99:>12.3f}")

    print("\n" + "=" * 104)
    print("CRITERIA (same three as Gate 3/4; all must hold)")
    print("=" * 104)
    ref = {}
    for c in ("H0b", "H1", "H2", "H3"):
        sub = df[df.config == c].set_index("object")
        if sub.empty:
            continue
        c1, c2, c3 = [], [], []
        for o in base.index:
            if o not in sub.index:
                continue
            b, r = base.loc[o], sub.loc[o]
            tail = r.normal_p99 < b.normal_p99
            pres = (r.defect_mean >= DEFECT_FLOOR * b.defect_mean
                    and r.frac_defect_below_p99 < b.frac_defect_below_p99)
            if o in COLLAPSED:
                det = (r.img_AUROC > b.img_AUROC
                       and r["AUPRO@0.05"] > b["AUPRO@0.05"])
            else:
                det = r.img_AUROC >= b.img_AUROC - HEALTH_TOL
            c1.append(tail); c2.append(pres); c3.append(det)
        ref[c] = (sum(c1), sum(c2), sum(c3), len(c1))
        print(f"  {LABEL.get(c, c):<30} tail {sum(c1)}/{len(c1)}   "
              f"preserve {sum(c2)}/{len(c2)}   detect {sum(c3)}/{len(c3)}")

    print("\n" + "=" * 104)
    print("DECISION")
    print("=" * 104)
    e_row = df[df.config == "E"].set_index("object")
    for o in base.index:
        g = []
        for c in ("H0b", "H1", "H2", "H3"):
            r = df[(df.object == o) & (df.config == c)]
            if r.empty or o not in e_row.index:
                continue
            g.append(f"{c}={r.iloc[0].img_AUROC:5.1f}")
        if g:
            print(f"  {o:<12} E={e_row.loc[o,'img_AUROC']:5.1f}  A={base.loc[o,'img_AUROC']:5.1f}  "
                  + "  ".join(g))
    print("\n  Which coefficient source reaches H0b/E decides what a model must "
          "output:")
    print("    H1 ~ H0b          -> predict U_target only")
    print("    H1 x, H2 ~ H0b    -> predict U_target + covariance")
    print("    H2 x, H0b ~ E     -> predict the full conditional distribution")
    print("    H3 ~ H0b          -> external coefficients are reusable; only "
          "the basis is object-specific")
    print("    H3 x              -> the anomaly distribution is object-specific "
          "too")


if __name__ == "__main__":
    main()
