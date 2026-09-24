# -*- coding: utf-8 -*-
"""
Gate 5D verdict: how much of the real defect DISTRIBUTION must survive?

All six configs run under the SAME soft separation loss, so a cloud that
naturally lands far from the bank still receives gradient and a null result
means "this information does not help", not "the constraint never engaged".

    K0  E              full real defect set          oracle ceiling
    K1  R40            mu_d + real rank-40 residual  compressed ceiling
    K2  centroid       mu_d replicated               is the anchor alone enough?
    K3  Gaussian       mu_d + U40 x N(0, Sigma_c)    first + second order?
    K4  bootstrap      mu_d + U40 x resampled real c is the empirical DISTRIBUTION
                                                     enough without the instances?
    K5  GMM-4          mu_d + U40 x GMM(4)           is multimodality the key?

TWO CAVEATS that must travel with the numbers:

  (a) The soft loss is NOT a neutral measurement fix. softplus >= relu, so at
      lam_defect=1 it applies more separation pressure than the hinge, and it
      visibly inflates the normal tail (sheet_metal p99 0.104 -> 0.333 for K0).
      Gate 5D's internal K0..K5 ranking is therefore valid, but K0/K1 cannot be
      compared against the earlier hard-hinge E and R40.

  (b) Because of that inflation the pre-registered "normal_p99 must fall"
      criterion fails for most K rows -- a property of the loss, not of the
      defect-cloud choice. The informative comparison is the RELATIVE ranking
      on wallplugs (the primary hard object).

Usage: python analysis/model_v0/gate5d_verdict.py
"""
import sys

import pandas as pd

CSV = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
PRIMARY = "wallplugs"
ORDER = ["A_half", "K0", "K1", "K2", "K3", "K4", "K5"]
DESC = {"A_half": "baseline (frozen DINOv2)", "K0": "K0 E (full real)",
        "K1": "K1 R40 (real rank-40)", "K2": "K2 centroid only",
        "K3": "K3 Gaussian (mean+cov)", "K4": "K4 bootstrap (empirical)",
        "K5": "K5 GMM-4"}


def main():
    df = pd.read_csv(CSV)
    if not any(c in set(df.config) for c in ("K2", "K3", "K4", "K5")):
        sys.exit("no Gate 5D configs - run run_oracle.py --configs "
                 "K0 K1 K2 K3 K4 K5 --eval-half")
    print("=" * 100)
    print("GATE 5D -- all configs under the SAME soft separation loss")
    print("=" * 100)
    for o in ["wallplugs", "sheet_metal", "vial", "can"]:
        tag = ("PRIMARY" if o == PRIMARY else
               "easy/positive control" if o == "sheet_metal" else
               "do-no-harm gate" if o == "vial" else "record only")
        print(f"\n--- {o}  ({tag}) ---")
        print(f"  {'config':<30}{'img_AUROC':>11}{'AUPRO':>8}{'normal_p99':>12}"
              f"{'defect_mean':>13}{'defect<p99':>12}")
        for c in ORDER:
            r = df[(df.object == o) & (df.config == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {DESC.get(c, c):<30}{r.img_AUROC:>11.1f}"
                  f"{r['AUPRO@0.05']:>8.1f}{r.normal_p99:>12.3f}"
                  f"{r.defect_mean:>13.3f}{r.frac_defect_below_p99:>12.3f}")

    print("\n" + "=" * 100)
    print(f"RANKING ON {PRIMARY} (the primary hard object)")
    print("=" * 100)
    rows = []
    for c in ["K0", "K1", "K2", "K3", "K4", "K5"]:
        r = df[(df.object == PRIMARY) & (df.config == c)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append((c, r.img_AUROC, r["AUPRO@0.05"]))
    for c, i, a in sorted(rows, key=lambda t: -t[1]):
        print(f"  {DESC[c]:<30} img {i:5.1f}   AU-PRO {a:5.1f}")

    print("\n" + "=" * 100)
    print("DECISION")
    print("=" * 100)
    print("  K4 (bootstrap) matching K0/K1 means the empirical DISTRIBUTION is")
    print("  enough and per-instance correspondence is NOT needed.")
    print("  K3 (Gaussian) clearly below K4 means mean + covariance is insufficient.")
    print("  K5 (GMM-4) between them means 4 modes capture only part of it.")
    print("  K2 (centroid) failing means the anchor alone is not enough.")
    print("\n  CAVEAT (b): the soft loss inflates the normal tail, and the configs")
    print("  that win on wallplugs are the ones that inflate sheet_metal's tail and")
    print("  destroy its AU-PRO. So the information question is answered, but this")
    print("  loss is not usable as-is.")


if __name__ == "__main__":
    main()
