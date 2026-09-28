# -*- coding: utf-8 -*-
"""
Phase F1 (analysis) -- which factor explains the SHARED failure of both detectors?

Reads the per-defect rows from phase_f1_failure.py and, for each detector
(kNN agg_all3 and SubspaceAD aug=0) separately, asks whether a factor predicts
low-FPR localisation failure.

Pooled correlation is NOT enough here, and that is the whole point: the previous
round's rho=-0.526 turned out to be between-object confounding (hard objects have
high normal tails AND low AUPRO at every layer). So every factor is judged on
three levels:

  1. global      Spearman over all rows
  2. within-obj  z-score each variable inside its object, then Spearman --
                 this removes "hard object" as an explanation
  3. per-object  sign agreement of the within-object correlation across objects,
                 so one object cannot carry the result
  4. regression  OLS with object fixed effects (within-object demeaning):
                    TPR ~ C_feat + log10(area_frac) + frag + shape

Pre-registered verdict (frozen before the run, see phase_f1_failure.py):

  A  C_feat drives both: within-object rho(C_feat, TPR) <= -0.4 for BOTH
     detectors AND survives object FE
  B  morphology drives both: within-object rho(area, TPR) >= +0.4 OR
     rho(frag, TPR) <= -0.4 for BOTH, surviving object FE
  C  neither -> the data does not identify a mechanism; stop inventing models.

Usage: python experiments/model_v0/phase_f1_analysis.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")


def ols_fe(df, y, xs):
    """OLS on within-object demeaned variables == object fixed effects."""
    d = df.copy()
    d["log_area"] = np.log10(d["area_frac"].clip(lower=1e-6))
    cols = xs + [y]
    dm = d.groupby("object")[cols].transform(lambda v: v - v.mean())
    A = np.column_stack([dm[c].values for c in xs] + [np.ones(len(dm))])
    b, *_ = np.linalg.lstsq(A, dm[y].values, rcond=None)
    resid = dm[y].values - A @ b
    dof = len(dm) - A.shape[1]
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    return {c: (b[i], b[i] / se[i]) for i, c in enumerate(xs)}


def analyse(df, det, factors):
    y = f"TPR_{det}"
    out = {}
    for f in factors:
        sub = df[[f, y, "object"]].dropna()
        rho_g, p_g = spearmanr(sub[f], sub[y])
        z = sub.copy()
        for c in (f, y):
            z[c] = z.groupby("object")[c].transform(
                lambda v: (v - v.mean()) / (v.std() + 1e-12))
        rho_w, p_w = spearmanr(z[f], z[y])
        signs = []
        for _, g in sub.groupby("object"):
            if len(g) >= 8 and g[f].std() > 0 and g[y].std() > 0:
                r, _ = spearmanr(g[f], g[y])
                if not np.isnan(r):
                    signs.append(r)
        agree = (np.mean([s < 0 for s in signs]) if np.mean(signs) < 0
                 else np.mean([s > 0 for s in signs])) if signs else float("nan")
        out[f] = {"global_rho": rho_g, "global_p": p_g,
                  "within_rho": rho_w, "within_p": p_w,
                  "per_obj_agree": agree, "n_obj": len(signs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phase_f1_failure")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(METRICS, f"{a.tag}.csv"))
    df = df[df["TPR_knn"].notna() & df["TPR_sa"].notna()]
    print(f"rows {len(df)}  objects {df.object.nunique()}  "
          f"distinct defect images {df.name.nunique()}")

    factors = ["C_feat", "area_frac", "n_cc", "largest_cc_frac",
               "perimeter2_over_area", "elongation"]
    print("\n" + "=" * 100)
    print("FACTOR -> LOW-FPR LOCALISATION (TPR@FPR=0.05), per detector")
    print("=" * 100)
    res = {d: analyse(df, d, factors) for d in ("knn", "sa")}
    print(f"\n{'factor':<22}{'det':<5}{'global rho':>12}{'within rho':>12}"
          f"{'p(within)':>12}{'per-obj agree':>15}")
    for f in factors:
        for d in ("knn", "sa"):
            r = res[d][f]
            print(f"{f:<22}{d:<5}{r['global_rho']:>12.3f}"
                  f"{r['within_rho']:>12.3f}{r['within_p']:>12.2e}"
                  f"{r['per_obj_agree']:>15.2f}")

    print("\n" + "=" * 100)
    print("OBJECT FIXED-EFFECTS REGRESSION   TPR ~ C_feat + log10(area) + "
          "fragmentation + shape")
    print("=" * 100)
    frags = {"largest_cc_frac": "largest_cc_frac", "n_cc": "n_cc",
             "perimeter2_over_area": "perimeter2_over_area"}
    for d in ("knn", "sa"):
        print(f"\n  detector = {d}")
        for fname, fcol in frags.items():
            co = ols_fe(df, f"TPR_{d}", ["C_feat", "log_area", fcol,
                                         "elongation"])
            s = "  ".join(f"{k}={v[0]:+.4f}(t={v[1]:+.1f})"
                          for k, v in co.items())
            print(f"    frag={fname:<22} {s}")

    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT")
    print("=" * 100)
    cf = {d: res[d]["C_feat"]["within_rho"] for d in ("knn", "sa")}
    ar = {d: res[d]["area_frac"]["within_rho"] for d in ("knn", "sa")}
    fr = {d: res[d]["largest_cc_frac"]["within_rho"] for d in ("knn", "sa")}
    A = all(cf[d] <= -0.4 for d in cf)
    B = all(ar[d] >= 0.4 for d in ar) or all(fr[d] <= -0.4 for d in fr)
    print(f"  within-object rho(C_feat, TPR): kNN {cf['knn']:+.3f}   "
          f"SA {cf['sa']:+.3f}   -> A {'HOLDS' if A else 'fails'}")
    print(f"  within-object rho(area,   TPR): kNN {ar['knn']:+.3f}   "
          f"SA {ar['sa']:+.3f}")
    print(f"  within-object rho(largest,TPR): kNN {fr['knn']:+.3f}   "
          f"SA {fr['sa']:+.3f}   -> B {'HOLDS' if B else 'fails'}")
    if A:
        print("\n  -> CASE A: low-contrast CONTEXTUAL localisation is the "
              "shared driver")
    elif B:
        print("\n  -> CASE B: SPARSE/DIFFUSE spatial aggregation is the "
              "shared driver")
    else:
        print("\n  -> CASE C: no single factor explains both detectors; "
              "do not invent a model on this evidence")


if __name__ == "__main__":
    main()
