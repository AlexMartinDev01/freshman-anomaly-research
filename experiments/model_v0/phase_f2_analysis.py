# -*- coding: utf-8 -*-
"""
Phase F2 (analysis) -- evaluate the frozen VisA replication criteria.

Criteria live in docs/PHASE_F2_PREREGISTRATION.md and are NOT restated or
adjusted here; this file only computes them.

The one methodological upgrade over the F1 analysis is F2-3's clustered SE.
F1 reported naive t-statistics on ~7.5k defect images that are not independent
(100 images share a category, a k-shot bank and a scene), which manufactures
false precision -- t=+44 is not 44 sigma. Here the variance is clustered by
category (G=12), with the usual finite-sample correction, and the number of
clusters is reported so the reader can judge how much the test can bear.

Usage: python experiments/model_v0/phase_f2_analysis.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")


def ols_cluster(df, y, xs, cluster="object"):
    """Within-category (FE) OLS with cluster-robust SE. No intercept: the
    demeaning absorbs it."""
    d = df.copy()
    d["log_area"] = np.log10(d["area_frac"].clip(lower=1e-6))
    cols = list(dict.fromkeys(xs + [y, cluster]))
    dm = d[cols].copy()
    for c in [y] + xs:
        dm[c] = dm.groupby(cluster)[c].transform(lambda v: v - v.mean())
    X = np.column_stack([dm[c].values for c in xs])
    yv = dm[y].values
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    groups = dm[cluster].values
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for g in pd.unique(groups):
        m = groups == g
        Xg, ug = X[m], u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
        G += 1
    n, k = X.shape
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtX_inv @ meat @ XtX_inv * corr
    se = np.sqrt(np.diag(V))
    out = {}
    for i, c in enumerate(xs):
        tv = b[i] / se[i]
        p = 2 * (1 - tdist.cdf(abs(tv), df=G - 1))
        out[c] = {"beta": b[i], "se": se[i], "t": tv, "p": p,
                  "lo": b[i] - 1.96 * se[i], "hi": b[i] + 1.96 * se[i]}
    return out, G


def within_rho(df, f, y, cluster="object"):
    """Pooled within-category Spearman: z-score both inside each category."""
    z = df[[f, y, cluster]].dropna().copy()
    for c in (f, y):
        z[c] = z.groupby(cluster)[c].transform(
            lambda v: (v - v.mean()) / (v.std() + 1e-12))
    return spearmanr(z[f], z[y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phase_f2_visa")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(METRICS, f"{a.tag}.csv"))
    df = df[df["TPR_knn"].notna() & df["TPR_sa"].notna()]
    G = df["object"].nunique()
    print(f"rows {len(df)}  categories {G}  "
          f"defect images {df.groupby(['object', 'name']).ngroups}")
    print(f"rows per category: {dict(df.groupby('object').size())}")

    print("\n" + "=" * 100)
    print("F2-1 / F2-2  within-category rho(C_feat, TPR) and direction agreement")
    print("=" * 100)
    res = {}
    for det in ("knn", "sa"):
        rho, p = within_rho(df, "C_feat", f"TPR_{det}")
        per = []
        for c, g in df.groupby("object"):
            if len(g) >= 8 and g["C_feat"].std() > 0:
                r, _ = spearmanr(g["C_feat"], g[f"TPR_{det}"])
                if not np.isnan(r):
                    per.append(r)
        agree = float(np.mean([r > 0 for r in per])) if per else float("nan")
        res[det] = {"rho": rho, "p": p, "agree": agree, "ncat": len(per)}
        print(f"  {det:<5} rho {rho:+.3f}  p={p:.2e}   "
              f"categories with positive rho: {int(agree * len(per))}/{len(per)} "
              f"({agree * 100:.0f}%)")

    print("\n" + "=" * 100)
    print(f"F2-3  C_feat coefficient with object FE and CATEGORY-CLUSTERED SE "
          f"(G={G} clusters)")
    print("=" * 100)
    for det in ("knn", "sa"):
        co, GG = ols_cluster(df, f"TPR_{det}",
                             ["C_feat", "log_area", "largest_cc_frac",
                              "n_cc", "elongation"])
        print(f"\n  detector = {det}   (clusters G={GG})")
        for k, v in co.items():
            print(f"    {k:<22} beta {v['beta']:+.4f}  SE {v['se']:.4f}  "
                  f"t {v['t']:+.2f}  p {v['p']:.2e}  "
                  f"95% CI [{v['lo']:+.4f}, {v['hi']:+.4f}]")

    print("\n" + "=" * 100)
    print("F2-4  same, on the continuous margin M instead of the binarised TPR")
    print("=" * 100)
    mres = {}
    for det in ("knn", "sa"):
        rho, p = within_rho(df, "C_feat", f"M_{det}")
        mres[det] = (rho, p)
        print(f"  {det:<5} within-category rho(C_feat, M_{det}) {rho:+.3f}  "
              f"p={p:.2e}")

    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT")
    print("=" * 100)
    f1 = all(res[d]["rho"] >= 0.4 for d in res)
    f2 = all(res[d]["agree"] >= 0.8 for d in res)
    f3 = []
    for det in ("knn", "sa"):
        co, _ = ols_cluster(df, f"TPR_{det}",
                            ["C_feat", "log_area", "largest_cc_frac", "n_cc",
                             "elongation"])
        c = co["C_feat"]
        f3.append(c["beta"] > 0 and c["p"] < 0.05 and c["lo"] > 0)
    f3 = all(f3)
    f4 = all(mres[d][0] >= 0.4 for d in mres)
    print(f"  F2-1 rho >= +0.4 for both detectors   "
          f"(kNN {res['knn']['rho']:+.3f}, SA {res['sa']['rho']:+.3f})  "
          f"-> {'OK' if f1 else 'FAIL'}")
    print(f"  F2-2 >= 80% categories positive       "
          f"(kNN {res['knn']['agree'] * 100:.0f}%, "
          f"SA {res['sa']['agree'] * 100:.0f}%)  -> {'OK' if f2 else 'FAIL'}")
    print(f"  F2-3 C_feat > 0, p < 0.05, clustered SE          "
          f"-> {'OK' if f3 else 'FAIL'}")
    print(f"  F2-4 M-based rho >= +0.4 for both     "
          f"(kNN {mres['knn'][0]:+.3f}, SA {mres['sa'][0]:+.3f})  "
          f"-> {'OK' if f4 else 'FAIL'}")
    ok = f1 and f2 and f3 and f4
    print(f"\n  -> {'REPLICATED: mechanism holds on VisA; proceed to Model V2 smoke test' if ok else 'NOT REPLICATED: stop, do not build a model'}")


if __name__ == "__main__":
    main()
