# -*- coding: utf-8 -*-
"""
Phase M1-mechanism (analysis) -- criterion C3 and the mechanism directions.

Criterion C3 (docs/PHASE_M1_CONFIRM_PREREGISTRATION.md):

    within-object  Spearman( C_feat_448 , dAUPRO ) < 0
    i.e. defects that are LESS separable at 448 gain MORE from 672

and section 4 asks for the same against the defect's scale on the 448 grid.

UNIT OF ANALYSIS.  One defect INSTANCE (an anomalous test image), averaged
over the (shot, split) cells it was measured in, so that bank-draw noise does
not masquerade as between-defect variance.  "Within-object" means the
correlation is computed inside each object across its own defect instances;
objects with fewer than MIN_N instances are excluded from the per-object
average (their rho is unstable) but are still counted in the pooled test.

TWO ESTIMATORS, reported together on purpose: a Fisher-z average of the
per-object Spearmans, and a within-object (object fixed-effects) regression
with object-clustered SE -- the same estimator F1/F2 used, and the same one
that cut F1's naive t from +44 to +4.8.  A mechanism claim should survive
both.

CAVEAT CARRIED FROM phase_m1_mechanism.py: the per-image AUPRO used here
re-normalises FPR inside each image, so it does NOT average to the headline
AUPRO, which shares one FPR axis across a cell.  It is an effect size per
defect, not "the AUPRO".

Usage: python experiments/model_v0/phase_m1_mechanism_analysis.py
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
MIN_N = 5          # min defect instances for a per-object Spearman to count


def ols_fe_cluster(df, y, xs, cluster="object"):
    """Within-cluster (FE) OLS with cluster-robust SE -- same estimator as
    phase_f2_analysis.ols_cluster, minus its hard-coded derived columns.
    No intercept: demeaning absorbs it."""
    d = df[[y] + xs + [cluster]].dropna().copy()
    for c in [y] + xs:
        d[c] = d.groupby(cluster)[c].transform(lambda v: v - v.mean())
    X = np.column_stack([d[c].values for c in xs])
    yv = d[y].values
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    u = yv - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    groups = d[cluster].values
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = 0
    for g in pd.unique(groups):
        m = groups == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
        G += 1
    n, k = X.shape
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv * corr))
    out = {}
    for i, c in enumerate(xs):
        tv = b[i] / se[i]
        out[c] = {"beta": b[i], "se": se[i], "t": tv,
                  "p": 2 * (1 - tdist.cdf(abs(tv), df=G - 1))}
    return out, G


def fisher_mean(rs):
    z = np.arctanh(np.clip(np.asarray(rs, dtype=float), -0.999, 0.999))
    return float(np.tanh(z.mean())), float(z.std(ddof=1) / np.sqrt(len(z))) if len(z) > 1 else np.nan


def per_object_rho(df, f, y):
    """Spearman inside each object, over its defect instances."""
    rs, objs = [], []
    for o, g in df.groupby("object"):
        g = g[[f, y]].dropna()
        if len(g) < MIN_N or g[f].std() == 0 or g[y].std() == 0:
            continue
        r, _ = spearmanr(g[f], g[y])
        if not np.isnan(r):
            rs.append(r)
            objs.append(o)
    return np.array(rs), objs


def report(df, f, y, label):
    rs, objs = per_object_rho(df, f, y)
    fm, fse = fisher_mean(rs)
    neg = float((rs < 0).mean()) if len(rs) else float("nan")
    print(f"\n  {label}")
    print(f"    per-object rho: n={len(rs)} objects (>= {MIN_N} instances), "
          f"mean {rs.mean():+.3f}, median {np.median(rs):+.3f}, "
          f"negative {int((rs < 0).sum())}/{len(rs)} ({neg * 100:.0f}%)")
    print(f"    Fisher-z mean rho {fm:+.3f}  (SE {fse:.3f})")
    for o, r in sorted(zip(objs, rs), key=lambda t: t[1]):
        print(f"      {o:<14} {r:+.3f}")
    return fm, rs, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="mech_a,mech_b,mech_c")
    a = ap.parse_args()
    parts = []
    for t in a.tags.split(","):
        p = os.path.join(METRICS, f"{t}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p)
        print(f"  {t:<10} {len(d):>6} rows, {d.object.nunique():>2} objects")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # A shard rewrites its whole CSV after every object, so a crash can leave a
    # duplicated object.  A duplicate here would double-weight that object in
    # every pooled statistic below.
    n0 = len(df)
    df = df.drop_duplicates(subset=["object", "name", "shot", "split"],
                            keep="last")
    if len(df) != n0:
        print(f"  note: dropped {n0 - len(df)} duplicate measurement rows")
    n_raw = len(df)
    # average each defect instance over its cells so bank-draw noise is not
    # mistaken for between-defect variance
    keys = ["object", "name"]
    inst = df.groupby(keys).agg(
        C_feat448=("C_feat448", "mean"), sep_auc448=("sep_auc448", "mean"),
        d_aupro=("d_aupro", "mean"), d_tpr=("d_tpr", "mean"),
        gt_patch_n=("gt_patch_n", "mean"), gt_patch_area=("gt_patch_area", "mean"),
        n_gt_px=("n_gt_px", "mean"), cells=("d_aupro", "size")).reset_index()
    inst["log_area"] = np.log10(inst["gt_patch_area"].clip(lower=1e-3))
    print(f"  {n_raw} measurement rows -> {len(inst)} defect instances over "
          f"{inst.object.nunique()} objects")
    print(f"  cells per instance: {dict(inst.cells.value_counts())}")

    print("\n" + "=" * 100)
    print("C3  within-object Spearman( C_feat_448 , dAUPRO ) < 0 ?")
    print("=" * 100)
    fm_c, rs_c, neg_c = report(inst, "C_feat448", "d_aupro",
                               "PRIMARY (preregistered): C_feat_448 vs dAUPRO")
    co, G = ols_fe_cluster(inst, "d_aupro", ["C_feat448"])
    c = co["C_feat448"]
    print(f"    pooled within-object FE regression, object-clustered SE "
          f"(G={G}):")
    print(f"      beta {c['beta']:+.4f}  SE {c['se']:.4f}  t {c['t']:+.2f}  "
          f"p {c['p']:.2e}")
    c3 = (fm_c < 0) and (c["beta"] < 0)
    print(f"    -> Fisher-z mean {fm_c:+.3f} (<0) and FE beta "
          f"{c['beta']:+.4f} (<0):  {'OK' if c3 else 'FAIL'}")

    print("\n  SECONDARY (not preregistered -- sep_auc is this project's "
          "scale-free statistic)")
    report(inst, "sep_auc448", "d_aupro", "sep_auc_448 vs dAUPRO")
    co2, _ = ols_fe_cluster(inst, "d_aupro", ["sep_auc448"])
    print(f"    FE beta {co2['sep_auc448']['beta']:+.4f}  "
          f"t {co2['sep_auc448']['t']:+.2f}  p {co2['sep_auc448']['p']:.2e}")

    print("\n  dTPR@5% instead of dAUPRO (the other metric section 4 allows) -- "
          "robustness only")
    report(inst, "C_feat448", "d_tpr", "C_feat_448 vs dTPR@5%")
    co3, _ = ols_fe_cluster(inst, "d_tpr", ["C_feat448"])
    print(f"    FE beta {co3['C_feat448']['beta']:+.6f}  "
          f"t {co3['C_feat448']['t']:+.2f}  p {co3['C_feat448']['p']:.2e}")

    print("\n" + "=" * 100)
    print("M1-mechanism section 4: defect SCALE on the 448 grid vs the gain")
    print("=" * 100)
    print("  expectation: anomalies small on the 448 grid gain more from 672")
    for f in ("gt_patch_n", "log_area"):
        report(inst, f, "d_aupro", f"{f} vs dAUPRO")
        co4, _ = ols_fe_cluster(inst, "d_aupro", [f])
        print(f"    FE beta {co4[f]['beta']:+.4f}  t {co4[f]['t']:+.2f}  "
              f"p {co4[f]['p']:.2e}")

    print("\n" + "=" * 100)
    print("JOINT: does separability survive controlling for defect size?")
    print("=" * 100)
    co5, G5 = ols_fe_cluster(inst, "d_aupro", ["C_feat448", "log_area",
                                               "gt_patch_n"])
    for k, v in co5.items():
        print(f"    {k:<14} beta {v['beta']:+.4f}  SE {v['se']:.4f}  "
              f"t {v['t']:+.2f}  p {v['p']:.2e}")
    print("    (C_feat_448 and log_area are collinear by construction -- a "
          "defect confined to a few patches is both small and poorly "
          "separated -- so read the two coefficients jointly, not separately)")

    print("\n" + "=" * 100)
    print("DESCRIPTIVE: cross-object pooled correlation")
    print("=" * 100)
    r, p = spearmanr(inst["C_feat448"], inst["d_aupro"])
    print(f"  pooled Spearman(C_feat_448, dAUPRO) = {r:+.3f}  p={p:.2e}")
    print("  This one mixes between-object level differences with the within-"
          "object mechanism; it is NOT criterion C3.")

    print("\n" + "=" * 100)
    print("C3 VERDICT")
    print("=" * 100)
    print(f"  within-object Fisher-z rho {fm_c:+.3f}  "
          f"({int((rs_c < 0).sum())}/{len(rs_c)} objects negative)")
    print(f"  FE beta {c['beta']:+.4f} (t {c['t']:+.2f})")
    print(f"  -> C3 {'PASS' if c3 else 'FAIL'}")
    print("  C1/C2 come from phase_m1_confirm_analysis.py; C4 from "
          "phase_m1_dose.py. The pre-registration requires ALL of C1-C4.")


if __name__ == "__main__":
    main()
