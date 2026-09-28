# -*- coding: utf-8 -*-
"""
Phase M1-confirm (analysis) -- merge the four shards and evaluate C1/C2.

Criteria are frozen in docs/PHASE_M1_CONFIRM_PREREGISTRATION.md section 2 and
are NOT restated, softened or adjusted here; this file only computes them.

    C1  mean dAUPRO > 0  and  >= 60% of object/cells positive
    C2  mean d_img_AUROC >= -0.3 and mean d_px_AUROC >= -0.3
    C3  within-object Spearman(C_feat_448, dAUPRO) < 0   -- in
        phase_m1_mechanism.py, which needs per-defect quantities
    C4  dose response 448 -> 560 -> 672                  -- in phase_m1_dose.py

The unit for C1/C2 is the (object, shot, split) cell, exactly as the
pre-registration writes it ("object/cell").  27 objects x 4 shots x 3 splits
= 324 cells, each carrying one r0 and one r1 row.

Everything this file prints beyond C1/C2 (cluster-robust SEs, per-object
tables, sign tests) is DESCRIPTIVE and is labelled as such -- it is not part
of the frozen criteria and must not be reported as if it were.

Usage: python experiments/model_v0/phase_m1_confirm_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
SHARDS = ["m1confirm_s1_mv_a", "m1confirm_s2_mv_b",
          "m1confirm_s3_vi_a", "m1confirm_s4_vi_b"]
KEY = ["object", "shot", "split"]
DELTAS = ["AUPRO", "img_AUROC", "px_AUROC", "sep_auc", "C_feat"]


def load():
    parts = []
    for s in SHARDS:
        p = os.path.join(METRICS, f"{s}.csv")
        if not os.path.exists(p):
            print(f"  !! missing {p}")
            return None
        d = pd.read_csv(p)
        d["shard"] = s
        print(f"  {s:<20} {len(d):>5} rows, {d.object.nunique():>2} objects")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # A shard writes its CSV after every object, so a crash can leave a
    # partially written object.  Duplicate (object, shot, split, repr) means
    # exactly that; keep the last write, which is the complete one.
    n0 = len(df)
    df = df.drop_duplicates(subset=KEY + ["repr"], keep="last")
    if len(df) != n0:
        print(f"  note: dropped {n0 - len(df)} duplicate rows (partial-object "
              f"re-writes from a crashed shard)")

    # The four object lists are disjoint by construction; a shared object would
    # mean two shards measured the same cell with different bank draws.
    dup = df.groupby("object")["shard"].nunique()
    bad = dup[dup > 1]
    if len(bad):
        raise RuntimeError(f"object measured by more than one shard: {list(bad.index)}")
    return df


def pivot(df):
    p = df.pivot_table(index=KEY, columns="repr",
                       values=[m for m in DELTAS if m in df.columns])
    missing = [r for r in ("r0", "r1") if r not in p.columns.get_level_values(1)]
    if missing:
        raise RuntimeError(f"missing representation(s): {missing}")
    return p


def cluster_se(delta, groups):
    """Cluster-robust SE of the mean, clustered on `groups` (objects).

    Same estimator as phase_f1_failure.ols_fe / phase_f2_analysis.ols_cluster:
    the naive SE over cells treats 12 correlated cells of one object as 12
    independent observations, which in F1 inflated t by ~9x.  Here it is used
    for a single mean, not a regression.
    """
    d = np.asarray(delta, dtype=float)
    g = np.asarray(groups)
    n = len(d)
    m = d.mean()
    meat = 0.0
    for gg in np.unique(g):
        s = (d[g == gg] - m).sum()
        meat += s * s
    G = len(np.unique(g))
    var = (G / (G - 1.0)) * meat / (n * n)
    return float(np.sqrt(var))


def main():
    print("=" * 100)
    print("PHASE M1-confirm -- merging shards")
    print("=" * 100)
    df = load()
    if df is None:
        return

    objs = sorted(df.object.unique())
    cells = df[df.repr == "r0"].groupby(KEY).ngroups
    print(f"\n  {len(objs)} objects, {cells} cells, {len(df)} rows")
    if len(objs) != 27:
        print(f"  !! expected 27 objects, found {len(objs)} -- judging anyway, "
              f"but the pre-registration covers the full 15 + 12")
    out = os.path.join(METRICS, "m1confirm_all.csv")
    df.to_csv(out, index=False)
    print(f"  -> {out}")

    piv = pivot(df)
    d = {m: (piv[(m, "r1")] - piv[(m, "r0")]) for m in DELTAS}
    dd = pd.DataFrame({m: v for m, v in d.items()})
    dd.index.names = KEY
    dd.to_csv(os.path.join(METRICS, "m1confirm_deltas.csv"))
    print(f"  -> {os.path.join(METRICS, 'm1confirm_deltas.csv')}")

    print("\n" + "=" * 100)
    print("PRE-REGISTERED CRITERIA C1 / C2   (unit = object/cell)")
    print("=" * 100)
    da = d["AUPRO"].dropna()
    frac_pos = float((da > 0).mean())
    c1 = (da.mean() > 0) and (frac_pos >= 0.60)
    print(f"\n  C1  mean dAUPRO = {da.mean():+.3f}   (need > 0)")
    print(f"      positive cells {int((da > 0).sum())}/{len(da)} = "
          f"{frac_pos * 100:.1f}%   (need >= 60%)")
    print(f"      -> {'OK' if c1 else 'FAIL'}")
    # "object/cell" is read here as the (object, shot, split) cell, because a
    # delta only exists per cell.  The object-level fraction is the natural
    # alternative reading; it is printed so the reader can check either
    # reading without having to rerun anything.
    per_obj = da.groupby(level=0).mean()
    print(f"      [alt reading] positive OBJECTS "
          f"{int((per_obj > 0).sum())}/{len(per_obj)} = "
          f"{float((per_obj > 0).mean()) * 100:.1f}%")

    # C2 is on the MEAN, per the pre-registration text.  The fraction of
    # positive cells is reported for context only.
    mi = d["img_AUROC"].dropna().mean()
    mp = d["px_AUROC"].dropna().mean()
    c2 = (mi >= -0.3) and (mp >= -0.3)
    print(f"\n  C2  mean d_img_AUROC = {mi:+.3f}   (need >= -0.3)")
    print(f"      mean d_px_AUROC  = {mp:+.3f}   (need >= -0.3)")
    print(f"      -> {'OK' if c2 else 'FAIL'}")

    print("\n" + "=" * 100)
    print("DESCRIPTIVE (not part of the frozen criteria)")
    print("=" * 100)
    for m in ["AUPRO", "px_AUROC", "img_AUROC", "sep_auc"]:
        v = d[m].dropna()
        g = v.index.get_level_values("object")
        se = cluster_se(v.values, g)
        print(f"  {m:<10} mean {v.mean():+8.3f}  naive SE "
              f"{v.std(ddof=1) / np.sqrt(len(v)):6.3f}  clustered SE "
              f"{se:6.3f}  t_clust {v.mean() / se:+.2f}   "
              f"(G={len(np.unique(g))} objects)")

    print("\n  per-object dAUPRO (mean over its 12 cells)")
    per = d["AUPRO"].groupby(level=0).mean().sort_values()
    for o, v in per.items():
        n = int((d["AUPRO"][o] > 0).sum())
        print(f"    {o:<14} {v:+8.3f}   {n}/12 cells positive"
              f"{'   <-- negative' if v < 0 else ''}")

    print("\n  per-object d_img_AUROC")
    for o, v in d["img_AUROC"].groupby(level=0).mean().sort_values().items():
        print(f"    {o:<14} {v:+8.3f}")

    print("\n  r1-vs-r0 absolute levels (mean over cells)")
    lv = piv.groupby(level=0).mean()
    for m in ["AUPRO", "px_AUROC", "img_AUROC", "sep_auc"]:
        print(f"    {m:<10} r0 {lv[(m, 'r0')].mean():8.3f}   "
              f"r1 {lv[(m, 'r1')].mean():8.3f}")

    print("\n" + "=" * 100)
    print("VERDICT SO FAR")
    print("=" * 100)
    print(f"  C1 {'PASS' if c1 else 'FAIL'}   C2 {'PASS' if c2 else 'FAIL'}"
          f"   C3 pending (phase_m1_mechanism.py)"
          f"   C4 pending (phase_m1_dose.py)")
    print("  The pre-registration requires ALL of C1-C4; C1/C2 alone do not"
          " authorise Model V3.")


if __name__ == "__main__":
    main()
