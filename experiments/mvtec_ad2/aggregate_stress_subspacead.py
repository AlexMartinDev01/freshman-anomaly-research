# -*- coding: utf-8 -*-
"""
Aggregate the cross-method stress test (SubspaceAD) into report tables.

Reads every benchmark_results.csv produced by run_stress_subspacead.py under
results/mvtec_ad2/stress_subspacead*/ and writes:

  results/mvtec_ad2/metrics/stress_subspacead_runs.csv      one row per run
  results/mvtec_ad2/metrics/stress_subspacead_summary.csv   mean/std per (obj, shot)
  results/mvtec_ad2/metrics/cross_method_<cat set>.csv      AnomalyDINO vs SubspaceAD

Usage: python experiments/mvtec_ad2/aggregate_stress_subspacead.py
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from ad2_pipeline import RESULTS_ROOT

STRESS_GLOB = os.path.join(RESULTS_ROOT, "stress_subspacead*", "**",
                           "benchmark_results.csv")
OUT_ROOT = os.path.join(RESULTS_ROOT, "metrics")

# The four categories used for the cross-method stress test in both methods.
STRESS_OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
METRICS = ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO",
           "Image F1", "Pixel F1"]
# percent-scaled metric -> column name in the aggregate tables
COL = {
    "Image AUROC": "img_AUROC", "Image AUPR": "img_AP",
    "Pixel AUROC": "px_AUROC", "AU-PRO": "AUPRO@0.05",
    "Image F1": "img_F1", "Pixel F1": "px_F1",
}


def parse_run_path(path):
    """Recover (shot, seed, image_res, backbone) from the output tree.

    .../stress_subspacead_res448/k1_seed0/can/mvtec_ad2_mean_layers-12-..._res448_
       ..._model-dinov2-with-registers-giantpca_ev0.99_i-scoremtop1p_k1_aug30xr_seed0/
       benchmark_results.csv

    Only the res448 tree nests a per-object directory; the earlier res672 runs
    wrote one combined CSV per (shot, seed), so the object is read from the
    CSV's own Category column instead of the path.
    """
    parts = os.path.normpath(path).split(os.sep)
    combo = next((p for p in parts if re.fullmatch(r"k\d+_seed\d+", p)), None)
    if combo is None:
        return None
    run_dir = os.path.basename(os.path.dirname(path))
    res = re.search(r"_res(\d+)_", run_dir)
    model = re.search(r"_model-(.+?)pca_ev", run_dir)
    return {
        "shot": int(combo.split("_seed")[0][1:]),
        "seed": int(combo.split("_seed")[1]),
        "image_res": int(res.group(1)) if res else -1,
        "backbone": model.group(1) if model else "unknown",
    }


def read_run(path):
    """One benchmark_results.csv -> list of metric dicts (skips 'Average' rows)."""
    meta = parse_run_path(path)
    if meta is None:
        return []
    with open(path, newline="") as fh:
        parsed = list(csv.DictReader(fh))
    rows = []
    for r in parsed:
        cat = (r.get("Category") or "").strip()
        if not cat or cat == "Average":   # 'Average' = the combined summary row
            continue
        row = dict(meta)
        row["object"] = cat
        # res672 CSVs were written by pandas, so Seed is float-formatted ("0.0000")
        row["seed_in_csv"] = int(float(r["Seed"]))
        for m in METRICS:
            v = r.get(m)
            row[COL[m]] = float(v) * 100.0 if v not in (None, "") else np.nan
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    files = sorted(glob.glob(STRESS_GLOB, recursive=True))
    rows = [r for f in files for r in read_run(f)]
    if not rows:
        sys.exit("no benchmark_results.csv found - run the stress test first")
    df = pd.DataFrame(rows)

    base_cols = ["object", "shot", "seed", "image_res", "backbone"]
    metric_cols = [COL[m] for m in METRICS]
    df = df[base_cols + metric_cols].sort_values(
        ["image_res", "shot", "object", "seed"]).reset_index(drop=True)

    runs_path = os.path.join(OUT_ROOT, "stress_subspacead_runs.csv")
    df.to_csv(runs_path, index=False)

    # ---- summary: mean +/- std per (image_res, object, shot) ----
    g = df.groupby(["image_res", "backbone", "object", "shot"], dropna=False)
    summ = g[metric_cols].agg(["mean", "std"]).round(2)
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ["n_seeds"] = g.size()
    summ = summ.reset_index()
    summ_path = os.path.join(OUT_ROOT, "stress_subspacead_summary.csv")
    summ.to_csv(summ_path, index=False)

    print(f"runs    -> {runs_path}   ({len(df)} runs from {len(files)} files)")
    print(f"summary -> {summ_path}")
    print("\n=== SubspaceAD, img AUROC (mean over seeds) ===")
    for res in sorted(df.image_res.unique()):
        piv = summ[summ.image_res == res].pivot_table(
            index="shot", columns="object", values="img_AUROC_mean")
        cols = [c for c in STRESS_OBJECTS if c in piv.columns]
        if not cols:
            continue
        piv = piv[cols]
        piv["4cat_mean"] = piv.mean(axis=1)
        print(f"\n-- image_res={res} --")
        print(piv.round(1).to_string())

    # ---- cross-method table vs AnomalyDINO ----
    ad_csv = os.path.join(OUT_ROOT, "ad2_public_summary.csv")
    if not os.path.exists(ad_csv):
        print("\n(ad2_public_summary.csv not found - skipping cross-method table)")
        return
    ad = pd.read_csv(ad_csv)
    ad = ad[ad.object.isin(STRESS_OBJECTS) & (ad.shot > 0)].copy()
    ad["img_AUROC"] = ad["img_AUROC"]        # already percent-scaled
    ad["AUPRO@0.05"] = ad["AUPRO@0.05"]
    ad["px_AUROC"] = ad["px_AUROC"]
    ad = ad.groupby(["object", "shot"]).agg(
        n_seeds=("img_AUROC", "size"),
        AnomalyDINO_img_AUROC=("img_AUROC", "mean"),
        AnomalyDINO_px_AUROC=("px_AUROC", "mean"),
        AnomalyDINO_AUPRO=("AUPRO@0.05", "mean")).round(2).reset_index()

    # match against the resolution whose comparison is apples-to-apples:
    # AnomalyDINO ran at 448, so use the res448 (or the smallest available) run.
    res_use = 448 if 448 in set(df.image_res) else int(df.image_res.min())
    ss = summ[(summ.image_res == res_use)][
        ["object", "shot", "img_AUROC_mean", "px_AUROC_mean",
         "AUPRO@0.05_mean"]].round(2)
    ss = ss.rename(columns={
        "img_AUROC_mean": "SubspaceAD_img_AUROC",
        "px_AUROC_mean": "SubspaceAD_px_AUROC",
        "AUPRO@0.05_mean": "SubspaceAD_AUPRO"})

    merged = pd.merge(ad, ss, on=["object", "shot"], how="outer")
    merged["delta_img_AUROC"] = (merged.SubspaceAD_img_AUROC
                                 - merged.AnomalyDINO_img_AUROC).round(1)
    merged = merged.sort_values(["shot", "object"])
    cmp_path = os.path.join(OUT_ROOT, f"cross_method_res{res_use}.csv")
    merged.to_csv(cmp_path, index=False)

    print(f"\ncross-method -> {cmp_path}  (SubspaceAD @ res{res_use})")
    print("\n=== Cross-method: SubspaceAD vs AnomalyDINO, img AUROC ===")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
