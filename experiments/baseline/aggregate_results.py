# -*- coding: utf-8 -*-
"""
Aggregate AnomalyDINO baseline results into tables.

Scans   results_MVTec/dinov2_vits14_448/<shot>-shot_preprocess=agnostic/metrics_seed=<s>.json
Writes  E:/work/freshman/results/metrics/baseline_summary.csv   (mean +/- std over seeds)
        E:/work/freshman/results/metrics/baseline_per_category.csv
and prints a markdown table.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

RESULTS_ROOT = r"E:\work\freshman\third_party\AnomalyDINO\results_MVTec\dinov2_vits14_448"
OUT_DIR = r"E:\work\freshman\results\metrics"
os.makedirs(OUT_DIR, exist_ok=True)

OBJECTS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
           "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

GLOBAL_KEYS = {
    "mean_classification_au_roc": "img_AUROC",
    "mean_classification_ap": "img_AP",
    "mean_classification_f1": "img_F1",
    "mean_segmentation_au_roc": "px_AUROC",
    "mean_segmentation_f1": "px_F1",
    "mean_au_pro": "AUPRO",
}
PER_OBJ_KEYS = {
    "classification_AUROC": "img_AUROC",
    "classification_AP": "img_AP",
    "classification_F1": "img_F1",
    "seg_AUROC": "px_AUROC",
    "seg_F1": "px_F1",
    "seg_AUPRO": "AUPRO",
}

# ------------------------------------------------------------------ collect
records = []
per_obj_rows = []
for shot_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*-shot_*"))):
    shot = int(os.path.basename(shot_dir).split("-shot")[0])
    for mf in sorted(glob.glob(os.path.join(shot_dir, "metrics_seed=*.json"))):
        seed = int(os.path.basename(mf).replace("metrics_seed=", "").replace(".json", ""))
        with open(mf) as f:
            m = json.load(f)
        row = {"shot": shot, "seed": seed}
        for k_json, k_out in GLOBAL_KEYS.items():
            row[k_out] = m.get(k_json, np.nan) * 100
        # merge pixel-level metrics from eval_pixel.py if present
        px_path = os.path.join(shot_dir, f"pixel_metrics_seed={seed}.json")
        if os.path.exists(px_path):
            with open(px_path) as f:
                px = json.load(f)
            row["px_AUROC"] = px["mean_px_AUROC"] * 100
            row["px_AP"] = px["mean_px_AP"] * 100
            row["px_F1"] = px["mean_px_F1"] * 100
            row["AUPRO"] = px["mean_AUPRO"] * 100
            for obj in OBJECTS:
                if obj in m and obj in px:
                    orow = {"shot": shot, "seed": seed, "object": obj}
                    for k_json, k_out in PER_OBJ_KEYS.items():
                        if k_out in ("img_AUROC", "img_AP", "img_F1"):
                            orow[k_out] = m[obj].get(k_json, np.nan) * 100
                        else:
                            orow[k_out] = px[obj][k_out] * 100
                    per_obj_rows.append(orow)
        else:
            for obj in OBJECTS:
                if obj in m:
                    orow = {"shot": shot, "seed": seed, "object": obj}
                    for k_json, k_out in PER_OBJ_KEYS.items():
                        orow[k_out] = m[obj].get(k_json, np.nan) * 100
                    per_obj_rows.append(orow)
        records.append(row)

df = pd.DataFrame(records).sort_values(["shot", "seed"])
df_obj = pd.DataFrame(per_obj_rows).sort_values(["shot", "object", "seed"])

# ------------------------------------------------------------------ summary
metric_cols = list(GLOBAL_KEYS.values())
grouped = df.groupby("shot")
summary = pd.DataFrame({
    "n_seeds": grouped.size(),
    **{f"{c}_mean": grouped[c].mean() for c in metric_cols},
    **{f"{c}_std": grouped[c].std(ddof=1) for c in metric_cols},
}).reset_index()

summary.to_csv(os.path.join(OUT_DIR, "baseline_summary.csv"), index=False, float_format="%.4f")
df.to_csv(os.path.join(OUT_DIR, "baseline_all_runs.csv"), index=False, float_format="%.4f")
df_obj.to_csv(os.path.join(OUT_DIR, "baseline_per_category.csv"), index=False, float_format="%.4f")

# ------------------------------------------------------------------ print
def fmt(mean, std):
    return f"{mean:5.2f} ± {std:4.2f}"

print("\n=== AnomalyDINO baseline on MVTec AD (DINOv2 ViT-S/14, res 448, agnostic) ===")
header = ["shot"] + metric_cols
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for _, r in summary.iterrows():
    cells = [str(int(r["shot"]))]
    for c in metric_cols:
        cells.append(fmt(r[f"{c}_mean"], r[f"{c}_std"]))
    print("| " + " | ".join(cells) + " |")

# per-category means for 1-shot (reference for later failure analysis)
print("\n=== Per-category results (1-shot, mean over seeds) ===")
sub = df_obj[df_obj["shot"] == 1].groupby("object")[metric_cols].mean()
print(sub.round(2).to_string())

print(f"\nSaved: {OUT_DIR}\\baseline_summary.csv, baseline_all_runs.csv, baseline_per_category.csv")
