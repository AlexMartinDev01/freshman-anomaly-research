# -*- coding: utf-8 -*-
"""
Evaluate an already-finished AnomalyDINO run (Windows-safe).

Applies the read_tiff monkeypatch (case-insensitive FS fix) and evaluates
one shot directory / one seed, writing metrics_seed=<seed>.json next to it.

Usage:
    python experiments/baseline/eval_only.py <shot> <seed>
Example:
    python experiments/baseline/eval_only.py 1 0
"""
import json
import os
import sys

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, AD_ROOT)
os.chdir(AD_ROOT)

import tifffile as tiff
import src.post_eval as post_eval


def read_tiff_windows(file_path_no_ext, exts=(".tiff",)):
    for ext in exts:
        f = file_path_no_ext + ext
        if os.path.exists(f):
            return tiff.imread(f)
    raise FileNotFoundError(file_path_no_ext)


post_eval.read_tiff = read_tiff_windows

if __name__ == "__main__":
    shot = int(sys.argv[1])
    seed = int(sys.argv[2])

    results_dir = os.path.join(
        "results_MVTec", "dinov2_vits14_448", f"{shot}-shot_preprocess=agnostic")

    post_eval.eval_finished_run(
        "MVTec",
        r"E:\work\freshman\data\mvtec_anomaly_detection",
        anomaly_maps_dir=os.path.join(results_dir, "anomaly_maps", f"seed={seed}"),
        output_dir=results_dir,
        seed=seed,
        pro_integration_limit=0.3,
        eval_clf=True,
        eval_segm=False,  # pixel metrics are computed by eval_pixel.py (memory-safe)
    )

    metric_path = os.path.join(results_dir, f"metrics_seed={seed}.json")
    with open(metric_path) as f:
        m = json.load(f)
    print(f"{shot}-shot seed={seed}  "
          f"img_AUROC={m['mean_classification_au_roc']*100:.2f}  "
          f"img_AP={m['mean_classification_ap']*100:.2f}  "
          f"img_F1={m['mean_classification_f1']*100:.2f}")
