# -*- coding: utf-8 -*-
"""
Evaluate AnomalyDINO on MVTec AD 2 TEST_pub (local evaluation, GT public).

Per (shot, seed, object):
  image-level : AUROC, AP, F1 @ t_img (t_img = mu + 3*sigma of validation scores)
  pixel-level : px_AUROC, px_AP, AU-PRO@0.05 (AD2 official), px_F1 @ t_px
                (t_px = mu + 3*sigma over all validation anomaly-map pixels)

Writes:
  results/mvtec_ad2/metrics/ad2_pub_<shot>shot_seed=<s>.json   (per object)
  results/mvtec_ad2/metrics/ad2_public_summary.csv             (all runs)

Usage: python experiments/mvtec_ad2/eval_ad2_public.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")

from pixel_metrics_binned import pixel_metrics_binned, pixel_f1_at_threshold
from ad2_pipeline import AD2_OBJECTS, AD2_ROOT, RESULTS_ROOT

METRICS_DIR = os.path.join(RESULTS_ROOT, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


def eval_object(obj, shot, seed):
    run = f"{shot}-shot_seed={seed}"
    csv_path = os.path.join(RESULTS_ROOT, "anomaly_scores", f"{obj}_{run}.csv")
    df = pd.read_csv(csv_path)
    val = df[df.split == "validation"]["score"].to_numpy()
    pub = df[df.split == "test_public"]
    y_true = (pub["type"] == "bad").astype(int).to_numpy()
    y_score = pub["score"].to_numpy()

    img_auroc = roc_auc_score(y_true, y_score)
    img_ap = average_precision_score(y_true, y_score)
    t_img = val.mean() + 3 * val.std()
    img_f1 = f1_score(y_true, (y_score > t_img).astype(int))

    # ---- pixel threshold from validation anomaly maps ----
    val_map_dir = os.path.join(RESULTS_ROOT, "anomaly_maps", run, obj,
                               "validation", "good")
    val_vals = []
    for f in sorted(os.listdir(val_map_dir)):
        val_vals.append(np.load(os.path.join(val_map_dir, f)).ravel())
    val_vals = np.concatenate(val_vals)
    t_px = val_vals.mean() + 3 * val_vals.std()
    del val_vals

    # ---- pixel metrics on test_public (histogram-based, full resolution) ----
    # Uses pixel_metrics_binned: O(1) memory, validated to match the exact
    # official computation (bottle diff = 0.0000; AD2 can diff <= 0.004).
    maps_dir = os.path.join(RESULTS_ROOT, "anomaly_maps", run, obj, "test_public")
    jobs = []
    for _, r in pub.iterrows():
        npy = os.path.join(maps_dir, r["type"], r["name"][:-4] + ".npy")
        gt = None
        if r["type"] == "bad":
            gt = os.path.join(AD2_ROOT, obj, "test_public", "ground_truth",
                              "bad", r["name"][:-4] + "_mask.png")
        jobs.append((npy, gt, (int(r["img_h"]), int(r["img_w"]))))
    m = pixel_metrics_binned(jobs, pro_limit=0.05)
    px_f1 = pixel_f1_at_threshold(jobs, t_px)

    return {
        "object": obj, "shot": shot, "seed": seed,
        "img_AUROC": img_auroc * 100, "img_AP": img_ap * 100,
        "img_F1@mu3sig": img_f1 * 100, "t_img": t_img,
        "val_score_mean": float(val.mean()), "val_score_std": float(val.std()),
        "pub_good_mean": float(pub[pub.type == 'good']["score"].mean()),
        "pub_bad_mean": float(pub[pub.type == 'bad']["score"].mean()),
        "px_AUROC": m["px_AUROC"] * 100, "px_AP": m["px_AP"] * 100,
        "AUPRO@0.05": m["AUPRO"] * 100, "px_F1@mu3sig": px_f1 * 100,
        "t_px": float(t_px),
    }


def main():
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(
            RESULTS_ROOT, "anomaly_scores", "*_*.csv"))):
        base = os.path.basename(csv_path)[:-4]
        # filename: <obj>_<shot>-shot_seed=<s>   (obj may itself contain '_')
        obj = next(o for o in AD2_OBJECTS if base.startswith(o + "_"))
        rest = base[len(obj) + 1:]
        shot = int(rest.split("-shot")[0])
        seed = int(rest.split("seed=")[1])
        out_json = os.path.join(METRICS_DIR, f"ad2_pub_{shot}shot_seed={seed}.json")
        existing = {}
        if os.path.exists(out_json):
            with open(out_json) as f:
                existing = json.load(f)
        if obj in existing:
            print(f"skip {obj} {shot}-shot seed={seed}")
            rows.extend(existing.values())
            continue
        print(f"eval {obj} | {shot}-shot seed={seed}", flush=True)
        r = eval_object(obj, shot, seed)
        existing[obj] = r
        with open(out_json, "w") as f:
            json.dump(existing, f, indent=2)
        rows.append(r)
        print(f"   img_AUROC={r['img_AUROC']:.2f} img_F1={r['img_F1@mu3sig']:.2f} "
              f"px_AUROC={r['px_AUROC']:.2f} AUPRO@0.05={r['AUPRO@0.05']:.2f} "
              f"px_F1={r['px_F1@mu3sig']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "ad2_public_summary.csv"), index=False)
    cols = ["img_AUROC", "img_AP", "img_F1@mu3sig", "px_AUROC", "px_AP",
            "AUPRO@0.05", "px_F1@mu3sig"]
    agg = df.groupby("shot")[cols].agg(["mean", "std"]).round(2)
    print("\n=== AD2 TEST_pub summary (mean ± std over objects*seeds) ===")
    print(agg.to_string())


if __name__ == "__main__":
    main()
