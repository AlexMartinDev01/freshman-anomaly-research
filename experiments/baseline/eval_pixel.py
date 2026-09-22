# -*- coding: utf-8 -*-
"""
Memory-efficient pixel-level evaluation for AnomalyDINO runs (MVTec AD).

Why this exists
---------------
The official eval_segm path (a) needs full-resolution .tiff anomaly maps
(~30 GB disk per seed) and (b) builds float64 arrays of shape
(n_images, H, W) in compute_pro, which peaks at ~8 GB for the largest
MVTec categories and gets OOM-killed on this 32 GB machine.

This script computes the *same* metrics directly from the saved 32x32 patch
distances (.npy), upsampling on the fly with the official dists2map recipe
(bilinear resize + Gaussian sigma=4), and replicates the official PRO-curve
computation with lighter dtypes. Aggregated metrics are independent of image
order, so sorted() file order is fine (official uses os.listdir order).

Metrics per category: pixel AUROC / AP / F1, AU-PRO (FPR limit 0.3).
Writes pixel_metrics_seed=<seed>.json next to the run's metrics files.

Usage:
    python experiments/baseline/eval_pixel.py            # all shots & seeds found
    python experiments/baseline/eval_pixel.py 1 0        # only 1-shot, seed 0
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
DATA_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
RESULTS_ROOT = os.path.join(AD_ROOT, "results_MVTec", "dinov2_vits14_448")

OBJECTS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
           "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
           "wood", "zipper"]
PRO_INTEGRATION_LIMIT = 0.3


def dists2map(dists, img_shape):
    """Identical to src.utils.dists2map (bilinear upsample + Gaussian sigma=4)."""
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(dists, sigma=4)


def compute_pixel_metrics(scores, gts, pro_limit=PRO_INTEGRATION_LIMIT):
    """scores: (n,H,W) float32, gts: (n,H,W) uint8 -> dict of 4 metrics.

    pro_limit: FPR integration limit for AU-PRO (0.3 for MVTec AD,
    0.05 for MVTec AD 2)."""
    n, H, W = scores.shape
    s_flat = scores.ravel()
    g_flat = gts.ravel()

    # ---- pixel-level AUROC / AP / F1 over all pixels ----
    auroc_px = roc_auc_score(g_flat, s_flat)
    ap_px = average_precision_score(g_flat, s_flat)
    prec, rec, _ = precision_recall_curve(g_flat, s_flat)
    f1 = 2 * prec * rec / (prec + rec)
    f1_px = float(np.max(f1[np.isfinite(f1)]))

    # ---- AU-PRO (official algorithm, lighter dtypes) ----
    structure = np.ones((3, 3), dtype=int)
    num_ok = int(np.count_nonzero(g_flat == 0))
    fp_changes = (g_flat == 0).astype(np.uint8)
    pro_changes = np.zeros(n * H * W, dtype=np.float32)
    num_regions = 0
    p3 = pro_changes.reshape(n, H, W)
    for i in range(n):
        if gts[i].max() == 0:
            continue
        labeled, n_comp = label(gts[i], structure)
        num_regions += n_comp
        for k in range(1, n_comp + 1):
            region = labeled == k
            p3[i][region] = 1.0 / np.count_nonzero(region)
    del gts, p3

    order = np.argsort(s_flat, kind="stable")[::-1].astype(np.uint32)
    np.take(s_flat, order, out=s_flat)          # scores, descending
    fp_sorted = np.take(fp_changes, order)
    pro_sorted = np.take(pro_changes, order)
    del fp_changes, pro_changes, order

    fprs = np.cumsum(fp_sorted, dtype=np.float64)
    fprs /= num_ok
    pros = np.cumsum(pro_sorted, dtype=np.float64)
    pros /= num_regions
    del fp_sorted, pro_sorted

    # keep only the last point of each run of equal scores (official behavior)
    keep_mask = np.append(np.diff(s_flat) != 0, True)
    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del s_flat, keep_mask

    fprs = np.concatenate(([0.0], fprs, [1.0]))
    pros = np.concatenate(([0.0], pros, [1.0]))
    np.clip(fprs, None, 1.0, out=fprs)
    np.clip(pros, None, 1.0, out=pros)

    au_pro = trapezoid_with_limit(fprs, pros, pro_limit)
    au_pro /= pro_limit

    return {"px_AUROC": float(auroc_px), "px_AP": float(ap_px),
            "px_F1": f1_px, "AUPRO": float(au_pro)}


def trapezoid_with_limit(fprs, pros, x_max):
    """Area under the PRO curve up to FPR=x_max, with linear interpolation of
    pro at x_max (matches official trapezoid())."""
    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    if fprs[-2] > x_max:  # last point is the appended 1.0
        idx = int(np.searchsorted(fprs, x_max))
        x0, x1 = fprs[idx - 1], fprs[idx]
        y0, y1 = pros[idx - 1], pros[idx]
        pro_at_max = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
        keep = fprs <= x_max
        fprs = np.append(fprs[keep], x_max)
        pros = np.append(pros[keep], pro_at_max)
    return float(np.sum(0.5 * (pros[1:] + pros[:-1]) * np.diff(fprs)))


def eval_category(obj, maps_dir):
    """Stream over test images of one category; return the 4 pixel metrics."""
    test_dir = os.path.join(DATA_ROOT, obj, "test")
    gt_dir = os.path.join(DATA_ROOT, obj, "ground_truth")

    entries = []  # (npy_path, gt_path_or_None, img_path)
    for sub in sorted(os.listdir(test_dir)):
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for f in sorted(os.listdir(sub_dir)):
            if not f.endswith(".png"):
                continue
            npy = os.path.join(maps_dir, obj, "test", sub, f[:-4] + ".npy")
            if not os.path.exists(npy):
                raise FileNotFoundError(npy)
            gt = None if sub == "good" else os.path.join(
                gt_dir, sub, f[:-4] + "_mask.png")
            entries.append((npy, gt, os.path.join(sub_dir, f)))

    img0 = cv2.imread(entries[0][2])
    H, W = img0.shape[:2]
    n = len(entries)
    scores = np.empty((n, H, W), dtype=np.float32)
    gts = np.zeros((n, H, W), dtype=np.uint8)
    for i, (npy, gt, img_path) in enumerate(entries):
        d = np.load(npy)
        scores[i] = dists2map(d, (H, W))
        if gt is not None:
            gts[i] = (cv2.imread(gt, cv2.IMREAD_GRAYSCALE) > 0)

    return compute_pixel_metrics(scores, gts)


def eval_seed(shot, seed):
    maps_dir = os.path.join(RESULTS_ROOT, f"{shot}-shot_preprocess=agnostic",
                            "anomaly_maps", f"seed={seed}")
    results = {}
    for obj in OBJECTS:
        m = eval_category(obj, maps_dir)
        results[obj] = m
        print(f"  {obj:<12} px_AUROC={m['px_AUROC']*100:6.2f}  "
              f"px_AP={m['px_AP']*100:6.2f}  px_F1={m['px_F1']*100:6.2f}  "
              f"AUPRO={m['AUPRO']*100:6.2f}", flush=True)
    obj_results = [r for k, r in results.items() if isinstance(r, dict)]
    results["mean_px_AUROC"] = float(np.mean([r["px_AUROC"] for r in obj_results]))
    results["mean_px_AP"] = float(np.mean([r["px_AP"] for r in obj_results]))
    results["mean_px_F1"] = float(np.mean([r["px_F1"] for r in obj_results]))
    results["mean_AUPRO"] = float(np.mean([r["AUPRO"] for r in obj_results]))
    out = os.path.join(RESULTS_ROOT, f"{shot}-shot_preprocess=agnostic",
                       f"pixel_metrics_seed={seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> mean: px_AUROC={results['mean_px_AUROC']*100:.2f}  "
          f"px_AP={results['mean_px_AP']*100:.2f}  "
          f"px_F1={results['mean_px_F1']*100:.2f}  "
          f"AUPRO={results['mean_AUPRO']*100:.2f}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        eval_seed(int(sys.argv[1]), int(sys.argv[2]))
    else:
        for shot_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*-shot_*"))):
            shot = int(os.path.basename(shot_dir).split("-shot")[0])
            maps_parent = os.path.join(shot_dir, "anomaly_maps")
            if not os.path.isdir(maps_parent):
                continue
            for seed_dir in sorted(os.listdir(maps_parent)):
                seed = int(seed_dir.split("=")[1])
                done = os.path.join(shot_dir, f"pixel_metrics_seed={seed}.json")
                if os.path.exists(done):
                    print(f"skip {shot}-shot seed={seed} (already evaluated)")
                    continue
                print(f"=== pixel eval: {shot}-shot seed={seed} ===")
                eval_seed(shot, seed)
