# -*- coding: utf-8 -*-
"""
Validate pixel_metrics_binned against exact full-array computation on
(1) MVTec bottle 1-shot seed 0   (exact: AU-PRO 96.4145, px_AUROC 98.9526)
(2) AD2 can 1-shot seed 0, FULL resolution (exact: px_AUROC 72.57, AUPRO@0.05 17.11)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
from pixel_metrics_binned import pixel_metrics_binned


def mvtec_jobs():
    maps = (r"E:\work\freshman\third_party\AnomalyDINO\results_MVTec"
            r"\dinov2_vits14_448\1-shot_preprocess=agnostic\anomaly_maps"
            r"\seed=0\bottle\test")
    data = r"E:\work\freshman\data\mvtec_anomaly_detection\bottle\test"
    jobs = []
    for sub in sorted(os.listdir(maps)):
        for f in sorted(os.listdir(os.path.join(maps, sub))):
            npy = os.path.join(maps, sub, f)
            img = cv2.imread(os.path.join(data, sub, f[:-4] + ".png"))
            gt = None if sub == "good" else os.path.join(
                r"E:\work\freshman\data\mvtec_anomaly_detection\bottle\ground_truth",
                sub, f[:-4] + "_mask.png")
            jobs.append((npy, gt, img.shape[:2]))
    return jobs


def ad2_can_jobs():
    maps = (r"E:\work\freshman\results\mvtec_ad2\anomaly_maps"
            r"\1-shot_seed=0\can\test_public")
    jobs = []
    for sub in ("good", "bad"):
        for f in sorted(os.listdir(os.path.join(maps, sub))):
            npy = os.path.join(maps, sub, f)
            img = cv2.imread(os.path.join(
                r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2\can\test_public",
                sub, f[:-4] + ".png"))
            gt = None if sub == "good" else os.path.join(
                r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2\can\test_public"
                r"\ground_truth\bad", f[:-4] + "_mask.png")
            jobs.append((npy, gt, img.shape[:2]))
    return jobs


print("=== (1) MVTec bottle 1-shot seed 0 ===")
m = pixel_metrics_binned(mvtec_jobs(), pro_limit=0.3)
print(f"binned : px_AUROC={m['px_AUROC']*100:.4f}  AUPRO={m['AUPRO']*100:.4f}")
print(f"exact  : px_AUROC=98.9526  AUPRO=96.4145")

print("\n=== (2) AD2 can 1-shot seed 0 (FULL resolution) ===")
m = pixel_metrics_binned(ad2_can_jobs(), pro_limit=0.05)
print(f"binned : px_AUROC={m['px_AUROC']*100:.4f}  AUPRO@0.05={m['AUPRO']*100:.4f}")
print(f"exact  : px_AUROC=72.5700  AUPRO@0.05=17.1100  (from smoke test, full-res)")
