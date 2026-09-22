# -*- coding: utf-8 -*-
"""
Smoke test: AnomalyDINO on MVTec AD 2, single category, 1-shot, seed 0.

Verifies the full chain on AD2's layout:
    train/good (1 ref) -> memory bank -> validation scores (threshold mu+3sigma)
    -> test_public scores -> image metrics -> pixel metrics from .npy + GT.

Usage:
    python experiments/mvtec_ad2/smoke_test_ad2.py [object]   (default: can)
"""
import os
import sys

import numpy as np
import pandas as pd

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, AD_ROOT)
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")

from src.backbones import get_model
from ad2_pipeline import run_ad2_experiment, AD2_ROOT, RESULTS_ROOT
from eval_pixel import compute_pixel_metrics, dists2map
import cv2

OBJ = sys.argv[1] if len(sys.argv) > 1 else "can"
SHOT, SEED = 1, 0

print("=" * 70)
print(f"AD2 SMOKE TEST: AnomalyDINO | object={OBJ} | 1-shot | seed 0")
print("=" * 70)

model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
csv_path = run_ad2_experiment(model, OBJ, SHOT, SEED,
                              splits=("validation", "test_public"))

df = pd.read_csv(csv_path)
val = df[df.split == "validation"]["score"].to_numpy()
pub = df[df.split == "test_public"]
y_true = (pub["type"] == "bad").astype(int).to_numpy()
y_score = pub["score"].to_numpy()

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
print(f"\nvalidation scores: n={len(val)} mean={val.mean():.4f} std={val.std():.4f}")
print(f"test_public: n={len(pub)} (good={int((y_true==0).sum())}, bad={int((y_true==1).sum())})")
print(f"image AUROC = {roc_auc_score(y_true, y_score)*100:.2f}")
print(f"image AP    = {average_precision_score(y_true, y_score)*100:.2f}")

thr = val.mean() + 3 * val.std()
print(f"threshold (mu+3sigma from validation) = {thr:.4f}")
print(f"image F1 @ thr = {f1_score(y_true, (y_score > thr).astype(int))*100:.2f}")

# ---------------- pixel metrics on test_public ----------------
maps_dir = os.path.join(RESULTS_ROOT, "anomaly_maps", f"{SHOT}-shot_seed={SEED}",
                        OBJ, "test_public")
scores_px, gts = [], []
for _, r in pub.iterrows():
    d = np.load(os.path.join(maps_dir, r["type"], r["name"][:-4] + ".npy"))
    scores_px.append(dists2map(d, (int(r["img_h"]), int(r["img_w"]))).astype(np.float32))
    if r["type"] == "bad":
        gt_path = os.path.join(AD2_ROOT, OBJ, "test_public", "ground_truth",
                               "bad", r["name"][:-4] + "_mask.png")
        gts.append((cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8))
    else:
        gts.append(np.zeros((int(r["img_h"]), int(r["img_w"])), dtype=np.uint8))

scores_px = np.stack(scores_px)
gts = np.stack(gts)
print(f"\npixel arrays: scores{scores_px.shape} gts{gts.shape} "
      f"({scores_px.nbytes/1e9:.2f} GB)")

m05 = compute_pixel_metrics(scores_px, gts, pro_limit=0.05)
print("\n--- pixel metrics (AD2 official: AU-PRO@0.05) ---")
print(f"px_AUROC={m05['px_AUROC']*100:.2f}  px_AP={m05['px_AP']*100:.2f}  "
      f"px_F1={m05['px_F1']*100:.2f}  AU-PRO@0.05={m05['AUPRO']*100:.2f}")

print("\nAD2 SMOKE TEST DONE \u2714")
