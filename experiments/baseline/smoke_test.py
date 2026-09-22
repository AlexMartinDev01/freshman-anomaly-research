# -*- coding: utf-8 -*-
"""
Smoke test: run AnomalyDINO on a single MVTec AD category (bottle), 1-shot, seed 0.

Goal: verify the full pipeline works end-to-end and print the tensor shapes /
data flow at each stage so we understand the method before the full baseline run.

    Image -> DINOv2 -> Patch Features -> Normal Memory Bank
          -> 1NN cosine distance -> Anomaly Map -> Image Score -> quick AUROC

Run from the repository root of AnomalyDINO:
    python E:/work/freshman/experiments/baseline/smoke_test.py
"""
import os
import sys
import time

import numpy as np
import torch

# make AnomalyDINO's src importable
ANOMALYDINO_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, ANOMALYDINO_ROOT)
os.chdir(ANOMALYDINO_ROOT)  # run script expects relative results dirs; keep them inside third_party

from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.post_eval import mean_top1p

DATA_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
OBJECT = "bottle"
OBJECT_ANOMALIES = {"bottle": ["broken_large", "broken_small", "contamination"]}
PLOTS_DIR = r"E:\work\freshman\results\smoke_test"
os.makedirs(os.path.join(PLOTS_DIR, OBJECT, "examples"), exist_ok=True)

print("=" * 70)
print("SMOKE TEST: AnomalyDINO | MVTec AD 'bottle' | 1-shot | seed 0")
print("=" * 70)

# ---------------------------------------------------------------- model
print("\n[1] Loading backbone DINOv2 ViT-S/14 ...")
model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
print(f"    params: {n_params:.1f} M")

# ------------------------------------------------- inspect the data flow
print("\n[2] Inspecting the data flow on one reference image ...")
ref_img = os.path.join(DATA_ROOT, OBJECT, "train", "good",
                       sorted(os.listdir(os.path.join(DATA_ROOT, OBJECT, "train", "good")))[0])
print("    reference image:", ref_img)
img_tensor, grid_size = model.prepare_image(ref_img)
print("    image tensor:", tuple(img_tensor.shape), "| grid size:", tuple(grid_size))
with torch.inference_mode():
    feats = model.extract_features(img_tensor)
print("    patch features:", tuple(feats.shape),
      f"  (= {grid_size[0]}x{grid_size[1]} patches x {feats.shape[1]}-dim)")

# ------------------------------------------------------- run detection
print("\n[3] Building 1-shot memory bank and running detection ...")
t0 = time.time()
anomaly_scores, time_bank, inference_times = run_anomaly_detection(
    model,
    OBJECT,
    data_root=DATA_ROOT,
    n_ref_samples=1,
    object_anomalies=OBJECT_ANOMALIES,
    plots_dir=PLOTS_DIR,
    save_examples=True,
    knn_metric="L2_normalized",   # 1 - cosine similarity
    knn_neighbors=1,
    faiss_on_cpu=True,            # faiss-gpu not available on Windows pip
    masking=False,                # smoke test: no masking / rotation
    mask_ref_images=False,
    rotation=False,
    seed=0,
    save_patch_dists=True,
    save_tiffs=False,
)
print(f"    memory bank time: {time_bank:.3f} s | "
      f"mean inference time: {np.mean(list(inference_times.values()))*1000:.1f} ms/img | "
      f"total: {time.time()-t0:.1f} s")

# ------------------------------------------------------ quick evaluation
print("\n[4] Quick check of anomaly scores ...")
good = {k: v for k, v in anomaly_scores.items() if k.startswith("good/")}
anom = {k: v for k, v in anomaly_scores.items() if not k.startswith("good/")}
print(f"    normal test images : n={len(good):3d}  "
      f"score mean={np.mean(list(good.values())):.4f}  max={np.max(list(good.values())):.4f}")
print(f"    anomalous images   : n={len(anom):3d}  "
      f"score mean={np.mean(list(anom.values())):.4f}  min={np.min(list(anom.values())):.4f}")

from sklearn.metrics import roc_auc_score, average_precision_score
y_true = [0] * len(good) + [1] * len(anom)
y_score = list(good.values()) + list(anom.values())
auroc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)
print(f"\n    bottle 1-shot image-level AUROC = {auroc*100:.2f} %")
print(f"    bottle 1-shot image-level AP    = {ap*100:.2f} %")

print("\n[5] Example plots & anomaly maps saved under:")
print("   ", PLOTS_DIR)
print("\nSMOKE TEST DONE ✔")
