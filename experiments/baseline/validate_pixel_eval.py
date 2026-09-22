# -*- coding: utf-8 -*-
"""
Validate experiments/baseline/eval_pixel.py against the OFFICIAL
AnomalyDINO evaluation on the bottle category (1-shot, seed 0).

1. Regenerate full-res .tiff anomaly maps from saved .npy patch distances
   (exactly what the official detection code does: dists2map).
2. Run the OFFICIAL eval_segmentation on those tiffs (patched read_tiff).
3. Run OUR streaming pixel eval on the same .npy files.
4. Assert the numbers match (PRO differs only by float32 vs float64 noise).
"""
import os
import shutil
import sys

import cv2
import numpy as np
import tifffile as tiff

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, AD_ROOT)
os.chdir(AD_ROOT)

import src.post_eval as pe
from src.utils import dists2map
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
from eval_pixel import eval_category

# ---- patch read_tiff (Windows case-insensitive FS bug) ----
def read_tiff_windows(p, exts=(".tiff",)):
    for ext in exts:
        f = p + ext
        if os.path.exists(f):
            return tiff.imread(f)
    raise FileNotFoundError(p)
pe.read_tiff = read_tiff_windows

DATA_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
MAPS_DIR = os.path.join(AD_ROOT, "results_MVTec", "dinov2_vits14_448",
                        "1-shot_preprocess=agnostic", "anomaly_maps", "seed=0")
TMP_TIFF_ROOT = r"E:\work\freshman\results\tiff_validation"
OBJ = "bottle"

# ---- 1. regenerate tiffs for bottle from npy ----
out_base = os.path.join(TMP_TIFF_ROOT, OBJ, "test")
test_dir = os.path.join(DATA_ROOT, OBJ, "test")
n_written = 0
for sub in sorted(os.listdir(test_dir)):
    sub_dir = os.path.join(test_dir, sub)
    if not os.path.isdir(sub_dir):
        continue
    for f in sorted(os.listdir(sub_dir)):
        if not f.endswith(".png"):
            continue
        npy = os.path.join(MAPS_DIR, OBJ, "test", sub, f[:-4] + ".npy")
        img = cv2.imread(os.path.join(sub_dir, f))
        amap = dists2map(np.load(npy), img.shape)
        os.makedirs(os.path.join(out_base, sub), exist_ok=True)
        tiff.imwrite(os.path.join(out_base, sub, f[:-4] + ".tiff"), amap)
        n_written += 1
print(f"regenerated {n_written} tiff maps for '{OBJ}'")

# ---- 2. official eval on tiffs ----
gt_files, pred_files = pe.parse_dataset_files(
    object_name=OBJ, dataset_base_dir=DATA_ROOT,
    anomaly_maps_dir=TMP_TIFF_ROOT, dataset="MVTec")
au_pro, auroc_px, f1_px = pe.eval_segmentation(
    gt_files, pred_files, pro_integration_limit=0.3, delete_tiff_files=True)

# ---- 3. our eval on npy ----
ours = eval_category(OBJ, MAPS_DIR)

# ---- 4. compare ----
print("\n=== validation: official vs ours (bottle, 1-shot, seed 0) ===")
print(f"AU-PRO    official={au_pro*100:.4f}  ours={ours['AUPRO']*100:.4f}  "
      f"diff={abs(au_pro-ours['AUPRO'])*100:.5f}")
print(f"px_AUROC  official={auroc_px*100:.4f}  ours={ours['px_AUROC']*100:.4f}  "
      f"diff={abs(auroc_px-ours['px_AUROC'])*100:.5f}")
print(f"px_F1     official={f1_px*100:.4f}  ours={ours['px_F1']*100:.4f}  "
      f"diff={abs(f1_px-ours['px_F1'])*100:.5f}")

ok = (abs(au_pro - ours["AUPRO"]) < 1e-4
      and abs(auroc_px - ours["px_AUROC"]) < 1e-6
      and abs(f1_px - ours["px_F1"]) < 1e-4)
print("\nVALIDATION", "PASSED ✔" if ok else "FAILED ✘")
shutil.rmtree(TMP_TIFF_ROOT, ignore_errors=True)
sys.exit(0 if ok else 1)
