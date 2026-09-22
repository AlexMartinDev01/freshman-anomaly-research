# -*- coding: utf-8 -*-
"""
Patch-level GT contrast (the key evidence loop).

For every anomalous test_public image:
  * map the GT mask onto the exact patch grid the model used
    (resize smaller edge to 448, crop to multiples of 14)
  * defect patch: >10% defect pixels; clean patch: 0 defect pixels
  * compare patch-distance scores:
      - mean defect-patch score vs mean clean-patch score
      - P( max clean-patch score > max defect-patch score )  <- the smoking gun:
        how often is the strongest response caused by NORMAL variation?
      - patch-level AUROC within the anomalous image (defect vs clean)
      - defect size in patches (tiny-defect analysis)

Runs on saved .npy distance maps only (no GPU needed).
Output: results/mvtec_ad2/metrics/patch_contrast.csv
        results/mvtec_ad2/figures/patch_contrast_<obj>.png
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from ad2_pipeline import AD2_OBJECTS, AD2_ROOT, RESULTS_ROOT

PATCH = 14
SMALL_EDGE = 448


def gt_to_patch_grid(gt_path, img_hw, grid):
    """Replicate the model's geometric transform on the GT mask, then
    compute the defect fraction per patch."""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    H, W = img_hw
    scale = SMALL_EDGE / min(H, W)
    nw = int(round(W * scale))
    nh = int(round(H * scale))
    gt_r = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
    gh, gw = grid[0] * PATCH, grid[1] * PATCH
    gt_r = gt_r[:gh, :gw]
    frac = gt_r.reshape(grid[0], PATCH, grid[1], PATCH).mean(axis=(1, 3)) / 255.0
    return frac  # (grid_h, grid_w) in [0,1]


def analyze_run(shot, seed, objects):
    run = f"{shot}-shot_seed={seed}"
    maps_root = os.path.join(RESULTS_ROOT, "anomaly_maps", run)
    rows = []
    img_rows = []
    for obj in objects:
        bad_dir = os.path.join(maps_root, obj, "test_public", "bad")
        if not os.path.isdir(bad_dir):
            continue
        gt_dir = os.path.join(AD2_ROOT, obj, "test_public", "ground_truth", "bad")
        for f in sorted(os.listdir(bad_dir)):
            if not f.endswith(".npy"):
                continue
            d = np.load(os.path.join(bad_dir, f))          # (gh, gw) patch dists
            name = f[:-4] + ".png"
            img = cv2.imread(os.path.join(AD2_ROOT, obj, "test_public", "bad", name))
            frac = gt_to_patch_grid(os.path.join(gt_dir, f[:-4] + "_mask.png"),
                                    img.shape[:2], d.shape)
            defect_mask = frac > 0.10
            clean_mask = frac == 0.0
            n_def = int(defect_mask.sum())
            if n_def == 0:
                continue
            d_def = d[defect_mask]
            d_clean = d[clean_mask]
            gun = float(d_clean.max()) > float(d_def.max())
            p_auroc = roc_auc_score(
                np.concatenate([np.ones(n_def), np.zeros(int(clean_mask.sum()))]),
                np.concatenate([d_def, d_clean]))
            rows.append({
                "object": obj, "shot": shot, "seed": seed, "image": f[:-4],
                "n_defect_patches": n_def,
                "defect_frac_of_image": n_def / d.size,
                "mean_defect_score": float(d_def.mean()),
                "mean_clean_score": float(d_clean.mean()),
                "max_defect_score": float(d_def.max()),
                "max_clean_score": float(d_clean.max()),
                "clean_beats_defect": gun,
                "patch_auroc": float(p_auroc),
            })
    return rows


def main():
    objects = AD2_OBJECTS
    all_rows = []
    for shot, seed in [(1, 0), (-1, 0)]:
        objs = objects if shot == 1 else ["can", "wallplugs", "vial", "sheet_metal"]
        all_rows.extend(analyze_run(shot, seed, objs))
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.join(RESULTS_ROOT, "metrics"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS_ROOT, "metrics", "patch_contrast.csv"),
              index=False)

    print("=== Patch-level GT contrast (1-shot seed 0, per object) ===")
    g = df[df.shot == 1].groupby("object")
    summ = g.agg(
        n_images=("image", "count"),
        defect_mean=("mean_defect_score", "mean"),
        clean_mean=("mean_clean_score", "mean"),
        P_clean_beats_defect=("clean_beats_defect", "mean"),
        patch_AUROC=("patch_auroc", "mean"),
        median_defect_patches=("n_defect_patches", "median"),
    ).round(3)
    print(summ.to_string())

    print("\n=== same, full-shot (4 objects) ===")
    g2 = df[df.shot == -1].groupby("object")
    summ2 = g2.agg(
        defect_mean=("mean_defect_score", "mean"),
        clean_mean=("mean_clean_score", "mean"),
        P_clean_beats_defect=("clean_beats_defect", "mean"),
        patch_AUROC=("patch_auroc", "mean"),
    ).round(3)
    print(summ2.to_string())


if __name__ == "__main__":
    main()
