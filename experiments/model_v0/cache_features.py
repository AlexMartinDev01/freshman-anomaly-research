# -*- coding: utf-8 -*-
"""
Cache frozen DINOv2 patch features so Model v0 training only ever runs the
adapter (the backbone is frozen, so its features never change).

The adapter is trained for hundreds of steps; re-running ViT-S/14 on every step
would dominate the cost, and caching makes the A/B/C/D ablation cheap enough to
run four times over.

Writes results/model_v0/cache/<obj>_{train,test}.npz
  train: feats (N, P, 384) fp16, names (N,)
  test : feats (N, P, 384) fp16, names, types, gt_frac (N, P) fp16

Usage: python experiments/model_v0/cache_features.py [obj ...]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from src.backbones import get_model
from ad2_pipeline import AD2_ROOT, extract_patch_features, list_png
from patch_contrast import gt_to_patch_grid

OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
CACHE = r"E:\work\freshman\results\model_v0\cache"


def feats_of(model, path):
    feats, grid, _ = extract_patch_features(model, path)
    return feats, grid


def cache_train(model, obj):
    d = os.path.join(AD2_ROOT, obj, "train", "good")
    names = list_png(d)
    arr, grid0 = [], None
    for f in names:
        feats, grid = feats_of(model, os.path.join(d, f))
        grid0 = grid0 or grid
        assert grid == grid0, f"grid changed within {obj}/train: {grid} vs {grid0}"
        arr.append(feats.astype(np.float16))
    out = os.path.join(CACHE, f"{obj}_train.npz")
    np.savez_compressed(out, feats=np.stack(arr), names=np.array(names))
    print(f"  train {obj}: {len(names)} imgs x {np.stack(arr).shape[1]} patches "
          f"grid={grid0} -> {out}")


def cache_test(model, obj):
    feats_all, names, types, gts, hws = [], [], [], [], []
    grid0 = None
    for typ in ("good", "bad"):
        d = os.path.join(AD2_ROOT, obj, "test_public", typ)
        gt_dir = os.path.join(AD2_ROOT, obj, "test_public", "ground_truth", "bad")
        for f in list_png(d):
            p = os.path.join(d, f)
            feats, grid = feats_of(model, p)
            grid0 = grid0 or grid
            assert grid == grid0, f"grid changed within {obj}/test: {grid} vs {grid0}"
            if typ == "bad":
                img = cv2.imread(p)
                frac = gt_to_patch_grid(
                    os.path.join(gt_dir, f[:-4] + "_mask.png"), img.shape[:2], grid)
            else:
                frac = np.zeros(grid, dtype=np.float32)
            # gt_frac (patch defect fraction) is used for the tail statistics;
            # img_hw is what pixel_metrics_binned needs to scale a patch map back
            # up for the official pixel-level metrics.
            feats_all.append(feats.astype(np.float16))
            names.append(f)
            types.append(typ)
            gts.append(frac.astype(np.float16))
            hws.append(img.shape[:2] if typ == "bad"
                       else cv2.imread(p).shape[:2])
    out = os.path.join(CACHE, f"{obj}_test.npz")
    np.savez_compressed(out, feats=np.stack(feats_all), names=np.array(names),
                        types=np.array(types), gt_frac=np.stack(gts),
                        img_hw=np.array(hws, dtype=np.int32))
    print(f"  test  {obj}: {len(names)} imgs "
          f"({sum(t == 'good' for t in types)} good / {sum(t == 'bad' for t in types)} bad)"
          f" x {np.stack(feats_all).shape[1]} patches grid={grid0} -> {out}")


def main():
    objects = sys.argv[1:] or OBJECTS
    os.makedirs(CACHE, exist_ok=True)
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    for obj in objects:
        print(f"=== {obj} ===", flush=True)
        cache_train(model, obj)
        cache_test(model, obj)


if __name__ == "__main__":
    main()
