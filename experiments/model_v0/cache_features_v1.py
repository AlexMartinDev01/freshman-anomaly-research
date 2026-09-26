# -*- coding: utf-8 -*-
"""
Cache frozen DINOv2 patch features for MVTec AD v1 (15 classes).

Gate 9A trains a Target-Conditioned Defect Selector on MVTec AD v1 (target, bank)
pairs and then tests it on MVTec AD 2 -- a completely unseen dataset. v1 is the
training pool purely because it has 15 classes (210 ordered pairs) against AD2's
8 (56), and because using it means NO AD2 class ever serves as a bank during
training. The AD2 cache is untouched.

MVTec v1 differs from AD2 in ways this script has to handle:
  - test images live in per-defect-type subdirectories, not one flat folder
  - ground-truth masks are per type
  - image sizes vary within some classes, and the patch grid is derived from the
    resized image, so the grid is asserted constant per class (as in AD2). Where
    a class genuinely has mixed sizes the assertion will fire rather than
    silently stacking mismatched grids.

Writes the same schema as the AD2 cache so the two are interchangeable:
  results/model_v0/cache_v1/<obj>_{train,test}.npz
    train: feats (N,P,384) fp16, names (N,)
    test : feats (N,P,384) fp16, names, types, gt_frac (N,P) fp16, img_hw (N,2)

Usage: python experiments/model_v0/cache_features_v1.py [obj ...]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from ad2_pipeline import extract_patch_features, list_png  # noqa: E402
from patch_contrast import gt_to_patch_grid  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
CACHE = r"E:\work\freshman\results\model_v0\cache_v1"


def cache_train(model, obj):
    d = os.path.join(V1_ROOT, obj, "train", "good")
    names = list_png(d)
    arr, grid0 = [], None
    for f in names:
        feats, grid, _ = extract_patch_features(model, os.path.join(d, f))
        grid0 = grid0 or grid
        assert grid == grid0, f"grid changed within {obj}/train: {grid} vs {grid0}"
        arr.append(feats.astype(np.float16))
    np.savez_compressed(os.path.join(CACHE, f"{obj}_train.npz"),
                        feats=np.stack(arr), names=np.array(names))
    print(f"  train {obj}: {len(names)} imgs x {np.stack(arr).shape[1]} patches "
          f"grid={grid0}", flush=True)


def cache_test(model, obj):
    root = os.path.join(V1_ROOT, obj, "test")
    gt_root = os.path.join(V1_ROOT, obj, "ground_truth")
    feats_all, names, types, gts, hws = [], [], [], [], []
    grid0 = None
    for typ in sorted(os.listdir(root)):                 # good + each defect type
        d = os.path.join(root, typ)
        if not os.path.isdir(d):
            continue
        for f in list_png(d):
            p = os.path.join(d, f)
            feats, grid, hw = extract_patch_features(model, p)
            grid0 = grid0 or grid
            assert grid == grid0, f"grid changed within {obj}/test: {grid} vs {grid0}"
            if typ == "good":
                frac = np.zeros(grid, dtype=np.float32)
            else:
                frac = gt_to_patch_grid(
                    os.path.join(gt_root, typ, f[:-4] + "_mask.png"),
                    hw[:2], grid)
            # name carries the defect type: v1 reuses filenames across types
            feats_all.append(feats.astype(np.float16))
            names.append(f"{typ}/{f}")
            types.append("good" if typ == "good" else "bad")
            gts.append(frac.astype(np.float16))
            hws.append(hw[:2])
    np.savez_compressed(os.path.join(CACHE, f"{obj}_test.npz"),
                        feats=np.stack(feats_all), names=np.array(names),
                        types=np.array(types), gt_frac=np.stack(gts),
                        img_hw=np.array(hws, dtype=np.int32))
    print(f"  test  {obj}: {len(names)} imgs "
          f"({sum(t == 'good' for t in types)} good / "
          f"{sum(t == 'bad' for t in types)} bad) x "
          f"{np.stack(feats_all).shape[1]} patches", flush=True)


def main():
    objs = sys.argv[1:] or sorted(
        d for d in os.listdir(V1_ROOT)
        if os.path.isdir(os.path.join(V1_ROOT, d, "train")))
    os.makedirs(CACHE, exist_ok=True)
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    print(f"caching {len(objs)} MVTec v1 classes -> {CACHE}", flush=True)
    for obj in objs:
        print(f"=== {obj} ===", flush=True)
        cache_train(model, obj)
        cache_test(model, obj)


if __name__ == "__main__":
    main()
