# -*- coding: utf-8 -*-
"""
AnomalyDINO pipeline for MVTec AD 2 (Phase 2).

Runs the *unchanged* AnomalyDINO detector (frozen DINOv2, 1NN cosine matching,
mean-top-1% image score) on MVTec AD 2's layout:

    <obj>/train/good/*.png                  reference pool (defect-free)
    <obj>/validation/good/*.png             validation (defect-free) -> threshold
    <obj>/test_public/{good,bad}/*.png      public test, GT in ground_truth/bad/
    <obj>/test_private/*.png                private test (same lighting)
    <obj>/test_private_mixed/*.png          private test (mixed lighting)

Per (object, shot, seed) this saves:
  - 32x32-ish patch distance maps as .npy per test image
  - a scores CSV (image-level mean_top1p score for val/test_public/... splits)

No tiffs are written here; full-res maps are recomputed from .npy on demand
(exactly like our validated eval_pixel.py).
"""
import csv
import os
import sys
import time

import cv2
import faiss
import numpy as np
import torch

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, AD_ROOT)

AD2_OBJECTS = ["can", "fabric", "fruit_jelly", "rice",
               "sheet_metal", "vial", "wallplugs", "walnuts"]
AD2_ROOT = r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2"
RESULTS_ROOT = r"E:\work\freshman\results\mvtec_ad2"


def mean_top1p(distances):
    """Official image score: mean of the top 1% patch distances."""
    flat = np.asarray(distances).flatten()
    k = int(len(flat) * 0.01)
    if k == 0:
        return float(np.max(flat))
    return float(np.mean(sorted(flat, reverse=True)[:k]))


def list_png(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".png"))


def extract_patch_features(model, img_path):
    image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor, grid = model.prepare_image(image)
    feats = model.extract_features(tensor)
    return feats, grid, image.shape


def build_memory_bank(model, obj, n_shot, seed, data_root=AD2_ROOT):
    """Few-shot reference selection identical to AnomalyDINO:
    sorted train/good list, slice [seed*n : (seed+1)*n]. No rotation
    augmentation for AD2 (agnostic rotation rules are MVTec-specific)."""
    ref_dir = os.path.join(data_root, obj, "train", "good")
    ref_files = list_png(ref_dir)
    if n_shot == -1:
        chosen = ref_files
    else:
        chosen = ref_files[seed * n_shot:(seed + 1) * n_shot]
    feats_all = []
    for f in chosen:
        feats, _, _ = extract_patch_features(model, os.path.join(ref_dir, f))
        feats_all.append(feats)
    feats_all = np.concatenate(feats_all, axis=0).astype("float32")
    faiss.normalize_L2(feats_all)
    index = faiss.IndexFlatL2(feats_all.shape[1])
    index.add(feats_all)
    return index, chosen


def score_split(model, obj, index, split, out_npy_dir=None, data_root=AD2_ROOT):
    """Score every image of a split. Returns list of dict rows.

    split: 'validation' | 'test_public' | 'test_private' | 'test_private_mixed'
    """
    rows = []
    if split == "validation":
        subdirs = [("good", os.path.join(data_root, obj, "validation", "good"))]
    elif split == "test_public":
        subdirs = [(s, os.path.join(data_root, obj, "test_public", s))
                   for s in ("good", "bad")]
    else:  # private splits are flat
        subdirs = [("unknown", os.path.join(data_root, obj, split))]

    for subtype, folder in subdirs:
        if not os.path.isdir(folder):
            continue
        for f in list_png(folder):
            feats, grid, img_shape = extract_patch_features(
                model, os.path.join(folder, f))
            faiss.normalize_L2(feats)
            dists, _ = index.search(feats, k=1)
            dists = dists.squeeze() / 2.0  # == cosine distance
            d_map = dists.reshape(grid)
            if out_npy_dir is not None:
                odir = os.path.join(out_npy_dir, obj, split, subtype)
                os.makedirs(odir, exist_ok=True)
                np.save(os.path.join(odir, f[:-4] + ".npy"), d_map)
            rows.append({
                "object": obj, "split": split, "type": subtype, "name": f,
                "score": mean_top1p(dists),
                "img_h": img_shape[0], "img_w": img_shape[1],
            })
    return rows


def run_ad2_experiment(model, obj, n_shot, seed, out_root=RESULTS_ROOT,
                       splits=("validation", "test_public"),
                       data_root=AD2_ROOT, tag=""):
    """Full detection for one (object, shot, seed). Returns scores csv path."""
    run_name = f"{n_shot}-shot_seed={seed}" + (f"_{tag}" if tag else "")
    out_dir = os.path.join(out_root, "anomaly_maps", run_name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_root, "anomaly_scores", f"{obj}_{run_name}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    t0 = time.time()
    index, ref_files = build_memory_bank(model, obj, n_shot, seed, data_root)
    t_bank = time.time() - t0

    all_rows = []
    t0 = time.time()
    for split in splits:
        all_rows.extend(score_split(model, obj, index, split,
                                    out_npy_dir=out_dir, data_root=data_root))
    t_inf = (time.time() - t0) / max(len(all_rows), 1)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[{obj} | {run_name}] refs={ref_files} bank={t_bank:.2f}s "
          f"infer={t_inf*1000:.1f}ms/img n={len(all_rows)} -> {csv_path}")
    return csv_path
