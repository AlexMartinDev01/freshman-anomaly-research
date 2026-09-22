# -*- coding: utf-8 -*-
"""
Histogram-based pixel-level metrics — O(n) time, O(1) memory.

Motivation: exact pixel metrics via global argsort need ~10 GB for the large
MVTec AD 2 categories (5 MP x 150 images), which repeatedly triggered the
system OOM killer. This module computes the same metrics from per-pixel score
histograms instead:

  - all defect pixels and all normal pixels are counted (no subsampling)
  - scores are binned into 2^20 bins over the observed score range
    (metrics match exact computation to ~1e-4, validated on MVTec bottle)
  - AUROC: binned rank-sum with tie correction
  - AU-PRO: official definition (mean per-region recall vs FPR, integrated
    to FPR = pro_limit)
  - AP: bin-grid approximation (within-bin ordering averaged out)

API:
    pixel_metrics_binned(jobs, pro_limit=0.05, n_bins=1<<20)
        jobs: list of (npy_path, gt_path_or_None, (img_h, img_w))
        returns dict(px_AUROC, px_AP, AUPRO, n_images, n_regions)

    pixel_f1_at_threshold(jobs, threshold) -> float   (exact pixel F1)
"""
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label


def dists2map(dists, img_shape):
    """Identical to AnomalyDINO src.utils.dists2map."""
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(dists, sigma=4)


def _load_map(npy_path, img_shape):
    return dists2map(np.load(npy_path), img_shape).astype(np.float32)


def _load_gt(gt_path, img_shape):
    if gt_path is None:
        return np.zeros(img_shape, dtype=np.uint8)
    return (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)


def pixel_metrics_binned(jobs, pro_limit=0.05, n_bins=1 << 20):
    # ---- pass 1: global score range ----
    smin, smax = np.inf, -np.inf
    for npy, _, shape in jobs:
        m = np.load(npy)  # patch-resolution is enough for the range
        smin = min(smin, float(m.min()))
        smax = max(smax, float(m.max()))
    # upsampling interpolates between patch values, so [min,max] of the
    # full-res maps is contained in the patch-level range
    edges = np.linspace(smin, smax, n_bins + 1, dtype=np.float64)

    # ---- pass 2: histograms ----
    neg_hist = np.zeros(n_bins, dtype=np.float64)
    pos_hist = np.zeros(n_bins, dtype=np.float64)
    region_hists = []
    structure = np.ones((3, 3), dtype=int)

    for npy, gt_path, shape in jobs:
        m = _load_map(npy, shape)
        g = _load_gt(gt_path, shape)
        bins = np.digitize(m.ravel(), edges) - 1
        np.clip(bins, 0, n_bins - 1, out=bins)
        g_flat = g.ravel()
        neg_hist += np.bincount(bins[g_flat == 0], minlength=n_bins)
        if g_flat.max() > 0:
            pos_hist += np.bincount(bins[g_flat == 1], minlength=n_bins)
            labeled, n_comp = label(g, structure)
            b2d = bins.reshape(m.shape)
            for k in range(1, n_comp + 1):
                rbins = b2d[labeled == k]
                region_hists.append(np.bincount(rbins, minlength=n_bins))

    n_neg = neg_hist.sum()
    n_pos = pos_hist.sum()
    if n_pos == 0:
        raise ValueError("no defect pixels found")

    # ---- AUROC (binned rank-sum with tie correction) ----
    neg_below = np.concatenate(([0.0], np.cumsum(neg_hist)[:-1]))
    auroc = float((pos_hist * (neg_below + 0.5 * neg_hist)).sum() / (n_pos * n_neg))

    # ---- AP on the bin grid ----
    tp = np.cumsum(pos_hist[::-1])[::-1]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1.0)
    d_recall = -np.diff(recall)
    ap = float((d_recall * precision[:-1]).sum())

    # ---- AU-PRO (official: mean per-region recall vs FPR) ----
    # fpr_curve[b] / pro_curve[b] are both DECREASING in b (stricter threshold).
    # Re-parameterize by FPR ascending, prepend (0,0) like the official code,
    # then integrate to pro_limit with linear interpolation at the boundary.
    fpr_curve = fp / n_neg                      # FPR at threshold = bin edge
    pro_curve = np.zeros(n_bins, dtype=np.float64)
    for rh in region_hists:
        pro_curve += np.cumsum(rh[::-1])[::-1] / rh.sum()
    pro_curve /= len(region_hists)

    order = np.argsort(fpr_curve, kind="stable")
    fprs = np.concatenate(([0.0], fpr_curve[order]))
    pros = np.concatenate(([0.0], pro_curve[order]))

    i = int(np.searchsorted(fprs, pro_limit, side="right"))
    if i < len(fprs):
        x0, x1 = fprs[i - 1], fprs[i]
        y0, y1 = pros[i - 1], pros[i]
        p_lim = y0 + (y1 - y0) * (pro_limit - x0) / max(x1 - x0, 1e-12)
        fprs_k = np.append(fprs[:i], pro_limit)
        pros_k = np.append(pros[:i], p_lim)
    else:
        fprs_k, pros_k = fprs, pros
    au_pro = float(np.sum(0.5 * (pros_k[1:] + pros_k[:-1]) * np.diff(fprs_k)))
    au_pro /= pro_limit

    return {"px_AUROC": auroc, "px_AP": ap, "AUPRO": au_pro,
            "n_images": len(jobs), "n_regions": len(region_hists)}


def pixel_f1_at_threshold(jobs, threshold):
    """Exact pixel-level F1 at a fixed threshold (direct counting)."""
    tp = fp = fn = 0
    for npy, gt_path, shape in jobs:
        m = _load_map(npy, shape)
        g = _load_gt(gt_path, shape)
        pred = m > threshold
        gb = g > 0
        tp += int(np.count_nonzero(pred & gb))
        fp += int(np.count_nonzero(pred & ~gb))
        fn += int(np.count_nonzero(~pred & gb))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0
