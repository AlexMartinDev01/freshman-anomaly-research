# -*- coding: utf-8 -*-
"""
Phase F1 -- why do BOTH detectors fail on the same objects? (defect-level audit)

The question is not "which detector is better" but "what makes an individual
defect instance hard for two structurally different scorers": a multi-level kNN
memory bank and a PCA-normal-subspace residual. Both are measured against the
same frozen DINOv2 ViT-S/14 @448 features, so any shared failure has to live in
the features or in the geometry of the defect itself.

Five families of per-defect quantities, all offline (no model is trained):

  1. area_frac        |GT| / |image|, at ORIGINAL mask resolution
  2. fragmentation    # connected components, largest-component share of GT
  3. shape            perimeter^2/area, elongation of the largest component
  4. C_feat           feature contrast under RAW DINO (final-layer distance to
                      the k-shot bank):
                          median d(defect patches) - median d(nearby normal
                          patches), where "nearby" = non-GT patches within a
                          Chebyshev radius R of the defect on the patch grid.
                      C_feat small == the defect is close to normal in feature
                      space. This is measured on RAW features, deliberately
                      independent of any detector's normalisation.
  5. margin / TPR     per detector, on the SAME evaluator path as AUPRO:
                          M   = median(score in GT) - P95(score outside GT)
                          TPR = fraction of GT pixels above the image's own
                                95th percentile of outside-GT scores
                      M is the robust primary; TPR is the binarised version and
                      is noisier for images whose GT covers a large area (the
                      P95 threshold is then estimated partly on defect pixels),
                      which is recorded per row as gt_covers_p95.

Detectors: kNN agg_all3 (maps_layer_confirm) and SubspaceAD aug=0 (dumps),
paired on the SAME (object, shot, split) so image-level comparison is exact.

Statistics are NOT pooled-Pearson only: global Spearman, within-object
standardised Spearman, per-object sign agreement, and an OLS with object fixed
effects (within-object demeaning) for both detectors.

Pre-registered verdict, frozen BEFORE the run:

  A  C_feat is the shared driver: within-object standardised rho(C_feat, TPR)
     <= -0.4 for BOTH detectors, and it survives object FE in the regression
     -> new question is low-contrast CONTEXTUAL localisation.
  B  spatial morphology is the shared driver: within-object standardised
     rho(area, TPR) >= +0.4 OR rho(fragmentation, TPR) <= -0.4 for BOTH
     detectors, surviving object FE
     -> new question is SPARSE/DIFFUSE spatial aggregation.
  C  neither holds for both detectors -> stop inventing models; the data does
     not identify a mechanism.

Usage: python experiments/model_v0/phase_f1_failure.py --objects a,b --tag t
"""
import argparse
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as cc_label

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import dists2map  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, MAPS, ML, V1, draw_images,  # noqa: E402
                                   l2n, load_test, load_train, nn_dist)

SA = r"E:\work\freshman\results\subspacead_b"
METRICS = os.path.join(RESULTS, "metrics")
NEAR_R = 3          # Chebyshev radius (patch grid) defining "nearby normal"
GT_BIN = 0.0        # a patch is "defect" if gt_frac > this


def gt_mask_path(obj, name):
    typ, f = str(name).split("/")
    return os.path.join(V1, obj, "ground_truth", typ, f[:-4] + "_mask.png")


def morphology(obj, name):
    """Defect geometry at ORIGINAL mask resolution."""
    p = gt_mask_path(obj, name)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    b = (m > 0).astype(np.uint8)
    area = int(b.sum())
    if area == 0:
        return None
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts))
    big = (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    ys, xs = np.nonzero(big)
    if len(xs) >= 5:
        (_, _), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        elong = float(max(w, h) / max(min(w, h), 1e-6))
    else:
        elong = float("nan")
    return {"area_frac": area / b.size, "n_cc": int(ncc),
            "largest_cc_frac": largest / area,
            "perimeter2_over_area": per * per / area, "elongation": elong}


def top1p(d):
    q = max(1, int(len(d) * 0.01))
    return float(np.sort(d)[-q:].mean())


def patch_sets(gt_frac, gh, gw):
    """bool grids: defect patches, and non-defect patches NEAR the defect."""
    g = gt_frac.reshape(gh, gw) > GT_BIN
    if not g.any():
        return None, None
    near = np.zeros_like(g)
    ys, xs = np.nonzero(g)
    for dy in range(-NEAR_R, NEAR_R + 1):
        for dx in range(-NEAR_R, NEAR_R + 1):
            yy = np.clip(ys + dy, 0, gh - 1)
            xx = np.clip(xs + dx, 0, gw - 1)
            near[yy, xx] = True
    near &= ~g
    if not near.any():
        near = ~g
    return g, near


def map_stats(map2d, gt_full, img_hw):
    """M and per-image TPR@FPR=0.05 on the SAME path as the evaluator."""
    full = dists2map(map2d, img_hw)
    g = gt_full
    if g.sum() == 0 or g.sum() == g.size:
        return None
    inside, outside = full[g], full[~g]
    thr = float(np.percentile(outside, 95))
    return {"M": float(np.median(inside) - thr),
            "TPR": float((inside > thr).mean()),
            "gt_covers_p95": float(g.mean())}


def load_sa_map(dump, obj, name):
    typ, f = str(name).split("/")
    p = os.path.join(dump, f"{obj}_{typ}_{f[:-4]}.npy")
    return np.load(p) if os.path.exists(p) else None


def run_cell(obj, shot, split, rows):
    te = load_test("final", obj)
    names = [str(n) for n in te["names"]]
    types = [str(t) for t in te["types"]]
    hws = [tuple(int(v) for v in h) for h in te["img_hw"]]
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]

    feats, offs = load_train("final", obj)
    idx = draw_images(offs, shot, split, obj)
    bank = l2n(torch.from_numpy(np.concatenate(
        [feats[offs[i]:offs[i + 1]] for i in idx]).astype(np.float32)).to(DEV))

    kmap = os.path.join(MAPS, obj, f"{shot}shot_s{split}", "agg_all3")
    dump = os.path.join(SA, "dumps", f"k{shot}_seed{split}_aug0")

    for i, nm in enumerate(names):
        if types[i] != "bad":
            continue
        gfp = np.asarray(te["gt_frac"][i], dtype=np.float32)
        g, near = patch_sets(gfp, gh, gw)
        if g is None:
            continue
        gfull = cv2.imread(gt_mask_path(obj, nm), cv2.IMREAD_GRAYSCALE)
        if gfull is None or (gfull > 0).sum() == 0:
            continue
        gfull = gfull > 0
        morph = morphology(obj, nm)
        if morph is None:
            continue

        kp = os.path.join(kmap, str(nm)[:-4].replace("/", "_") + ".npy")
        if not os.path.exists(kp):
            continue
        km = np.load(kp)
        sa = load_sa_map(dump, obj, nm)
        if sa is None:
            continue

        # raw DINO final-layer distances to the k-shot bank -> feature contrast
        q = torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
        with torch.no_grad():
            d = nn_dist(l2n(q), bank).cpu().numpy()
        c_feat = float(np.median(d[g.ravel()]) - np.median(d[near.ravel()]))

        mk = map_stats(km, gfull, hws[i])
        ms = map_stats(sa, gfull, hws[i])
        if mk is None or ms is None:
            continue
        rows.append({"object": obj, "shot": shot, "split": split, "name": nm,
                     "C_feat": c_feat,
                     "M_knn": mk["M"], "TPR_knn": mk["TPR"],
                     "M_sa": ms["M"], "TPR_sa": ms["TPR"],
                     "gt_covers_p95": mk["gt_covers_p95"],
                     "d_defect": top1p(d[g.ravel()]),
                     **morph})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="phase_f1_failure")
    a = ap.parse_args()
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1) if os.path.isdir(os.path.join(V1, d, "train")))
    shots = [int(s) for s in a.shots.split(",")]
    rows = []
    t0 = time.time()
    for obj in objs:
        for shot in shots:
            for sp in range(a.splits):
                run_cell(obj, shot, sp, rows)
        print(f"  {obj:<12} done ({time.time() - t0:5.0f}s, {len(rows)} rows)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print(f"\n  total {len(df)} rows over {df.object.nunique()} objects, "
          f"{df.name.nunique()} distinct defect images")


if __name__ == "__main__":
    main()
