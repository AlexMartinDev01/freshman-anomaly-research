# -*- coding: utf-8 -*-
"""
Phase M1 -- Feature Separability Rescue: R0 vs R1 vs R2.

Criteria are frozen in docs/PHASE_M1_PREREGISTRATION.md and are NOT restated
or adjusted here; this file only measures.

    R0   DINOv2-S/14 @448, blocks 5/8/11, z-score then average (agg_all3)
    R1   DINOv2-S/14 @672, same three blocks, same aggregation
    R2   frozen wide_resnet50_2 layer2+layer3 concatenated (PatchCore-style),
         single map (no layer aggregation -- there is only one representation)

All three are measured in the SAME process with the SAME `draw_images` bank and
the same evaluator, so every cell is aligned and R0 doubles as a cross-check
against `gate_r2_layer_confirm.csv`.

Mechanism quantities, computed from RAW distances to the k-shot bank:

    C_feat = median d(defect) - median d(nearby normal)      (scale-dependent)
    sep_auc = within-image rank-AUC of d separating defect patches from nearby
              normal patches                                  (scale-free)

The pre-registration judges the mechanism on `sep_auc`, because C_feat's
absolute size depends on each representation's distance scale and is therefore
not comparable across representation families. C_feat is reported for
continuity with F1/F2.

Usage: python experiments/model_v0/phase_m1_rescue.py --objects transistor
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_f1_failure import patch_sets, NEAR_R  # noqa: E402

V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
R = RESULTS
# R0 needs DIFFERENT roots for train and test on MVTec: `cache_ml` holds the
# test features every project number was computed from, but its train split is
# a flattened, subsampled patch pool -- not per-image. Using it here would
# rebuild exactly the fixed-patch-budget bank that Phase R step 3 removed, and
# `draw_images` would silently draw from a single fake "image".
# `pfx` is the filename prefix: the M1 caches write "<repr>_<obj>_{train,test}"
# while the older cache_ml_img / cache_visa use the bare "<obj>_{train,test}".
CACHE = {
    "r0": {"mvtec_tr": os.path.join(R, "cache_ml_img"),
           "mvtec_te": os.path.join(R, "cache_ml"),
           "visa": os.path.join(R, "cache_visa"),
           "layers": ["mid", "midlate", "final"], "pfx": ""},
    "r1": {"mvtec_tr": os.path.join(R, "cache_m1_dino672"),
           "mvtec_te": os.path.join(R, "cache_m1_dino672"),
           "visa": os.path.join(R, "cache_m1_dino672"),
           "layers": ["mid", "midlate", "final"], "pfx": "dino672_"},
    "r2": {"mvtec_tr": os.path.join(R, "cache_m1_wrn50"),
           "mvtec_te": os.path.join(R, "cache_m1_wrn50"),
           "visa": os.path.join(R, "cache_m1_wrn50"),
           "layers": ["l23"], "pfx": "wrn50_"},
}
METRICS = os.path.join(R, "metrics")
MAPS = os.path.join(R, "maps_m1")


def load_cached(root, layer, key):
    p = os.path.join(root, layer, f"{key}.npz")
    with np.load(p, allow_pickle=True) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    if "offsets" not in d:                      # cache_ml test is stacked
        f = d["feats"]
        n, P, D = f.shape
        d["feats"] = f.reshape(-1, D)           # MUST flatten: the offsets
        d["offsets"] = np.arange(0, n * P + 1, P, dtype=np.int64)   # are flat
        d["grids"] = np.tile(np.array([32, 32], dtype=np.int32), (n, 1))
        d["gt_frac"] = d["gt_frac"].reshape(-1)
    return d


def gt_file(root, obj, name):
    typ, f = str(name).split("/")
    d = os.path.join(root, obj, "ground_truth", typ)
    for cand in (f[:-4] + "_mask.png", f[:-4] + ".png"):
        if os.path.exists(os.path.join(d, cand)):
            return os.path.join(d, cand)
    return None


def measure(d, gtflat, offs, i, grid):
    """C_feat and sep_auc for one image. `d` is that image's raw distance
    vector (length P); the boolean patch sets are built on the same grid."""
    o0, o1 = int(offs[i]), int(offs[i + 1])
    assert len(d) == o1 - o0, f"distance/grid mismatch: {len(d)} vs {o1 - o0}"
    g, near = patch_sets(gtflat[o0:o1].astype(np.float32), grid[0], grid[1])
    if g is None:
        return None
    lab = np.concatenate([np.ones(g.sum()), np.zeros(near.sum())])
    sc = np.concatenate([d[g.ravel()], d[near.ravel()]])
    auc = roc_auc_score(lab, sc)
    return (float(np.median(d[g.ravel()]) - np.median(d[near.ravel()])), auc)


def load_obj(repr_name, obj):
    """Load one (representation, object)'s train+test ONCE.

    These used to be loaded inside `run`, i.e. re-decompressed for every one of
    the 6 (shot, split) cells. For WRN50 on a VisA PCB category the train split
    alone is 905 x 3136 x 1536 fp16 = 8.7 GB, so that was ~18 minutes of pure
    decompression per object and a repeated ~10 GB memory spike.
    """
    cfg = CACHE[repr_name]
    root = V1 if os.path.isdir(os.path.join(V1, obj)) else VISA
    layers = cfg["layers"]
    src_tr = cfg["mvtec_tr"] if root == V1 else cfg["visa"]
    src_te = cfg["mvtec_te"] if root == V1 else cfg["visa"]
    pfx = cfg["pfx"]
    tr = {l: load_cached(src_tr, l, f"{pfx}{obj}_train") for l in layers}
    te = {l: load_cached(src_te, l, f"{pfx}{obj}_test") for l in layers}
    return root, layers, tr, te


def run(repr_name, obj, shot, split, rows, root, layers, tr, te):
    names = [str(x) for x in te[layers[0]]["names"]]
    types = [str(x) for x in te[layers[0]]["types"]]
    grids = [tuple(int(v) for v in g) for g in te[layers[0]]["grids"]]
    offs = te[layers[0]]["offsets"]
    gtflat = te[layers[0]]["gt_frac"]

    per, raw_final = {}, None
    for l in layers:
        idx = draw_images(tr[l]["offsets"], shot, split, obj)
        fo, oo = tr[l]["feats"], tr[l]["offsets"]
        bank = l2n(torch.from_numpy(np.concatenate(
            [fo[oo[i]:oo[i + 1]] for i in idx]).astype(np.float32)).to(DEV))
        maps = []
        for i in range(len(names)):
            q = torch.from_numpy(
                te[l]["feats"][offs[i]:offs[i + 1]].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bank).cpu().numpy())
        per[l] = maps
        if l == layers[-1]:
            raw_final = maps

    Z = {}
    for l in layers:
        a = np.concatenate([m.ravel() for m in per[l]])
        mu, sd = a.mean(), a.std() + 1e-12          # pooled over the test set
        Z[l] = [(m - mu) / sd for m in per[l]]
    if len(layers) > 1:
        agg = [sum(zs) / len(layers) for zs in zip(*[Z[l] for l in layers])]
    else:
        agg = Z[layers[0]]

    y = np.array([1 if t == "bad" else 0 for t in types])
    sc = np.array([mean_top1p(m) for m in agg])
    d = os.path.join(MAPS, repr_name, obj, f"{shot}shot_s{split}")
    os.makedirs(d, exist_ok=True)
    # ALL test images go into `jobs`, good ones with gt=None -- exactly as
    # gate_r2_layer_confirm does. Restricting to anomaly images changes n_neg
    # and the histogram range in pixel_metrics_binned, which moves px_AUROC and
    # AUPRO while leaving image AUROC untouched (it only uses image scores).
    # That is a silent incomparability with every other number in the project.
    jobs, cfs, aucs = [], [], []
    for i in range(len(y)):
        gh, gw = grids[i]
        p = os.path.join(d, str(names[i])[:-4].replace("/", "_") + ".npy")
        np.save(p, agg[i].reshape(gh, gw))
        # original image size: read the source image, since only cache_ml
        # stores img_hw and dists2map must resize the grid to (W, H)
        src_img = os.path.join(root, obj, "test", *str(names[i]).split("/"))
        im = cv2.imread(src_img, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cannot read {src_img} for its size")
        hw = im.shape[:2]
        gpath = gt_file(root, obj, names[i]) if y[i] else None
        if gpath is not None:
            gimg = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
            if gimg is None or (gimg > 0).sum() == 0:
                gpath = None                    # empty mask == all normal
        jobs.append((p, gpath, hw))
        if y[i]:
            m = measure(raw_final[i], gtflat, offs, i, grids[i])
            if m is not None:
                cfs.append(m[0])
                aucs.append(m[1])
    if not jobs:
        return
    mm = pixel_metrics_binned(jobs, pro_limit=0.05)
    rows.append({"object": obj, "shot": shot, "split": split,
                 "repr": repr_name, "n_bad": len(jobs),
                 "img_AUROC": roc_auc_score(y, sc) * 100,
                 "px_AUROC": mm["px_AUROC"] * 100, "AUPRO": mm["AUPRO"] * 100,
                 "C_feat": float(np.mean(cfs)) if cfs else np.nan,
                 "sep_auc": float(np.mean(aucs)) if aucs else np.nan})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--reprs", default="r0,r1,r2")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="phase_m1_rescue")
    a = ap.parse_args()
    objs = a.objects.split(",")
    reprs = a.reprs.split(",")
    shots = [int(s) for s in a.shots.split(",")]
    rows, t0 = [], time.time()
    for obj in objs:
        for rn in reprs:
            root, layers, tr, te = load_obj(rn, obj)
            for shot in shots:
                for sp in range(a.splits):
                    run(rn, obj, shot, sp, rows, root, layers, tr, te)
            del tr, te          # free the 8.7 GB train before the next repr
            print(f"  {obj:<12} {rn:<4} done ({time.time() - t0:6.0f}s, "
                  f"{len(rows)} rows)", flush=True)
        print(f"  {obj:<12} ALL done ({time.time() - t0:6.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)
    pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print(f"\n  {len(rows)} rows over {len(objs)} objects")


if __name__ == "__main__":
    main()
