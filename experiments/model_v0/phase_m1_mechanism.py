# -*- coding: utf-8 -*-
"""
Phase M1-mechanism -- measure per-DEFECT-INSTANCE quantities.

Criterion C3 (docs/PHASE_M1_CONFIRM_PREREGISTRATION.md) is

    within-object Spearman( C_feat_448 , dAUPRO ) < 0

and section 4 of the M1 pre-registration asks for the per-defect study:

    C_feat^448 low  =>  d_672 large
    defect small on the 448 grid  =>  d_672 large

Both need one row per defect instance, i.e. per anomalous TEST IMAGE.  The
rescue run only wrote cell-level means, so this script re-derives the
per-image numbers from the maps it left on disk.

Per image and per representation it computes, against the ORIGINAL image size:

    TPR@5%   fraction of GT pixels above the image's own P95 of non-GT pixels
             (pre-registration section 4 sanctions "dAUPRO or dTPR@5%")
    AUPRO    exact single-image AU-PRO, integrated to FPR 0.05

WARNING -- these per-image AUPROs do NOT average to the headline AUPRO.  The
official metric integrates each region's recall against a FPR computed over
ALL normal pixels of the cell, so it is not additive over images.  Per-image
AUPRO re-normalises the FPR within each image.  It is a legitimate per-defect
effect size and is the only quantity that can be correlated within an object,
but it must never be quoted as "the AUPRO".

C_feat_448 / sep_auc_448 come from the 448 bank rebuilt with the SAME
`draw_images` call as the rescue run, and from `phase_m1_rescue.measure`, so
they are bit-identical with that run's cell means once averaged.

Usage:
    python experiments/model_v0/phase_m1_mechanism.py --objects bottle,screw
    python experiments/model_v0/phase_m1_mechanism.py --objects all --tag m1mech
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import dists2map  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import (CACHE, MAPS, V1, VISA, gt_file,  # noqa: E402
                             load_cached, measure)

METRICS = os.path.join(RESULTS, "metrics")
MVTEC_ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
             "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
             "transistor", "wood", "zipper"]
VISA_ALL = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
            "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def img_tpr(m, gb):
    """TPR at the image's own 5% false-positive budget."""
    neg, pos = m[~gb], m[gb]
    if neg.size == 0 or pos.size == 0:
        return np.nan
    return float((pos > np.percentile(neg, 95)).mean())


def img_aupro(m, gb, pro_limit=0.05, n_grid=1000):
    """Exact single-image AU-PRO.

    Same definition as pixel_metrics_binned (mean per-region recall vs FPR,
    integrated to `pro_limit`, normalised by it), but with exact thresholds
    instead of histogram bins -- one image is small enough to sort.
    """
    n_neg = int((~gb).sum())
    if n_neg == 0 or not gb.any():
        return np.nan
    lab, n_comp = label(gb, np.ones((3, 3), dtype=int))
    if n_comp == 0:
        return np.nan
    order = np.argsort(-m.ravel(), kind="stable")
    lab_o = lab.ravel()[order]
    fpr = np.concatenate(([0.0], np.cumsum(~gb.ravel()[order]) / n_neg))
    pro = np.zeros(len(fpr))
    for k in range(1, n_comp + 1):
        inr = (lab_o == k)
        pro += np.concatenate(([0.0], np.cumsum(inr) / inr.sum()))
    pro /= n_comp
    # np.interp needs strictly increasing x; among equal FPRs keep the LAST
    # (largest) recall -- more pixels admitted cannot lower recall.
    keep = np.concatenate((np.diff(fpr) > 0, [True]))
    grid = np.linspace(0.0, pro_limit, n_grid)
    p = np.interp(grid, fpr[keep], pro[keep])
    return float(np.sum(np.diff(grid) * (p[1:] + p[:-1]) * 0.5) / pro_limit)


def map_path(obj, shot, split, repr_name, name):
    """Rebuild the exact path phase_m1_rescue.run() saved."""
    return os.path.join(MAPS, repr_name, obj, f"{shot}shot_s{split}",
                        str(name)[:-4].replace("/", "_") + ".npy")


def run_object(obj, shots, splits):
    cfg = CACHE["r0"]
    root = V1 if os.path.isdir(os.path.join(V1, obj)) else VISA
    src = cfg["mvtec_tr"] if root == V1 else cfg["visa"]
    te_f = load_cached(cfg["mvtec_te"] if root == V1 else cfg["visa"], "final",
                       f"{obj}_test")
    tr_f = load_cached(src, "final", f"{obj}_train")
    offs, gtflat, grids = te_f["offsets"], te_f["gt_frac"], te_f["grids"]
    tr_off = tr_f["offsets"]

    rows = []
    for shot in shots:
        for sp in range(splits):
            idx = draw_images(tr_off, shot, sp, obj)
            fo, oo = tr_f["feats"], tr_f["offsets"]
            bank = l2n(torch.from_numpy(np.concatenate(
                [fo[oo[i]:oo[i + 1]] for i in idx]).astype(np.float32)).to(DEV))
            # raw final-layer distances, exactly as phase_m1_rescue.run() gets
            # them (it uses raw_final = the last layer of `layers`)
            dists = []
            for i in range(len(te_f["names"])):
                q = torch.from_numpy(
                    te_f["feats"][offs[i]:offs[i + 1]].astype(np.float32)).to(DEV)
                with torch.no_grad():
                    dists.append(nn_dist(l2n(q), bank).cpu().numpy())
            rows.extend(_cell_with(obj, shot, sp, root, te_f, offs, gtflat,
                                   grids, dists))
    return rows


def _cell_with(obj, shot, split, root, te_f, offs, gtflat, grids, dists):
    """One cell's per-image rows, `dists` being the raw 448 final-layer maps."""
    names = [str(x) for x in te_f["names"]]
    types = [str(x) for x in te_f["types"]]
    out = []
    for i in range(len(names)):
        if types[i] == "good":
            continue
        src = os.path.join(root, obj, "test", *names[i].split("/"))
        im = cv2.imread(src, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"cannot read {src}")
        hw = im.shape[:2]
        gpath = gt_file(root, obj, names[i])
        if gpath is None:
            continue
        gi = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
        if gi is None or (gi > 0).sum() == 0:
            continue
        gb = (cv2.resize(gi, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0)

        r = {"object": obj, "shot": shot, "split": split, "name": names[i],
             "n_gt_px": int(gb.sum())}
        ok = True
        for rn in ("r0", "r1"):
            p = map_path(obj, shot, split, rn, names[i])
            if not os.path.exists(p):
                ok = False
                break
            m = dists2map(np.load(p), hw)
            r[f"tpr_{rn}"] = img_tpr(m, gb)
            r[f"aupro_{rn}"] = img_aupro(m, gb)
        if not ok:
            continue

        o0, o1 = int(offs[i]), int(offs[i + 1])
        gf = gtflat[o0:o1].astype(np.float32)
        r["gt_patch_n"] = int((gf > 0.5).sum())
        r["gt_patch_area"] = float(gf.sum())
        cf = measure(dists[i], gtflat, offs, i, grids[i])
        if cf is not None:
            r["C_feat448"], r["sep_auc448"] = cf
        else:
            r["C_feat448"], r["sep_auc448"] = np.nan, np.nan
        r["d_tpr"] = r["tpr_r1"] - r["tpr_r0"]
        r["d_aupro"] = r["aupro_r1"] - r["aupro_r0"]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True,
                    help="comma list, or 'all' for 15 MVTec + 12 VisA")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="m1mechanism")
    a = ap.parse_args()
    objs = MVTEC_ALL + VISA_ALL if a.objects == "all" else a.objects.split(",")
    shots = [int(s) for s in a.shots.split(",")]

    rows, t0 = [], time.time()
    for obj in objs:
        rows.extend(run_object(obj, shots, a.splits))
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)
        print(f"  {obj:<12} done ({time.time() - t0:6.0f}s, {len(rows)} rows)",
              flush=True)
    print(f"\n  {len(rows)} defect-instance rows over {len(objs)} objects")


if __name__ == "__main__":
    main()
