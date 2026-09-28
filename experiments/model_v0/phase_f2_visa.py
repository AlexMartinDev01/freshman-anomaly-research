# -*- coding: utf-8 -*-
"""
Phase F2 -- independent replication of the F1 mechanism on VisA.

F1 (MVTec) found that both detectors' low-FPR localisation tracks the local
feature contrast C_feat, but its pre-registered inequality carried a sign error,
so that dataset is hypothesis-generating only. F2 re-tests the corrected
hypothesis on VisA, which this project had never touched.

Criteria are frozen in docs/PHASE_F2_PREREGISTRATION.md and are NOT restated or
adjusted here.

Visible differences from the MVTec run, all declared in the pre-registration:

* VisA images are NOT square (aspect 1.08-1.63, constant within a category), so
  our aspect-preserving grid is e.g. (32,48) while SubspaceAD's aspect-destroying
  resize to 448x448 always gives (32,32). This is NOT a map-alignment error:
  dists2map resizes each grid to the original (W,H), which is the correct
  interpretation of each detector's own map. What it does mean is that
  SubspaceAD's backbone saw a stretched image -- a property of their pipeline on
  non-square inputs that works AGAINST the hypothesis here, not for it.
* Every map is reshaped with its own per-image grid (offsets/grids), never a
  per-category constant.
* VisA is binary good/bad, so the analysis is category-level only.

Usage: python experiments/model_v0/phase_f2_visa.py --cats candle,pcb1 --tag t
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import dists2map  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, LAYERS, draw_images, l2n, nn_dist  # noqa: E402
from phase_f1_failure import patch_sets, map_stats  # noqa: E402

VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
CV = os.path.join(RESULTS, "cache_visa")
SA = r"E:\work\freshman\results\subspacead_visa"
METRICS = os.path.join(RESULTS, "metrics")


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def s(x):
    return [str(v) for v in x]


def gt_path(cat, name):
    typ, f = str(name).split("/")
    return os.path.join(VISA, cat, "ground_truth", "bad", f[:-4] + ".png")


def morphology(cat, name):
    m = cv2.imread(gt_path(cat, name), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    b = (m > 0).astype(np.uint8)
    area = int(b.sum())
    if area == 0:
        return None
    from scipy.ndimage import label as cc_label
    lab, ncc = cc_label(b)
    sizes = np.bincount(lab.ravel())[1:]
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts))
    big = (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    ys, xs = np.nonzero(big)
    if len(xs) >= 5:
        (_, _), (w, h), _ = cv2.minAreaRect(
            np.stack([xs, ys], 1).astype(np.float32))
        elong = float(max(w, h) / max(min(w, h), 1e-6))
    else:
        elong = float("nan")
    return {"area_frac": area / b.size, "n_cc": int(ncc),
            "largest_cc_frac": int(sizes.max()) / area,
            "perimeter2_over_area": per * per / area, "elongation": elong}


def run_cat(cat, shot, split, rows):
    te = {l: load_npz(os.path.join(CV, l, f"{cat}_test.npz")) for l in LAYERS}
    names, types = s(te["final"]["names"]), s(te["final"]["types"])
    grids = [tuple(int(v) for v in g) for g in te["final"]["grids"]]
    offs_t = te["final"]["offsets"]
    gtflat = te["final"]["gt_frac"]

    banks = {}
    for l in LAYERS:
        tr = load_npz(os.path.join(CV, l, f"{cat}_train.npz"))
        idx = draw_images(tr["offsets"], shot, split, cat)
        fo, oo = tr["feats"], tr["offsets"]
        banks[l] = l2n(torch.from_numpy(np.concatenate(
            [fo[oo[i]:oo[i + 1]] for i in idx]).astype(np.float32)
        ).to(DEV))

    dump = os.path.join(SA, "dumps", f"k{shot}_seed{split}_aug0")
    # per-layer maps for every test image
    maps = {l: [] for l in LAYERS}
    for i in range(len(names)):
        o0, o1 = int(offs_t[i]), int(offs_t[i + 1])
        for l in LAYERS:
            q = torch.from_numpy(
                te[l]["feats"][o0:o1].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps[l].append(nn_dist(l2n(q), banks[l]).cpu().numpy())
    # z-score per layer over the category's test set, then average
    Z = {}
    for l in LAYERS:
        a = np.concatenate([m.ravel() for m in maps[l]])
        mu, sd = a.mean(), a.std() + 1e-12
        Z[l] = [(m - mu) / sd for m in maps[l]]
    agg = [(a + b + c) / 3 for a, b, c in zip(Z["mid"], Z["midlate"], Z["final"])]

    for i, nm in enumerate(names):
        if types[i] != "bad":
            continue
        gh, gw = grids[i]
        o0, o1 = int(offs_t[i]), int(offs_t[i + 1])
        gfp = gtflat[o0:o1].astype(np.float32)
        g, near = patch_sets(gfp, gh, gw)
        if g is None:
            continue
        gfull = cv2.imread(gt_path(cat, nm), cv2.IMREAD_GRAYSCALE)
        if gfull is None or (gfull > 0).sum() == 0:
            continue
        gfull = gfull > 0
        morph = morphology(cat, nm)
        if morph is None:
            continue
        # image size from the GT mask (masks are saved at original resolution)
        hw = (gfull.shape[0], gfull.shape[1])

        sap = os.path.join(dump, f"{cat}_bad_{nm.split('/')[1][:-4]}.npy")
        if not os.path.exists(sap):
            continue
        sa = np.load(sap)

        # C_feat on RAW final-layer distances to the k-shot bank
        raw = maps["final"][i]
        c_feat = float(np.median(raw[g.ravel()]) - np.median(raw[near.ravel()]))

        mk = map_stats(agg[i].reshape(gh, gw), gfull, hw)
        ms = map_stats(sa, gfull, hw)
        if mk is None or ms is None:
            continue
        rows.append({"object": cat, "shot": shot, "split": split, "name": nm,
                     "C_feat": c_feat, "M_knn": mk["M"], "TPR_knn": mk["TPR"],
                     "M_sa": ms["M"], "TPR_sa": ms["TPR"],
                     "gt_covers_p95": mk["gt_covers_p95"],
                     "grid_h": gh, "grid_w": gw, **morph})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="phase_f2_visa")
    a = ap.parse_args()
    cats = a.cats.split(",") if a.cats else sorted(
        d for d in os.listdir(VISA) if os.path.isdir(os.path.join(VISA, d, "train")))
    shots = [int(x) for x in a.shots.split(",")]
    rows = []
    t0 = time.time()
    for cat in cats:
        for shot in shots:
            for sp in range(a.splits):
                run_cat(cat, shot, sp, rows)
        print(f"  {cat:<14} done ({time.time() - t0:5.0f}s, {len(rows)} rows)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print(f"\n  total {len(df)} rows over {df.object.nunique()} categories, "
          f"{df.groupby(['object', 'name']).ngroups} distinct defect images")


if __name__ == "__main__":
    main()
