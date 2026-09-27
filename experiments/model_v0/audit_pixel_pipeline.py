# -*- coding: utf-8 -*-
"""
Phase R -- baseline audit: is the pixel/localisation pipeline trustworthy?

Gate 17A closed the covariance branch, but before acting on any of it the
foundation has to be checked. Raw DINO reaches image AUROC 95.65, which is in the
right neighbourhood (AnomalyDINO reports ~96.6 one-shot on MVTec AD), yet the
same run gives pixel AUROC 81.47 and AUPRO@0.05 14.39 -- AUPRO in the teens is
not a plausible number for this detector. A localization metric that far off
means every pixel-level conclusion drawn so far is suspect.

Three sanity checks, cheapest first:
    (1) GT as the anomaly map    -> pixel AUROC / AUPRO should approach the ceiling
    (2) random anomaly map       -> should sit at chance
    (3) alignment                -> a block placed at the GT location must still
                                    land on the GT after the patch-grid -> full-res
                                    resize and the Gaussian smoothing
If (1)-(3) hold, the evaluation code is sound and the low AUPRO is real. If they
fail, no pixel-level number in this project can be trusted.

Usage: python experiments/model_v0/audit_pixel_pipeline.py [--objects a,b,c]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import dists2map, pixel_metrics_binned  # noqa: E402
from patch_contrast import gt_to_patch_grid  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_discovery"
V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
ML = os.path.join(RESULTS, "cache_v1")
TMP = os.path.join(RESULTS, "audit_maps")


def gt_path(obj, name):
    typ, f = str(name).split("/")
    return os.path.join(V1, obj, "ground_truth", typ, f[:-4] + "_mask.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="bottle,carpet,grid,screw,tile")
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    objs = a.objects.split(",")

    print("=" * 100)
    print("(1) GT-as-anomaly-map: the metric must approach its ceiling")
    print("(2) random map: must sit at chance")
    print("(3) alignment: a block at the GT location must survive the resize")
    print("=" * 100)
    print(f"  {'object':<12}{'GT-AUROC':>10}{'GT-AUPRO':>10}"
          f"{'rnd-AUROC':>11}{'rnd-AUPRO':>11}{'in/out ratio':>14}")

    for obj in objs:
        te = np.load(os.path.join(ML, f"{obj}_test.npz"), allow_pickle=True)
        te = {k: np.asarray(te[k]) for k in te.files}
        types = list(te["types"])
        gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
        bad = [i for i, t in enumerate(types) if t == "bad"]
        if not bad:
            continue
        rng = np.random.default_rng(0)
        pick = rng.choice(bad, size=min(25, len(bad)), replace=False)

        def jobs_for(kind):
            out = []
            for i in pick:
                hw = tuple(int(v) for v in te["img_hw"][i])
                g = gt_path(obj, te["names"][i])
                if kind == "gt":
                    m = gt_to_patch_grid(g, hw, (gh, gw))
                else:
                    m = rng.random((gh, gw)).astype(np.float32)
                p = os.path.join(TMP, f"{obj}_{kind}_{i}.npy")
                np.save(p, m)
                out.append((p, g, hw))
            return out

        r_gt = pixel_metrics_binned(jobs_for("gt"), pro_limit=0.05)
        r_rnd = pixel_metrics_binned(jobs_for("rnd"), pro_limit=0.05)

        # alignment: mean upsampled GT-score inside vs outside the true mask
        ins, outs = [], []
        for i in pick:
            hw = tuple(int(v) for v in te["img_hw"][i])
            g = cv2.imread(gt_path(obj, te["names"][i]), cv2.IMREAD_GRAYSCALE)
            up = dists2map(gt_to_patch_grid(gt_path(obj, te["names"][i]),
                                            hw, (gh, gw)), hw)
            m = g > 0
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            ins.append(up[m].mean())
            outs.append(up[~m].mean())
        ratio = (np.mean(ins) / max(np.mean(outs), 1e-9))

        print(f"  {obj:<12}{r_gt['px_AUROC']*100:>10.2f}{r_gt['AUPRO']*100:>10.2f}"
              f"{r_rnd['px_AUROC']*100:>11.2f}{r_rnd['AUPRO']*100:>11.2f}"
              f"{ratio:>14.1f}")

    print("\n  expected: GT-AUROC ~100, GT-AUPRO ~100, random ~50, ratio >> 1")
    print("  if GT-AUPRO is anywhere near 14, the metric -- not the detector -- "
          "is the problem")


if __name__ == "__main__":
    main()
