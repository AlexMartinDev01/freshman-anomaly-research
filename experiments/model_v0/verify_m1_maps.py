# -*- coding: utf-8 -*-
"""
Verify that the maps on disk reconstruct the official numbers.

phase_m1_mechanism.py reads per-image maps back from
results/model_v0/maps_m1/<repr>/<object>/<shot>shot_s<split>/<name>.npy and
assumes that path names the same map the rescue run scored.  If the path
reconstruction is off by even the ".png" stripping rule, every aupro_* column
in the mechanism CSV silently describes a different image (or nothing).

This rebuilds the GLOBAL metrics for a few cells straight from those maps and
compares them with the values in m1confirm_all.csv, which came from the maps
in memory.  A match means the on-disk maps are the ones that produced the
headline numbers.

Usage: python experiments/model_v0/verify_m1_maps.py [--cells 3]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from phase_m1_mechanism import map_path  # noqa: E402
from phase_m1_rescue import V1, VISA, gt_file, load_cached  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=4)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(METRICS, "m1confirm_all.csv"))
    c0 = df[(df.repr == "r0")].copy()
    # spread the check across datasets and shots
    picks = (c0.sort_values(["object", "shot", "split"])
               .groupby("object").head(1).iloc[::max(1, 27 // a.cells)]
               .head(a.cells))
    bad = 0
    for _, row in picks.iterrows():
        obj, shot, sp = row["object"], int(row["shot"]), int(row["split"])
        vis = not os.path.isdir(os.path.join(V1, obj))
        root = VISA if vis else V1
        te = load_cached(os.path.join(RESULTS, "cache_visa") if vis
                         else os.path.join(RESULTS, "cache_ml"), "final",
                         f"{obj}_test")
        names = [str(x) for x in te["names"]]
        types = [str(x) for x in te["types"]]
        jobs = []
        for i, nm in enumerate(names):
            src = os.path.join(root, obj, "test", *nm.split("/"))
            im = cv2.imread(src, cv2.IMREAD_COLOR)
            hw = im.shape[:2]
            p = map_path(obj, shot, sp, "r0", nm)
            gp = None
            if types[i] != "good":
                gp = gt_file(root, obj, nm)
                if gp is not None:
                    g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                    if g is None or (g > 0).sum() == 0:
                        gp = None
            jobs.append((p, gp, hw))
        mm = pixel_metrics_binned(jobs, pro_limit=0.05)
        got = (mm["px_AUROC"] * 100, mm["AUPRO"] * 100)
        want = (row["px_AUROC"], row["AUPRO"])
        d = (abs(got[0] - want[0]), abs(got[1] - want[1]))
        ok = d[0] < 1e-6 and d[1] < 1e-6
        bad += 0 if ok else 1
        print(f"  {obj:<12} {shot}-shot s{sp}  px_AUROC {got[0]:9.5f} vs "
              f"{want[0]:9.5f} (d {d[0]:.2e})   AUPRO {got[1]:9.5f} vs "
              f"{want[1]:9.5f} (d {d[1]:.2e})   {'OK' if ok else 'MISMATCH'}")
        print(f"      n_images={mm['n_images']} n_regions={mm['n_regions']} "
              f"csv rows for this cell: "
              f"{int(((df.object == obj) & (df.shot == shot) & (df.split == sp)).sum())}")
    print()
    if bad:
        raise SystemExit(f"{bad} cell(s) MISMATCH -- the on-disk maps are NOT "
                         f"the ones that produced the headline numbers")
    print("OK: the on-disk maps reproduce m1confirm_all.csv exactly, so "
          "phase_m1_mechanism.py is reading the right maps.")


if __name__ == "__main__":
    main()
