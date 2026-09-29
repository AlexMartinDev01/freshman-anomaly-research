# -*- coding: utf-8 -*-
"""
V3 aggregation -- per-cell metrics, then the frozen G1-G5 verdict.

Runs ONLY after the completeness audit is green. Reads the 324 cells' stored
maps and scores them with the project's shared evaluator
(`pixel_metrics_binned`, pro_limit=0.05, every test image in `jobs` with
good images carrying gt=None) -- re-deriving metrics any other way would make
these numbers incomparable with every other result in the project.

  --measure     per (cell, arm) image/px AUROC + AUPRO  -> CSV, shardable
  --analyze     category-level aggregation, then G1..G5 exactly as frozen

Usage:
    python experiments/model_v0/v3_aggregate.py --measure --tag m0
    python experiments/model_v0/v3_aggregate.py --analyze --tags m0,m1,m2,m3
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from phase_m1_rescue import V1, VISA, gt_file  # noqa: E402
from gate_r2_layer_confirm import draw_images  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

V3ROOT = os.path.join(RESULTS, "v3")
AGG = os.path.join(RESULTS, "v3_agg")
# a0 and a2 are stored in their own files (different grids: a0 is 448-grid and
# refines nothing, a2 is the 672-global reference), so they are not keys in
# arms.npz.  ALL THREE are needed: G1 is A3-vs-A0, G2 and G5 need A2, G3 needs
# the ten A4 seeds.
ARMS = ["a0", "a2", "a3", "a5"] + [f"a4s{j}" for j in range(10)]
N_SEEDS = 10


def cells():
    out = []
    for dp, _, fs in os.walk(V3ROOT):
        if "DONE.json" in fs:
            out.append(dp)
    return sorted(out)


def measure_cell(d, tmp):
    """(cell, arm) -> img_AUROC, px_AUROC, AUPRO, using the shared evaluator."""
    cid = os.path.relpath(d, V3ROOT).replace(os.sep, "/")
    ds, obj, kk, ss = cid.split("/")
    root = V1 if ds == "mvtec" else VISA
    z = np.load(os.path.join(d, "arms.npz"))
    a0 = np.load(os.path.join(d, "a0.npy"))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    te = np.load(os.path.join(RESULTS,
                              "cache_visa" if ds == "visa" else "cache_ml",
                              "final",
                              f"{obj}_test.npz"), allow_pickle=True)
    names = [str(x) for x in te["names"]]
    types = [str(x) for x in te["types"]]
    y = np.array([0 if t == "good" else 1 for t in types])

    # image sizes + GT paths, read once per cell
    jobs_meta = []
    for i, nm in enumerate(names):
        src = os.path.join(root, obj, "test", *nm.split("/"))
        im = cv2.imread(src, cv2.IMREAD_COLOR)
        hw = im.shape[:2]
        gp = None
        if y[i]:
            gp = gt_file(root, obj, nm)
            if gp is not None:
                g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                if g is None or (g > 0).sum() == 0:
                    gp = None
        jobs_meta.append((hw, gp))

    rows = []
    for arm in ARMS:
        if arm == "a0":
            M = a0
        elif arm == "a2":
            M = np.load(os.path.join(d, "a2.npy"))
        elif arm in z.files:
            M = z[arm]
        else:
            continue
        if M.shape[0] != len(names):
            raise RuntimeError(f"{cid} {arm}: {M.shape[0]} vs {len(names)}")
        sub = os.path.join(tmp, arm)
        os.makedirs(sub, exist_ok=True)
        jobs = []
        for i in range(len(names)):
            p = os.path.join(sub, f"{i}.npy")
            np.save(p, M[i])
            jobs.append((p, jobs_meta[i][1], jobs_meta[i][0]))
        mm = pixel_metrics_binned(jobs, pro_limit=0.05)
        sc = np.array([float(m.mean()) for m in M])   # image score = mean over map
        rows.append(dict(cell_id=cid, dataset=ds, object=obj, shot=int(kk[1:]),
                         split=int(ss[1:]), arm=arm,
                         img_AUROC=(roc_auc_score(y, sc) * 100 if y.min() != y.max() else np.nan),
                         px_AUROC=mm["px_AUROC"] * 100,
                         AUPRO=mm["AUPRO"] * 100, n_images=len(names)))
        shutil.rmtree(sub, ignore_errors=True)
    return rows


def run_measure(tag, shard, nshard):
    os.makedirs(AGG, exist_ok=True)
    cs = cells()
    cs = [c for i, c in enumerate(cs) if i % nshard == shard]
    out = os.path.join(AGG, f"{tag}.csv")
    # resume: a shard killed mid-run must not redo finished cells.  The CSV is
    # rewritten after every cell, so this is the same DONE-before-work idea the
    # cell runner uses, in miniature.
    rows, done = [], set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(prev.cell_id.unique())
        print(f"  resuming: {len(done)} cells already measured", flush=True)
    todo = [d for d in cs
            if os.path.relpath(d, V3ROOT).replace(os.sep, "/") not in done]
    t0 = time.time()
    for n, d in enumerate(todo, 1):
        tmp = tempfile.mkdtemp(prefix="v3agg_")
        try:
            rows.extend(measure_cell(d, tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        el = time.time() - t0
        print(f"  {n}/{len(todo)}  {el:6.0f}s  eta {el / n * (len(todo) - n) / 60:.1f}min  "
              f"{os.path.relpath(d, V3ROOT)}", flush=True)
    print(f"  {tag}: {len(rows)} rows, {len(done) + len(todo)} cells "
          f"({time.time() - t0:.0f}s this run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--tag", default="m0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.measure:
        run_measure(a.tag, a.shard, a.nshard)
    else:
        raise SystemExit("use --measure (analyze not implemented yet)")


if __name__ == "__main__":
    main()
