# -*- coding: utf-8 -*-
"""
Phase 8 prep -- build the EXTERNAL defect pool.

The whole Phase 8 question is whether defect knowledge from other classes can
substitute for target-class defect labels. That needs a pool of defect patches
from classes OTHER than the target, so this extracts and stores it once.

For every AD2 class it writes:
    results/model_v0/external/<obj>_defect.npy   fp32 (M, 384) all patches with
                                                 gt_frac > 0.10 in test_public
    results/model_v0/external/<obj>_normal.npy   fp32 (K, 384) a fixed-size
                                                 random subsample of that
                                                 class's TRAIN normals, used
                                                 only for class-level matching

The normal subsample is capped because the class-matching statistic (Phase 8 R1)
only needs a distributional summary; keeping all of it would be gigabytes for
no gain.

A note on what "external" means here. The pool comes from the other AD2 classes,
so this is cross-CLASS transfer inside one dataset, not transfer from a truly
separate corpus. Any positive result therefore licenses the weaker claim and
must be stated that way.

Usage: python experiments/model_v0/build_external_pool.py [obj ...]
       (default: every class that is not the Phase 8 target set)
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import CACHE, RESULTS  # noqa: E402

EXTERNAL = os.path.join(RESULTS, "external")
NORMAL_CAP = 20000          # patches kept per class for class matching
TARGETS = ["wallplugs", "vial", "sheet_metal", "can"]


def build(obj):
    te = np.load(os.path.join(CACHE, f"{obj}_test.npz"), allow_pickle=True)
    tr = np.load(os.path.join(CACHE, f"{obj}_train.npz"), allow_pickle=True)
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
             for i in range(len(types)) if types[i] == "bad"]
    parts = [p for p in parts if len(p)]
    defect = np.concatenate(parts) if parts else np.zeros((0, 384), np.float32)

    trn = tr["feats"].reshape(-1, 384).astype(np.float32)
    rng = np.random.default_rng(0)
    if len(trn) > NORMAL_CAP:
        trn = trn[rng.choice(len(trn), size=NORMAL_CAP, replace=False)]

    np.save(os.path.join(EXTERNAL, f"{obj}_defect.npy"), defect)
    np.save(os.path.join(EXTERNAL, f"{obj}_normal.npy"), trn)
    print(f"  {obj:<14} defect {len(defect):>6}  normal_subsample {len(trn):>6}",
          flush=True)
    return obj, len(defect)


def main():
    os.makedirs(EXTERNAL, exist_ok=True)
    objs = sys.argv[1:] or sorted(
        f[:-len("_test.npz")] for f in os.listdir(CACHE)
        if f.endswith("_test.npz"))
    print(f"building external pool for {len(objs)} classes -> {EXTERNAL}")
    for o in objs:
        build(o)


if __name__ == "__main__":
    main()
