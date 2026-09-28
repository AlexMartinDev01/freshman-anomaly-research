# -*- coding: utf-8 -*-
"""
Verify that R0 and R1 really share the same bank draws.

docs/PHASE_M1_PREREGISTRATION.md section 3 promises "每个 representation 都用
同一套 bank draws".  `draw_images(offsets, shot, split, obj)` picks k image
INDICES from the train split, so two things must hold:

  1. the train image LIST (order included) is identical across caches -- a
     different sort order means index i names a different image;
  2. the test image list is identical, so r0[i] and r1[i] are the same picture.

If either fails, R0 and R1 are compared on different banks or different test
images and every dAUPRO in M1-confirm is meaningless.

Only `names` / `types` / `grids` / `offsets` are touched; loading the feature
arrays here would cost tens of GB for nothing.

Usage: python experiments/model_v0/verify_m1_align.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
MVTEC_ALL = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
             "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
             "transistor", "wood", "zipper"]
VISA_ALL = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
            "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def members(path, keys):
    """Read only the listed members -- NpzFile decompresses lazily."""
    p = os.path.join(path)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in keys if k in z.files}


def cmp(a, b):
    if a is None or b is None:
        return "MISSING"
    if a.shape != b.shape:
        return f"SHAPE {a.shape} vs {b.shape}"
    return "same" if np.array_equal(a, b) else "DIFFER"


def check(obj):
    vis = not os.path.isdir(os.path.join(V1, obj))
    R = RESULTS
    if vis:
        c0 = {"tr": os.path.join(R, "cache_visa", "final", f"{obj}_train.npz"),
              "te": os.path.join(R, "cache_visa", "final", f"{obj}_test.npz")}
    else:
        c0 = {"tr": os.path.join(R, "cache_ml_img", "final", f"{obj}_train.npz"),
              "te": os.path.join(R, "cache_ml", "final", f"{obj}_test.npz")}
    c1 = {"tr": os.path.join(R, "cache_m1_dino672", "final", f"dino672_{obj}_train.npz"),
          "te": os.path.join(R, "cache_m1_dino672", "final", f"dino672_{obj}_test.npz")}

    out = {}
    tr0 = members(c0["tr"], ["names", "offsets", "grids"])
    tr1 = members(c1["tr"], ["names", "offsets", "grids"])
    te0 = members(c0["te"], ["names", "types", "offsets"])
    te1 = members(c1["te"], ["names", "types", "offsets"])
    out["train names"] = cmp(tr0["names"], tr1["names"]) if tr0 and tr1 else "MISSING"
    out["train n"] = cmp(np.array([len(tr0["names"])]), np.array([len(tr1["names"])])) \
        if tr0 and tr1 else "MISSING"
    out["test names"] = cmp(te0["names"], te1["names"]) if te0 and te1 else "MISSING"
    out["test types"] = cmp(te0["types"], te1["types"]) if te0 and te1 else "MISSING"
    out["test n"] = cmp(np.array([len(te0["names"])]), np.array([len(te1["names"])])) \
        if te0 and te1 else "MISSING"
    g0 = tr0["grids"][0] if tr0 and "grids" in tr0 else None
    g1 = tr1["grids"][0] if tr1 and "grids" in tr1 else None
    out["train grid"] = f"{tuple(g0)} -> {tuple(g1)}" if g0 is not None and g1 is not None else "?"
    return out


def main():
    bad = []
    print(f"{'object':<14} {'train names':<12} {'train n':<9} {'test names':<12} "
          f"{'test types':<12} {'test n':<9} grid")
    for obj in MVTEC_ALL + VISA_ALL:
        r = check(obj)
        line = (f"{obj:<14} {r['train names']:<12} {r['train n']:<9} "
                f"{r['test names']:<12} {r['test types']:<12} {r['test n']:<9} "
                f"{r['train grid']}")
        print(line)
        for k, v in r.items():
            if v in ("DIFFER", "MISSING") or v.startswith("SHAPE"):
                bad.append((obj, k, v))
    print("\n" + "=" * 90)
    if bad:
        print("MISALIGNED -- R0 and R1 would NOT share banks/images:")
        for o, k, v in bad:
            print(f"  {o:<14} {k}: {v}")
        raise SystemExit(1)
    print("OK: R0 and R1 share the identical train list (so identical bank "
          "draws) and the identical test list (so r0[i] and r1[i] are the "
          "same image) for all 27 objects.")


if __name__ == "__main__":
    main()
