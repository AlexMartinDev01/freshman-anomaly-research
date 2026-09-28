# -*- coding: utf-8 -*-
"""
Phase M1 (analysis) -- evaluate the frozen rescue criteria.

Criteria live in docs/PHASE_M1_PREREGISTRATION.md and are NOT restated or
adjusted here; this file only computes them.

  M1-1  mechanism: mean(sep_auc_R - sep_auc_R0) over the 6 hard objects > 0,
        and at least 4/6 hard objects improve
  M1-2  localisation: mean AUPRO gain over the hard objects >= +2.0
  M1-3  consistency: >= 5/7 tested objects positive on AUPRO
  M1-4  guard: mean image AUROC gain >= -0.3

Judged separately for R1 (DINO@672) and R2 (WRN50 layer2+3). Control `carpet`
is reported but excluded from M1-1/M1-2.

The mechanism is judged on `sep_auc`, not on raw `C_feat`: C_feat's absolute
magnitude depends on each representation's distance scale, so comparing it
across representation families would be a scale confound (see the
pre-registration, section 4). C_feat is printed alongside for continuity.

Usage: python experiments/model_v0/phase_m1_analysis.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
HARD = ["transistor", "screw", "zipper", "pcb4", "pcb2", "pcb3"]
ALL7 = HARD + ["carpet"]
REPRS = ["r1", "r2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="phase_m1_A_mvtec,phase_m1_B_visa")
    a = ap.parse_args()
    parts = []
    for t in a.tags.split(","):
        p = os.path.join(METRICS, f"{t}.csv")
        if not os.path.exists(p):
            print(f"  !! missing {p}")
            return
        d = pd.read_csv(p)
        print(f"  {t}: {len(d)} rows, {d.object.nunique()} objects")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    piv = df.pivot_table(index=["object", "shot", "split"], columns="repr",
                         values=["AUPRO", "img_AUROC", "px_AUROC", "sep_auc",
                                 "C_feat"])
    objs = [o for o in ALL7 if o in piv.index.get_level_values(0)]
    if len(objs) != 7:
        print(f"  !! only {len(objs)}/7 objects present -- refusing to judge")
        return

    print("\n" + "=" * 100)
    print("PHASE M1 -- Feature Separability Rescue (R0=448 vs R1=672 vs R2=CNN)")
    print("=" * 100)
    for met in ["AUPRO", "px_AUROC", "img_AUROC", "sep_auc"]:
        g = df.groupby(["object", "repr"])[met].mean().unstack()
        print(f"\n  {met}  (mean over shots x splits)")
        print(g[["r0", "r1", "r2"]].round(3).to_string())

    print("\n" + "=" * 100)
    print("PRE-REGISTERED CRITERIA")
    print("=" * 100)
    verdicts = {}
    for R in REPRS:
        print(f"\n  ---- {R} vs R0 ----")
        d_sep = (piv[("sep_auc", R)] - piv[("sep_auc", "r0")]).groupby(level=0).mean()
        d_pr = (piv[("AUPRO", R)] - piv[("AUPRO", "r0")]).groupby(level=0).mean()
        d_im = (piv[("img_AUROC", R)] - piv[("img_AUROC", "r0")]).groupby(level=0).mean()
        d_px = (piv[("px_AUROC", R)] - piv[("px_AUROC", "r0")]).groupby(level=0).mean()

        hard_sep = d_sep[HARD]
        m1 = hard_sep.mean() > 0 and int((hard_sep > 0).sum()) >= 4
        m2 = d_pr[HARD].mean() >= 2.0
        m3 = int((d_pr > 0).sum()) >= 5
        m4 = d_im.mean() >= -0.3
        print(f"    AUPRO   per object: " +
              "  ".join(f"{o}={v:+.2f}" for o, v in d_pr.items()))
        print(f"    sep_auc per object: " +
              "  ".join(f"{o}={v:+.3f}" for o, v in d_sep.items()))
        print(f"    img     per object: " +
              "  ".join(f"{o}={v:+.2f}" for o, v in d_im.items()))
        print(f"\n    M1-1 sep_auc hard mean {hard_sep.mean():+.3f} > 0 and "
              f"{int((hard_sep > 0).sum())}/6 positive   -> {'OK' if m1 else 'FAIL'}")
        print(f"    M1-2 AUPRO hard mean {d_pr[HARD].mean():+.2f} (need >= +2.0)"
              f"        -> {'OK' if m2 else 'FAIL'}")
        print(f"    M1-3 AUPRO positive {int((d_pr > 0).sum())}/7 (need >= 5)"
              f"               -> {'OK' if m3 else 'FAIL'}")
        print(f"    M1-4 img_AUROC mean {d_im.mean():+.2f} (need >= -0.3)"
              f"          -> {'OK' if m4 else 'FAIL'}")
        ok = m1 and m2 and m3 and m4
        verdicts[R] = ok
        print(f"    => {R} {'PASSES all four' if ok else 'does NOT pass'}")

    print("\n" + "=" * 100)
    print("OUTCOME (frozen before the run)")
    print("=" * 100)
    r1, r2 = verdicts.get("r1", False), verdicts.get("r2", False)
    if r1:
        print("  -> A: high resolution wins -> High-resolution / coarse-to-fine "
              "DINO anomaly localization")
        if r2:
            print("     (R2 also passes; the CNN-vs-resolution comparison decides "
                  "which mechanism dominates)")
    elif r2:
        print("  -> B: the CNN local representation wins while 672 does not -> "
              "hybrid DINO semantic + local texture branch")
    else:
        print("  -> C: neither rescues separability -> STOP model development; "
              "do not try a fourth backbone")


if __name__ == "__main__":
    main()
