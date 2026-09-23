# -*- coding: utf-8 -*-
"""
Model v0 analysis: did suppressing the normal tail actually work?

The diagnosis's bet is: normal-tail length drives the failure. So the pass/fail
criterion is NOT just "AUROC went up" -- it is the specific signature

    normal p99  down        (the tail was actually suppressed)
    defect mean ~ flat      (defect evidence was not destroyed)
    frac_defect_below_p99 down   (defect and normal distributions separated)
    AUROC / AU-PRO up       (the separation translated into detection)

A config that raises AUROC while shrinking the defect response just as much has
not fixed anything -- AUROC is rank-based, so it is immune to a uniform shrink.
`frac_defect_below_p99` is the scale-robust separation measure to read.

Usage: python analysis/model_v0/compare_tail.py [--csv path]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
DEFAULT = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"

CONFIG_DESC = {
    "A": "baseline (frozen DINOv2)",
    "B0": "adapter + MEAN (no preserve)",
    "B": "adapter + MEAN + PRESERVE",
    "C": "adapter + TAIL (no preserve)",
    "D": "adapter + TAIL + PRESERVE  <- v0",
}
KEY = ["img_AUROC", "px_AUROC", "AUPRO@0.05", "normal_p95", "normal_p99",
       "normal_p99_5", "normal_mean", "defect_mean", "defect_p50",
       "frac_defect_below_p99", "defect_normal_margin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    base = df[df.config == "A"].set_index("object")

    print("=" * 100)
    print("Model v0 -- Oracle ablation (full train/good: 80% bank / 20% tail-query)")
    print("=" * 100)
    cols = [c for c in KEY if c in df.columns]
    for obj in df.object.unique():
        print(f"\n--- {obj} ---")
        sub = df[df.object == obj][["config"] + cols].copy()
        for c in cols:
            if c in ("img_AUROC", "px_AUROC", "AUPRO@0.05"):
                sub[c] = sub[c].round(1)
            else:
                sub[c] = sub[c].round(4)
        sub["config"] = sub["config"].map(lambda c: f"{c}: {CONFIG_DESC.get(c, c)}")
        print(sub.to_string(index=False))
        if obj in base.index:
            b = base.loc[obj]
            d = df[df.object == obj].set_index("config")
            print(f"  delta vs A:  "
                  f"img_AUROC " + "  ".join(
                      f"{c}={d.loc[c,'img_AUROC']-b['img_AUROC']:+.1f}"
                      for c in d.index if c != "A"))

    print("\n" + "=" * 100)
    print("Mechanism check: tail suppressed? defect preserved? (ratio to config A)")
    print("=" * 100)
    rows = []
    for obj in df.object.unique():
        if obj not in base.index:
            continue
        b = base.loc[obj]
        for _, r in df[(df.object == obj) & (df.config != "A")].iterrows():
            rows.append({
                "object": obj, "config": r["config"],
                "p99_ratio": r["normal_p99"] / b["normal_p99"],
                "mean_ratio": r["normal_mean"] / b["normal_mean"],
                "defect_ratio": r["defect_mean"] / b["defect_mean"],
                "below_p99_before": b["frac_defect_below_p99"],
                "below_p99_after": r["frac_defect_below_p99"],
                "d_img_AUROC": r["img_AUROC"] - b["img_AUROC"],
                "d_AUPRO": r["AUPRO@0.05"] - b["AUPRO@0.05"],
            })
    m = pd.DataFrame(rows).round(3)
    print(m.to_string(index=False))
    print("\n  p99_ratio < 1  -> normal tail shrank")
    print("  defect_ratio ~ 1 -> defect evidence preserved (a low ratio means the")
    print("                     adapter shrank everything, which AUROC cannot see)")
    print("  below_p99_after should fall well below below_p99_before")


if __name__ == "__main__":
    main()
