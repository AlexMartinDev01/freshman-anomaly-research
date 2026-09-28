# -*- coding: utf-8 -*-
"""
Phase R step 3 -- merge the 4 confirmation shards and judge them as one run.

The shards exist only to cut wall-clock; objects are independent, and the
k-shot draw is seeded from the object NAME (see `obj_seed`), so the merged
result is identical to what a single-process run would have produced.

The verdict comes from `gate_r2_layer_confirm.report`, i.e. the same code that
would have judged the unsharded run -- the thresholds are not restated here.

Usage: python experiments/model_v0/gate_r2_confirm_merge.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate_r2_layer_confirm import METRICS, report  # noqa: E402

TAGS = [f"gate_r2_confirm_s{i}" for i in range(1, 5)]
SHOTS = [1, 2, 4, 8]


def main():
    parts = []
    for t in TAGS:
        p = os.path.join(METRICS, f"{t}.csv")
        if not os.path.exists(p):
            print(f"  !! missing {p}")
            return
        d = pd.read_csv(p)
        print(f"  {t}: {d['object'].nunique()} objects, {len(d)} rows")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    n_obj = df["object"].nunique()
    dups = df.duplicated(subset=["object", "shot", "split", "config"]).sum()
    print(f"\n  merged: {n_obj} objects, {len(df)} rows, "
          f"{dups} duplicate (object, shot, split, config) keys")
    if n_obj != 15 or dups:
        print("  !! refusing to report on an incomplete or overlapping merge")
        return
    df.to_csv(os.path.join(METRICS, "gate_r2_layer_confirm.csv"), index=False)
    report(df, SHOTS)


if __name__ == "__main__":
    main()
