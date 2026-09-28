# -*- coding: utf-8 -*-
"""
Merge the per-object Model V2 smoke shards and evaluate the frozen criteria.

The shards exist only to cut wall-clock: objects are independent and the k-shot
draw is seeded from the object NAME, so the merged result equals what a single
process would have produced. The verdict comes from `model_v2_context.report`,
i.e. the same code that would have judged the unsharded run -- the M1/M2/M3
thresholds are not restated here.

Usage: python experiments/model_v0/model_v2_merge.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
from model_v2_context import CTRL, HARD, report  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
OBJS = HARD + CTRL
SHOTS = [1, 4]


def main():
    parts = []
    for i, o in enumerate(OBJS, 1):
        p = os.path.join(METRICS, f"model_v2_ctx_s{i}.csv")
        if not os.path.exists(p):
            print(f"  !! missing {p}")
            return
        d = pd.read_csv(p)
        print(f"  shard {i} ({o}): {len(d)} rows")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    n_obj = df["object"].nunique()
    dup = df.duplicated(subset=["object", "shot", "split", "config"]).sum()
    print(f"\n  merged: {n_obj} objects, {len(df)} rows, {dup} duplicate keys")
    if n_obj != len(OBJS) or dup:
        print("  !! refusing to report on an incomplete or overlapping merge")
        return
    report(df, OBJS, SHOTS)


if __name__ == "__main__":
    main()
