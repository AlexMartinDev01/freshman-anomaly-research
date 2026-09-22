# -*- coding: utf-8 -*-
"""
Image-level aggregation sweep (Task B).

Hypothesis: on AD2 the pixel-level signal still works, but image-level ranking
collapses. If changing the patch->image aggregation recovers image AUROC, the
bottleneck is the aggregation, not the DINOv2 encoder.

Uses the saved per-image patch distance maps (.npy) — features, memory bank,
and kNN are untouched. Aggregations compared:
    max, top 0.1% / 0.5% / 1%(=current) / 2% / 5% means, p99, p99.9

Writes results/mvtec_ad2/metrics/aggregation_sweep.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from ad2_pipeline import AD2_OBJECTS, RESULTS_ROOT


def mean_topk(d, frac):
    flat = np.asarray(d).ravel()
    k = max(1, int(len(flat) * frac))
    return float(np.mean(np.partition(flat, -k)[-k:]))


AGGREGATIONS = {
    "max": lambda d: float(np.max(d)),
    "top0.1%": lambda d: mean_topk(d, 0.001),
    "top0.5%": lambda d: mean_topk(d, 0.005),
    "top1%": lambda d: mean_topk(d, 0.01),     # current AnomalyDINO score
    "top2%": lambda d: mean_topk(d, 0.02),
    "top5%": lambda d: mean_topk(d, 0.05),
    "p99": lambda d: float(np.percentile(d, 99)),
    "p99.9": lambda d: float(np.percentile(d, 99.9)),
}

rows = []
for run_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "anomaly_maps", "*-shot_seed=*"))):
    run = os.path.basename(run_dir)
    shot = int(run.split("-shot")[0])
    seed = int(run.split("seed=")[1])
    for obj in AD2_OBJECTS:
        pub_dir = os.path.join(run_dir, obj, "test_public")
        if not os.path.isdir(pub_dir):
            continue
        scores = {a: [] for a in AGGREGATIONS}
        labels = []
        for subtype in ("good", "bad"):
            sub = os.path.join(pub_dir, subtype)
            for f in sorted(os.listdir(sub)):
                if not f.endswith(".npy"):
                    continue
                d = np.load(os.path.join(sub, f))
                for a, fn in AGGREGATIONS.items():
                    scores[a].append(fn(d))
                labels.append(1 if subtype == "bad" else 0)
        y = np.array(labels)
        for a in AGGREGATIONS:
            rows.append({
                "object": obj, "shot": shot, "seed": seed, "agg": a,
                "img_AUROC": roc_auc_score(y, scores[a]) * 100,
            })

df = pd.DataFrame(rows)
os.makedirs(os.path.join(RESULTS_ROOT, "metrics"), exist_ok=True)
out = os.path.join(RESULTS_ROOT, "metrics", "aggregation_sweep.csv")
df.to_csv(out, index=False)

print("=== aggregation sweep: image AUROC, mean±std over objects×seeds ===")
piv = df.groupby(["shot", "agg"])["img_AUROC"].agg(["mean", "std"]).round(2)
print(piv.to_string())

print("\n=== best aggregation per (shot, object) ===")
best = df.loc[df.groupby(["shot", "object"])["img_AUROC"].idxmax()]
print(best.pivot_table(index="object", columns="shot", values="agg",
                       aggfunc="first").to_string())
print("\nsaved:", out)
