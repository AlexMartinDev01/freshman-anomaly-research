# -*- coding: utf-8 -*-
"""
Gate 10A -- lambda sweep for the identity-removal trade-off.

The single point R2 (lambda = 1.0) showed object-ID accuracy falling 1.00 -> 0.80
while anomaly decodability fell 0.95 -> 0.81, and R2 did not beat R1 downstream.
Before calling that a failure of the APPROACH rather than of one hyperparameter,
map the whole trade-off curve: sweep the adversarial weight and record
(object-ID accuracy, anomaly accuracy) for each.

The approach is refuted only if NO lambda lands in the useful quadrant:
object-ID down AND anomaly accuracy preserved. A curve that slides down the
diagonal is the Gate 4 whitening failure -- removing identity removes the
anomaly signal with it.

Usage: python experiments/model_v0/gate10a_lambda.py [--folds N] [--steps N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate10a_projection import Pool, probes, train_proj  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

V1 = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
LAMS = [0.0, 0.05, 0.1, 0.3, 1.0, 3.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()

    classes = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                     if f.endswith("_train.npz"))
    pool = Pool(classes, V1)
    folds = classes[::max(1, len(classes) // a.folds)][:a.folds]

    rows = []
    for heldout in folds:
        meta = [c for c in classes if c != heldout]
        for lam in LAMS:
            P = train_proj(pool, meta, "R2", steps=a.steps, lam_obj=lam, seed=0)
            r = probes(pool, meta, P)
            rows.append({"heldout": heldout, "lam": lam,
                         "obj": r["proj"]["obj"], "anom": r["proj"]["anom"]})
            print(f"  {heldout:<12} lam={lam:<5} objID={r['proj']['obj']:.3f}  "
                  f"anom={r['proj']['anom']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate10a_lambda.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, "gate10a_lambda.csv"), index=False)
    g = df.groupby("lam")[["obj", "anom"]].mean()
    print("\n" + "=" * 72)
    print("IDENTITY / ANOMALY TRADE-OFF  (raw DINO baseline: obj=1.000, "
          "anom=0.947)")
    print("=" * 72)
    print(g.round(3).to_string())
    print("\n  useful quadrant = obj well below 1.00 AND anom near 0.947")
    base = g.loc[0.0]
    g2 = g.copy()
    g2["obj_drop"] = (base.obj - g2.obj).round(3)
    g2["anom_drop"] = (base.anom - g2.anom).round(3)
    g2["anom_lost_per_id"] = (g2.anom_drop / g2.obj_drop.replace(0, np.nan)).round(2)
    print(g2.round(3).to_string())


if __name__ == "__main__":
    main()
