# -*- coding: utf-8 -*-
"""
AD2 TEST_pub baseline: AnomalyDINO (frozen DINOv2-S/14, 448, 1NN) on all
8 objects, shots x seeds. Detection only — saves per-image scores CSV and
32x32-ish patch distance .npy files. Pixel/image metrics are computed
afterwards by eval_ad2_public.py.

Usage: python experiments/mvtec_ad2/run_ad2_baseline.py
"""
import os
import sys
import time

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")

from src.backbones import get_model
from ad2_pipeline import run_ad2_experiment, AD2_OBJECTS, RESULTS_ROOT

SHOTS = [1, 2, 4]
SEEDS = [0, 1, 2]

if __name__ == "__main__":
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    t_all = time.time()
    for shot in SHOTS:
        for seed in SEEDS:
            csv_exists = all(
                os.path.exists(os.path.join(
                    RESULTS_ROOT, "anomaly_scores",
                    f"{obj}_{shot}-shot_seed={seed}.csv"))
                for obj in AD2_OBJECTS)
            if csv_exists:
                print(f"=== {shot}-shot seed={seed}: already done, skip ===")
                continue
            print(f"=== {shot}-shot seed={seed} ===", flush=True)
            t0 = time.time()
            for obj in AD2_OBJECTS:
                run_ad2_experiment(model, obj, shot, seed,
                                   splits=("validation", "test_public"))
            print(f"    combo time: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")
