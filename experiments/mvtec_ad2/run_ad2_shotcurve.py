# -*- coding: utf-8 -*-
"""
Extended shot curve on 4 representative AD2 categories:
    can, wallplugs, vial, sheet_metal
Shots: 8 / 16 / 32 / 64 / full(-1). Seeds: 0,1,2 (full-shot: seed 0 only).
1/2/4-shot results already exist and are skipped automatically.
"""
import os
import sys
import time

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")

from src.backbones import get_model
from ad2_pipeline import run_ad2_experiment, RESULTS_ROOT

OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
CONFIGS = [(8, [0, 1, 2]), (16, [0, 1, 2]), (32, [0, 1, 2]),
           (64, [0, 1, 2]), (-1, [0])]  # -1 = full-shot

if __name__ == "__main__":
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    t_all = time.time()
    for shot, seeds in CONFIGS:
        for seed in seeds:
            todo = [obj for obj in OBJECTS if not os.path.exists(
                os.path.join(RESULTS_ROOT, "anomaly_scores",
                             f"{obj}_{shot}-shot_seed={seed}.csv"))]
            if not todo:
                print(f"=== {shot}-shot seed={seed}: done, skip ===", flush=True)
                continue
            print(f"=== {shot}-shot seed={seed} (objects: {todo}) ===", flush=True)
            t0 = time.time()
            for obj in todo:
                run_ad2_experiment(model, obj, shot, seed,
                                   splits=("validation", "test_public"))
            print(f"    combo time: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"SHOT CURVE DETECTION ALL DONE in {(time.time()-t_all)/60:.1f} min")
