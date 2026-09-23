# -*- coding: utf-8 -*-
"""
Cross-method stress test: SubspaceAD on MVTec AD 2 — MEMORY-SAFE version.

One subprocess per (shot, seed, object) so that system RAM is released between
runs and an OOM kill only costs a single combo. (Running all 4 categories in
one process accumulated pixel arrays and repeatedly got killed on this 32 GB
machine with ~20 GB used by other apps.)

Usage: python experiments/mvtec_ad2/run_stress_subspacead.py
"""
import glob
import os
import subprocess
import time

AD = r"E:\work\freshman\third_party\SubspaceAD"
PY = r"E:\work\freshman\.venv\Scripts\python.exe"
DATA = r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2"
OUT = r"E:\work\freshman\results\mvtec_ad2\stress_subspacead_res448"

OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
SHOTS = [1, 2, 4]
SEEDS = [0, 1, 2]
IMAGE_RES = "448"  # hardware adaptation: canonical is 672 (H100). At 672 the
# ViT-giant saturates the 8GB card -> WDDM paging -> watchdog kills the run.
# AnomalyDINO numbers are also at 448, so this is the matched comparison.

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    env["SUBSPACEAD_NO_SALIENCY"] = "1"  # skip 14GB attention maps (see extractor patch)
    t_all = time.time()
    for shot in SHOTS:
        for seed in SEEDS:
            for obj in OBJECTS:
                outdir = os.path.join(OUT, f"k{shot}_seed{seed}", obj)
                done = glob.glob(os.path.join(outdir, "**",
                                              "benchmark_results.csv"),
                                 recursive=True)
                if done:
                    print(f"=== k={shot} seed={seed} {obj}: done, skip ===",
                          flush=True)
                    continue
                print(f"=== k={shot} seed={seed} {obj} ===", flush=True)
                t0 = time.time()
                cmd = [
                    PY, "main.py",
                    "--dataset_name", "mvtec_ad2",
                    "--dataset_path", DATA,
                    "--categories", obj,
                    "--image_res", IMAGE_RES,
                    "--k_shot", str(shot),
                    "--layers=-12,-13,-14,-15,-16,-17,-18",
                    "--model_ckpt", "facebook/dinov2-with-registers-giant",
                    "--aug_count", "30",
                    "--pca_ev", "0.99",
                    "--seed", str(seed),
                    "--agg_method", "mean",
                    "--pro_integration_limit", "0.05",
                    "--outdir", outdir,
                ]
                for attempt in range(3):
                    r = subprocess.run(cmd, cwd=AD, env=env)
                    done = glob.glob(os.path.join(outdir, "**",
                                                  "benchmark_results.csv"),
                                     recursive=True)
                    if r.returncode == 0 and done:
                        print(f"    done in {(time.time()-t0)/60:.1f} min",
                              flush=True)
                        break
                    print(f"    attempt {attempt+1} failed/killed "
                          f"(rc={r.returncode}), retrying...", flush=True)
                else:
                    print(f"!!! k={shot} seed={seed} {obj} FAILED 3x",
                          flush=True)
    print(f"STRESS TEST ALL DONE in {(time.time()-t_all)/60:.1f} min")
