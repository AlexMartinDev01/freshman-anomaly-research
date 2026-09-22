# -*- coding: utf-8 -*-
"""
Cross-method stress test: SubspaceAD on MVTec AD 2.

Runs the OFFICIAL SubspaceAD main.py (canonical few-shot config from
scripts/benchmark_few_shot.sh) on the 4 representative AD2 categories,
shots x seeds. Results land in results/mvtec_ad2/stress_subspacead/.

Usage: python experiments/mvtec_ad2/run_stress_subspacead.py
"""
import os
import subprocess
import sys
import time

AD = r"E:\work\freshman\third_party\SubspaceAD"
PY = r"E:\work\freshman\.venv\Scripts\python.exe"
DATA = r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2"
OUT = r"E:\work\freshman\results\mvtec_ad2\stress_subspacead"

OBJECTS = ["can", "wallplugs", "vial", "sheet_metal"]
SHOTS = [1, 2, 4]
SEEDS = [0, 1, 2]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    env["HTTPS_PROXY"] = "http://127.0.0.1:15732"
    env["HTTP_PROXY"] = "http://127.0.0.1:15732"
    # skip attention-map saliency (unused with default masking): storing all
    # 40 layers of ViT-g attention maps at 672px OOMs an 8 GB card
    env["SUBSPACEAD_NO_SALIENCY"] = "1"
    t_all = time.time()
    for shot in SHOTS:
        for seed in SEEDS:
            outdir = os.path.join(OUT, f"k{shot}_seed{seed}")
            done_marker = os.path.join(outdir, "benchmark_results.csv")
            if os.path.exists(done_marker):
                print(f"=== k={shot} seed={seed}: done, skip ===", flush=True)
                continue
            print(f"=== k={shot} seed={seed} ===", flush=True)
            t0 = time.time()
            cmd = [
                PY, "main.py",
                "--dataset_name", "mvtec_ad2",
                "--dataset_path", DATA,
                "--categories", *OBJECTS,
                "--image_res", "672",
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
            r = subprocess.run(cmd, cwd=AD, env=env)
            if r.returncode != 0:
                print(f"!!! k={shot} seed={seed} FAILED (rc={r.returncode})",
                      flush=True)
            else:
                print(f"    done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"STRESS TEST ALL DONE in {(time.time()-t_all)/60:.1f} min")
