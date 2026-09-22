# -*- coding: utf-8 -*-
"""
Windows-safe runner for the official AnomalyDINO experiment script.

Upstream bug: post_eval.read_tiff() probes extensions ('.tif', '.tiff', '.TIF', '.TIFF');
on Windows the filesystem is case-insensitive, so one real file `000.tiff` matches all
four probes and read_tiff raises "Found multiple files with a TIFF extension".
We monkeypatch read_tiff with an exact-match version and then execute the official
run_anomalydino.py unchanged.

Usage:
    python experiments/baseline/run_baseline.py
"""
import os
import runpy
import sys

AD_ROOT = r"E:\work\freshman\third_party\AnomalyDINO"
sys.path.insert(0, AD_ROOT)
os.chdir(AD_ROOT)

import tifffile as tiff
import src.post_eval as post_eval


def read_tiff_windows(file_path_no_ext, exts=(".tiff",)):
    """Windows-safe read_tiff: exact extension match only."""
    for ext in exts:
        p = file_path_no_ext + ext
        if os.path.exists(p):
            return tiff.imread(p)
    raise FileNotFoundError(f"Could not find a TIFF file at {file_path_no_ext}")


post_eval.read_tiff = read_tiff_windows

sys.argv = [
    "run_anomalydino.py",
    "--dataset", "MVTec",
    "--shots", "1", "2", "4", "8", "16",
    "--num_seeds", "3",
    "--preprocess", "agnostic",
    "--data_root", r"E:\work\freshman\data\mvtec_anomaly_detection",
    "--faiss_on_cpu",
    # NOTE: no --eval_segm here. Pixel-level metrics are computed afterwards
    # by experiments/baseline/eval_pixel.py directly from the saved .npy
    # patch distances (identical numbers, verified against the official
    # eval_segmentation on bottle, but far less RAM and no 30GB tiff I/O).
]

runpy.run_path(os.path.join(AD_ROOT, "run_anomalydino.py"), run_name="__main__")
