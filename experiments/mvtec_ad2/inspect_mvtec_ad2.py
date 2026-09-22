# -*- coding: utf-8 -*-
"""
Inspect the MVTec AD 2 dataset: per-object sample counts for every split,
image resolution / channels / file format, and GT availability.

Writes results/mvtec_ad2/metrics/mvtec_ad2_stats.csv and prints a summary.

Run after the dataset is extracted to E:/work/freshman/data/mvtec_ad2/.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = r"E:\work\freshman\data\mvtec_ad2\mvtec_ad_2"
OUT_CSV = r"E:\work\freshman\results\mvtec_ad2\metrics\mvtec_ad2_stats.csv"

OBJECTS = ["can", "fabric", "fruit_jelly", "rice",
           "sheet_metal", "vial", "wallplugs", "walnuts"]
SPLITS = ["train", "validation", "test_public",
          "test_private", "test_private_mixed"]


def probe_image(path):
    with Image.open(path) as im:
        return im.size, im.mode, im.format  # (W,H), mode, format


def count_split(obj_dir, split):
    """Return dict with per-subtype counts for one split."""
    split_dir = os.path.join(obj_dir, split)
    info = {"split": split, "n_good": 0, "n_bad": 0, "n_total": 0,
            "res": None, "mode": None, "format": None, "has_gt": False}
    if not os.path.isdir(split_dir):
        info["n_total"] = -1  # missing
        return info

    if split in ("train", "validation"):
        # train/good/*.png, validation/good/*.png
        for sub in sorted(os.listdir(split_dir)):
            sub_dir = os.path.join(split_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            n = len([f for f in os.listdir(sub_dir) if f.endswith(".png")])
            if sub == "good":
                info["n_good"] = n
            else:
                info["n_bad"] += n
            if info["res"] is None and n > 0:
                size, mode, fmt = probe_image(
                    os.path.join(sub_dir, sorted(os.listdir(sub_dir))[0]))
                info["res"], info["mode"], info["format"] = size, mode, fmt
    elif split == "test_public":
        # test_public/{good,bad}/*.png + ground_truth/bad/*_mask.png
        for sub in ("good", "bad"):
            sub_dir = os.path.join(split_dir, sub)
            if os.path.isdir(sub_dir):
                n = len([f for f in os.listdir(sub_dir) if f.endswith(".png")])
                info["n_good" if sub == "good" else "n_bad"] = n
                if info["res"] is None and n > 0:
                    size, mode, fmt = probe_image(
                        os.path.join(sub_dir, sorted(os.listdir(sub_dir))[0]))
                    info["res"], info["mode"], info["format"] = size, mode, fmt
        info["has_gt"] = os.path.isdir(
            os.path.join(split_dir, "ground_truth", "bad"))
    else:
        # test_private / test_private_mixed: flat, names like 000_regular.png
        files = [f for f in os.listdir(split_dir) if f.endswith(".png")]
        info["n_total"] = len(files)
        if files:
            size, mode, fmt = probe_image(os.path.join(split_dir, files[0]))
            info["res"], info["mode"], info["format"] = size, mode, fmt
        info["n_good"] = -1  # labels not public
        info["n_bad"] = -1
        return info

    info["n_total"] = info["n_good"] + info["n_bad"]
    return info


def main():
    rows = []
    for obj in OBJECTS:
        obj_dir = os.path.join(DATA_ROOT, obj)
        if not os.path.isdir(obj_dir):
            print(f"WARNING: missing object dir {obj_dir}")
            continue
        for split in SPLITS:
            info = count_split(obj_dir, split)
            rows.append({
                "object": obj,
                "split": split,
                "n_good": info["n_good"],
                "n_bad": info["n_bad"],
                "n_total": info["n_total"],
                "resolution": f"{info['res'][0]}x{info['res'][1]}" if info["res"] else "?",
                "mode": info["mode"],
                "format": info["format"],
                "has_gt": info["has_gt"],
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    pivot = df.pivot(index="object", columns="split", values="n_total")
    pivot = pivot[["train", "validation", "test_public",
                   "test_private", "test_private_mixed"]]
    print("=== MVTec AD 2: samples per object/split ===")
    print(pivot.to_string())
    print("\n=== resolutions / formats ===")
    print(df[["object", "split", "resolution", "mode", "has_gt"]].to_string(index=False))
    print(f"\nsaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
