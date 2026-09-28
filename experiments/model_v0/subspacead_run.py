# -*- coding: utf-8 -*-
"""
Phase R step 3 -- drive the OFFICIAL SubspaceAD implementation (option B:
ViT-S/14 @ 448) over MVTec AD v1 and dump raw patch-grid maps.

Option B, per the agreed plan: same backbone as our AnomalyDINO kNN baseline
(facebook/dinov2-small = DINOv2 ViT-S/14, no registers), same resolution (448),
so the comparison isolates the scoring mechanism rather than model capacity.
Option A (giant @448) is a later scale diagnostic; option C (canonical
giant @672) needs more VRAM than this machine has (8 GB).

Layer choice: `--layers "-7,-4,-1"`. HF `hidden_states[i]` is the output of
block i (1-indexed) and the tuple has num_layers+1 entries, so on this 12-block
model those three resolve to blocks 6/9/12 == our agg_all3's blocks 5/8/11
(0-indexed) == exactly the depth range our kNN baseline averages. Note HF
returns PRE-LayerNorm states while our cache is post-LN -- that difference is
inherent to their code path and cannot be removed from the outside.

Augmentation is a separate axis: `--aug_count 30` (their canonical few-shot
recipe, random rotation) is part of SubspaceAD's method, not of the k-shot
protocol, so it is run as its own setting rather than folded into the baseline.

Every invocation runs ALL 15 categories. Passing `--categories` would change
which images each category draws, because the k-shot shuffle uses one global
`random` stream seeded once per process and consumed once per category in
sorted order -- so a subset run is not comparable to a full run. The chosen
filenames are logged and are what the paired baseline reads back.

Usage:
    python experiments/model_v0/subspacead_run.py --shots 1 --seeds 0 --augs 0 \
        --tag smoke
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

REPO = r"E:\work\freshman\third_party\SubspaceAD"
V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
OUT = r"E:\work\freshman\results\subspacead_b"
OUT_VISA = r"E:\work\freshman\results\subspacead_visa"
ML = r"E:\work\freshman\results\model_v0\cache_ml"
ML_VISA = r"E:\work\freshman\results\model_v0\cache_visa"


def expected_maps(dataset="mvtec_ad"):
    """Total test images across all objects, from our own test caches.

    A dump with fewer files than this is not a partial run, it is a naming
    collision: the dump directory is shared by all 15 categories and defect
    names repeat across them, so any key that omits the object silently
    overwrites. Counting is the only cheap way to notice -- every collided file
    still exists, so a per-file existence check passes.
    """
    if dataset == "visa":
        # count straight from the dataset tree, not from our own cache: the
        # cache may still be building, and a guard that reads a partial cache
        # would report a collision that isn't there.
        n = 0
        for c in os.listdir(VISA):
            t = os.path.join(VISA, c, "test")
            if not os.path.isdir(t):
                continue
            for sub in os.listdir(t):
                dd = os.path.join(t, sub)
                if os.path.isdir(dd):
                    n += sum(1 for f in os.listdir(dd)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return n
    n = 0
    d = os.path.join(ML, "final")   # every layer holds the same test list
    for f in os.listdir(d):
        if f.endswith("_test.npz"):
            with np.load(os.path.join(d, f), allow_pickle=True) as z:
                n += len(np.asarray(z["names"]))
    return n


def check_dump(dump, tag, dataset="mvtec_ad"):
    exp = expected_maps(dataset)
    got = len(os.listdir(dump)) if os.path.isdir(dump) else 0
    if got != exp:
        print(f"  !! {tag}: dumped {got} maps but {exp} test images exist -- "
              f"naming collision or partial run. Dumps are NOT usable.",
              flush=True)
        return False
    return True


def run_one(shot, seed, aug, dataset="mvtec_ad"):
    out_root = OUT if dataset == "mvtec_ad" else OUT_VISA
    root = V1 if dataset == "mvtec_ad" else VISA
    tag = f"k{shot}_seed{seed}_aug{aug}"
    dump = os.path.join(out_root, "dumps", tag)
    outdir = os.path.join(out_root, "runs", tag)
    os.makedirs(dump, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ)
    env["SUBSPACEAD_DUMP_DIR"] = dump
    env["PYTHONPATH"] = os.path.join(REPO, "src")
    cmd = [
        sys.executable, "-u", "main.py",
        "--dataset_name", dataset,
        "--dataset_path", root,
        "--image_res", "448",
        "--k_shot", str(shot),
        "--layers", "-7,-4,-1",
        "--model_ckpt", "facebook/dinov2-small",
        "--aug_count", str(aug),
        "--pca_ev", "0.99",
        "--agg_method", "mean",
        "--seed", str(seed),
        "--pro_integration_limit", "0.05",
        "--outdir", outdir,
    ]
    t0 = time.time()
    with open(os.path.join(out_root, f"{tag}.log"), "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    n = len(os.listdir(dump)) if os.path.isdir(dump) else 0
    print(f"  {tag:<22} exit={r.returncode}  dumped {n} maps  "
          f"{time.time() - t0:5.0f}s", flush=True)
    return r.returncode == 0 and check_dump(dump, tag, dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--augs", default="0,30")
    ap.add_argument("--dataset", default="mvtec_ad",
                    choices=["mvtec_ad", "visa"])
    a = ap.parse_args()
    shots = [int(x) for x in a.shots.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    augs = [int(x) for x in a.augs.split(",")]
    print(f"SubspaceAD official run [{a.dataset}]: ViT-S/14 @448, "
          f"shots={shots} seeds={seeds} augs={augs}")
    ok = 0
    for aug in augs:
        for shot in shots:
            for seed in seeds:
                ok += run_one(shot, seed, aug, a.dataset)
    print(f"\n  {ok}/{len(augs) * len(shots) * len(seeds)} invocations produced maps")
    print(f"  dumps under {os.path.join(OUT if a.dataset == 'mvtec_ad' else OUT_VISA, 'dumps')}")


if __name__ == "__main__":
    main()
