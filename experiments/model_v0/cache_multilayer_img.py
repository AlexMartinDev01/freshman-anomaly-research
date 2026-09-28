# -*- coding: utf-8 -*-
"""
Phase R step 3 (prep) -- re-cache train/good patch features PER IMAGE.

Why this exists
---------------
`cache_ml` stores train/good as ONE flattened patch pool per object
(subsampled to 6000 patches), so the layer smoke test could only draw a fixed
PATCH budget (shot x 1024) from it. That is not a k-shot sample: 1 "shot" could
be 1024 patches all coming from a single image, and at 8-shot the budget was
clamped to the 6000-patch pool, so 8-shot was not 8 images' worth of coverage
either.

This cache keeps every train image separate, so a k-shot draw selects k IMAGES
and uses all of their patches -- the same convention as AnomalyDINO/SubspaceAD.

The test cache is NOT rewritten: `gate_r2_layer_confirm.py` reads test features
from `cache_ml` unchanged, so the map/GT/evaluator path is bit-identical to the
run being confirmed and only the bank construction changes.

Schema: results/model_v0/cache_ml_img/<layer>/<obj>_train.npz
    feats    (N_total, 384) float16   all train patches, image-major
    offsets  (n_img+1,)     int64     image i owns feats[offsets[i]:offsets[i+1]]
    names    (n_img,)       <U      train/good file names, sorted

Grid shape is asserted constant within an object (it is, for MVTec v1 train),
so `offsets` is a convenience rather than a necessity -- but asserting beats
assuming, and a silent shape mismatch would be a 1024-way layout scramble.

All three layers come from ONE forward pass (get_intermediate_layers takes a
list), which is why this is 3x cheaper than re-running cache_multilayer.py.

Usage: python experiments/model_v0/cache_multilayer_img.py [obj ...]
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from ad2_pipeline import list_png  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
OUT = os.path.join(RESULTS, "cache_ml_img")
LAYERS = [("mid", 5), ("midlate", 8), ("final", 11)]


def main():
    objs = sys.argv[1:] or sorted(
        d for d in os.listdir(V1_ROOT)
        if os.path.isdir(os.path.join(V1_ROOT, d, "train")))
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    for obj in objs:
        d_out = {name: os.path.join(OUT, name) for name, _ in LAYERS}
        for p in d_out.values():
            os.makedirs(p, exist_ok=True)
        if all(os.path.exists(os.path.join(p, f"{obj}_train.npz"))
               for p in d_out.values()):
            print(f"  {obj:<12} already cached", flush=True)
            continue
        d = os.path.join(V1_ROOT, obj, "train", "good")
        files = list_png(d)
        acc = {name: [] for name, _ in LAYERS}
        offs = [0]
        grid0 = None
        t0 = time.time()
        for f in files:
            img = cv2.cvtColor(cv2.imread(os.path.join(d, f), cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)
            tensor, grid = model.prepare_image(img)
            with torch.no_grad():
                t = tensor.unsqueeze(0).to(model.device)
                # One forward, all three layers. Verified bit-identical to three
                # separate single-index calls (dinov2 applies `norm` after the
                # block loop, on the collected outputs only, so a multi-index
                # call cannot perturb the running residual stream).
                toks = model.model.get_intermediate_layers(
                    t, n=[i for _, i in LAYERS])
            # toks is one tensor per requested layer, in request order.
            for (name, _), tok in zip(LAYERS, toks):
                acc[name].append(tok.squeeze(0).cpu().numpy().astype(np.float16))
            if grid0 is None:
                grid0 = tuple(grid)
            elif tuple(grid) != grid0:
                raise RuntimeError(
                    f"{obj}: grid changed within train/good at {f}: "
                    f"{tuple(grid)} vs {grid0}. The offsets schema handles this, "
                    f"but nothing downstream expects it -- re-check before "
                    f"proceeding.")
            offs.append(offs[-1] + int(np.prod(grid)))
        offs = np.asarray(offs, dtype=np.int64)
        assert offs[-1] == sum(len(a) for a in acc["mid"]), "patch count mismatch"
        for name, _ in LAYERS:
            A = np.concatenate(acc[name]).reshape(-1, 384)
            np.savez_compressed(os.path.join(d_out[name], f"{obj}_train.npz"),
                                feats=A, offsets=offs, names=np.array(files))
        print(f"  {obj:<12} {len(files):>4} images  {offs[-1]:>7} patches  "
              f"grid {grid0}  {time.time() - t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
