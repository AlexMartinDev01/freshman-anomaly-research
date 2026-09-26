# -*- coding: utf-8 -*-
"""
Cache frozen DINOv2 patch features at three depths, for the bounded multi-layer
check that closes the representation side of the project.

    mid       block 5    texture / edges / local shape
    midlate   block 8
    final     block 11   what every gate so far has used

Same images, same transform, same grid as the existing single-layer cache, so
every previously established number stays comparable.

Train normal patches are SUBSAMPLED (TRAIN_CAP per class): they are only used to
estimate a 384x384 covariance, and storing all ~280k patches per layer would cost
gigabytes for no gain. Test patches are stored in full because retrieval needs
them.

Writes results/model_v0/cache_ml/<layer>/<obj>_{train,test}.npz with the same
schema as the single-layer cache.

Usage: python experiments/model_v0/cache_multilayer.py [obj ...]
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from ad2_pipeline import list_png  # noqa: E402
from patch_contrast import gt_to_patch_grid  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
OUT = os.path.join(RESULTS, "cache_ml")
LAYERS = {"mid": 5, "midlate": 8, "final": 11}
TRAIN_CAP = 6000


def feats_at(model, path, layer):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor, grid = model.prepare_image(img)
    with torch.no_grad():
        t = tensor.unsqueeze(0).to(model.device)
        tok = model.model.get_intermediate_layers(t, n=[layer])[0].squeeze()
    return tok.cpu().numpy(), grid, img.shape[:2]


def main():
    objs = sys.argv[1:] or sorted(
        d for d in os.listdir(V1_ROOT)
        if os.path.isdir(os.path.join(V1_ROOT, d, "train")))
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    for layer_name, idx in LAYERS.items():
        d_out = os.path.join(OUT, layer_name)
        os.makedirs(d_out, exist_ok=True)
        for obj in objs:
            tpath = os.path.join(d_out, f"{obj}_train.npz")
            epath = os.path.join(d_out, f"{obj}_test.npz")
            if os.path.exists(tpath) and os.path.exists(epath):
                continue
            # ---- train (subsampled) ----
            d = os.path.join(V1_ROOT, obj, "train", "good")
            arr, names = [], []
            for f in list_png(d):
                feats, grid, _ = feats_at(model, os.path.join(d, f), idx)
                arr.append(feats.astype(np.float16))
                names.append(f)
            A = np.stack(arr).reshape(-1, 384)
            rng = np.random.default_rng(0)
            if len(A) > TRAIN_CAP:
                A = A[rng.choice(len(A), TRAIN_CAP, replace=False)]
            np.savez_compressed(tpath, feats=A.reshape(1, -1, 384),
                                names=np.array(["subsampled"]))
            # ---- test (full) ----
            root = os.path.join(V1_ROOT, obj, "test")
            gt_root = os.path.join(V1_ROOT, obj, "ground_truth")
            fa, nm, ty, gts, hws = [], [], [], [], []
            for typ in sorted(os.listdir(root)):
                dd = os.path.join(root, typ)
                if not os.path.isdir(dd):
                    continue
                for f in list_png(dd):
                    feats, grid, hw = feats_at(model, os.path.join(dd, f), idx)
                    if typ == "good":
                        frac = np.zeros(grid, dtype=np.float32)
                    else:
                        frac = gt_to_patch_grid(
                            os.path.join(gt_root, typ, f[:-4] + "_mask.png"),
                            hw, grid)
                    fa.append(feats.astype(np.float16))
                    nm.append(f"{typ}/{f}")
                    ty.append("good" if typ == "good" else "bad")
                    gts.append(frac.astype(np.float16))
                    hws.append(hw)
            np.savez_compressed(epath, feats=np.stack(fa), names=np.array(nm),
                                types=np.array(ty), gt_frac=np.stack(gts),
                                img_hw=np.array(hws, dtype=np.int32))
            print(f"  {layer_name:<8} {obj:<12} train {len(A)}  "
                  f"test {len(nm)} ({sum(t == 'good' for t in ty)}/"
                  f"{sum(t == 'bad' for t in ty)})", flush=True)


if __name__ == "__main__":
    main()
