# -*- coding: utf-8 -*-
"""
Phase F2 (prep) -- per-image multi-layer cache for VisA.

Same three frozen layers as every MVTec run (block 5 / 8 / 11 of DINOv2 ViT-S/14
at smaller-edge 448), same transform, same one-forward-for-three-layers trick
(verified bit-identical to three separate single-index calls on MVTec).

One structural difference from `cache_multilayer_img.py`: **VisA images within a
category have different resolutions and aspect ratios**, so after resizing the
smaller edge to 448 and cropping to a multiple of 14 the patch grid VARIES per
image. MVTec allowed stacking test features into one (n, P, 384) array; here
everything is stored flat with `offsets`, plus a `grids` (n, 2) array so the
per-image (h, w) patch shape is recoverable. Anything that reshapes a VisA map
must use `grids[i]`, not a per-category constant.

Layout (from SubspaceAD's `tools/prepare_visa.py`, split 1cls):
    <cat>/train/good/*.JPG
    <cat>/test/{good,bad}/*.JPG
    <cat>/ground_truth/bad/*.png

Schema per category per layer:
    train: feats (N,384) f16, offsets (n_img+1,), names (n_img,)
    test : feats (N,384) f16, offsets (n_img+1,), names (n_img,),
           types (n_img,) 'good'|'bad', grids (n_img,2) int32,
           gt_frac (N,) f16   flattened, sliced by the same offsets

Usage: python experiments/model_v0/cache_visa.py [cat ...]
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
from patch_contrast import gt_to_patch_grid  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
OUT = os.path.join(RESULTS, "cache_visa")
LAYERS = [("mid", 5), ("midlate", 8), ("final", 11)]


def list_img(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def feats_all(model, path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t, grid = model.prepare_image(img)
    with torch.no_grad():
        toks = model.model.get_intermediate_layers(
            t.unsqueeze(0).to(model.device), n=[i for _, i in LAYERS])
    return [tk.squeeze(0).cpu().numpy().astype(np.float16) for tk in toks], \
        tuple(int(v) for v in grid), img.shape[:2]


def main():
    cats = sys.argv[1:] or sorted(
        d for d in os.listdir(VISA)
        if os.path.isdir(os.path.join(VISA, d, "train")))
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    for cat in cats:
        outs = {n: os.path.join(OUT, n) for n, _ in LAYERS}
        for p in outs.values():
            os.makedirs(p, exist_ok=True)
        if all(os.path.exists(os.path.join(p, f"{cat}_test.npz"))
               for p in outs.values()):
            print(f"  {cat:<14} already cached", flush=True)
            continue

        # ---------------- train ----------------
        d = os.path.join(VISA, cat, "train", "good")
        files = list_img(d)
        acc = {n: [] for n, _ in LAYERS}
        offs = [0]
        grids = []
        t0 = time.time()
        for f in files:
            fs, grid, _ = feats_all(model, os.path.join(d, f))
            for (n, _), x in zip(LAYERS, fs):
                acc[n].append(x)
            offs.append(offs[-1] + int(np.prod(grid)))
            grids.append(grid)
        offs = np.asarray(offs, dtype=np.int64)
        for n, _ in LAYERS:
            A = np.concatenate(acc[n]).reshape(-1, 384)
            np.savez_compressed(os.path.join(outs[n], f"{cat}_train.npz"),
                                feats=A, offsets=offs, names=np.array(files),
                                grids=np.asarray(grids, dtype=np.int32))
        n_train = len(files)

        # ---------------- test ----------------
        root = os.path.join(VISA, cat, "test")
        gtr = os.path.join(VISA, cat, "ground_truth", "bad")
        fa = {n: [] for n, _ in LAYERS}
        gtflat = []
        nm, ty, gs, offs = [], [], [], [0]
        for sub in sorted(os.listdir(root)):
            dd = os.path.join(root, sub)
            if not os.path.isdir(dd):
                continue
            for f in list_img(dd):
                fs, grid, hw = feats_all(model, os.path.join(dd, f))
                if sub == "bad":
                    gp = os.path.join(gtr, f[:-4] + ".png")
                    frac = (gt_to_patch_grid(gp, hw, grid) if os.path.exists(gp)
                            else np.zeros(grid, dtype=np.float32))
                else:
                    frac = np.zeros(grid, dtype=np.float32)
                for (n, _), x in zip(LAYERS, fs):
                    fa[n].append(x)
                gtflat.append(frac.astype(np.float16).ravel())
                nm.append(f"{sub}/{f}")
                ty.append("good" if sub == "good" else "bad")
                gs.append(grid)
                offs.append(offs[-1] + int(np.prod(grid)))
        offs = np.asarray(offs, dtype=np.int64)
        GT = np.concatenate(gtflat)
        assert len(GT) == offs[-1], "gt/offsets mismatch"
        for n, _ in LAYERS:
            A = np.concatenate(fa[n]).reshape(-1, 384)
            np.savez_compressed(os.path.join(outs[n], f"{cat}_test.npz"),
                                feats=A, offsets=offs, names=np.array(nm),
                                types=np.array(ty),
                                grids=np.asarray(gs, dtype=np.int32),
                                gt_frac=GT)
        print(f"  {cat:<14} train {n_train:>3}  test {len(nm):>4} "
              f"({sum(t == 'good' for t in ty)}/{sum(t == 'bad' for t in ty)}) "
              f"grids {len(set(gs))} distinct  {time.time() - t0:5.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
