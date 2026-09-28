# -*- coding: utf-8 -*-
"""
Phase M1 (prep) -- per-image caches for the two rescue representations.

    dino672   DINOv2-S/14 at smaller-edge 672, blocks 5/8/11
              -> the SAME three layers and the same agg_all3 recipe as R0, so
                 the only thing that changes versus R0 is the input resolution
    wrn50     frozen torchvision wide_resnet50_2 (ImageNet), layer2 + layer3,
              layer3 bilinearly upsampled to layer2's grid and concatenated
              (1536-D per patch) -- the standard PatchCore-style local feature

Both are written in the same per-image schema used everywhere else
(feats / offsets / names / grids / gt_frac), so the evaluation pipeline is
shared and the only variable is the representation.

NOTE on the GT grid: `patch_contrast.gt_to_patch_grid` hard-codes PATCH=14 and
SMALL_EDGE=448. It is correct for DINO at 448 but WRONG for 672 (stride 14,
smaller edge 672) and wrong for the CNN (stride 8). Both need the same
construction with their own stride/edge, so it is re-implemented here rather
than reusing a function whose constants do not apply.

Usage: python experiments/model_v0/cache_m1_repr.py --repr dino672
       python experiments/model_v0/cache_m1_repr.py --repr wrn50
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
VISA = r"E:\work\freshman\data\VisA_pytorch\1cls"
MVTEC_OBJS = ["transistor", "screw", "zipper", "carpet"]
VISA_CATS = ["pcb4", "pcb2", "pcb3"]

DINO_LAYERS = [("mid", 5), ("midlate", 8), ("final", 11)]
DINO_EDGE, DINO_STRIDE = 672, 14
CNN_EDGE, CNN_STRIDE = 448, 8
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_img(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def gt_to_grid(gt_path, img_hw, grid, edge, stride):
    """Patch-grid defect fraction for an arbitrary (edge, stride).

    Mirrors the model's geometry exactly: resize the smaller edge to `edge`
    preserving aspect, crop to a multiple of `stride`, then average the mask
    inside each stride x stride cell.
    """
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    H, W = img_hw
    sc = edge / min(H, W)
    nw, nh = int(round(W * sc)), int(round(H * sc))
    g = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
    gh, gw = grid[0] * stride, grid[1] * stride
    g = g[:gh, :gw]
    return (g.reshape(grid[0], stride, grid[1], stride)
             .mean(axis=(1, 3)) / 255.0).astype(np.float32)


class WRN50:
    """Frozen wide_resnet50_2, layer2 + layer3, PatchCore-style."""

    def __init__(self, device="cuda"):
        import torchvision.models as M
        self.device = device
        self.m = M.wide_resnet50_2(
            weights=M.Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
        for p in self.m.parameters():
            p.requires_grad_(False)

    def feats(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        sc = CNN_EDGE / min(H, W)
        nw, nh = int(round(W * sc)), int(round(H * sc))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        gh = (nh // CNN_STRIDE) * CNN_STRIDE
        gw = (nw // CNN_STRIDE) * CNN_STRIDE
        img = img[:gh, :gw]
        x = torch.from_numpy(
            ((img - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.m.maxpool(self.m.relu(self.m.bn1(self.m.conv1(x))))
            f = self.m.layer1(f)
            l2 = self.m.layer2(f)                       # (1, 512, h/8, w/8)
            l3 = self.m.layer3(l2)                      # (1, 1024, h/16, w/16)
            l3u = torch.nn.functional.interpolate(
                l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
            cat = torch.cat([l2, l3u], dim=1)           # (1, 1536, h/8, w/8)
        grid = (cat.shape[-2], cat.shape[-1])
        return (cat.squeeze(0).permute(1, 2, 0).reshape(-1, 1536)
                .cpu().numpy().astype(np.float16), grid, (H, W))


def dino_feats(model, path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t, grid = model.prepare_image(rgb)
    with torch.no_grad():
        toks = model.model.get_intermediate_layers(
            t.unsqueeze(0).to(model.device), n=[i for _, i in DINO_LAYERS])
    return ([tk.squeeze(0).cpu().numpy().astype(np.float16) for tk in toks],
            tuple(int(v) for v in grid), rgb.shape[:2])


def run_repr(repr_name, root, names, out_layers, layer_dir):
    for l in out_layers:
        os.makedirs(os.path.join(layer_dir, l), exist_ok=True)
    if repr_name == "dino672":
        model = get_model("dinov2_vits14", "cuda", smaller_edge_size=DINO_EDGE)
        edge, stride = DINO_EDGE, DINO_STRIDE
        extract = lambda p: dino_feats(model, p)                     # noqa: E731
        multi = True
    else:
        model = WRN50("cuda")
        edge, stride = CNN_EDGE, CNN_STRIDE
        extract = model.feats
        multi = False

    for name in names:
        nkey = repr_name + "_" + name
        if all(os.path.exists(os.path.join(layer_dir, l, f"{nkey}_test.npz"))
               for l in out_layers):
            print(f"  {name:<12} already cached", flush=True)
            continue
        t0 = time.time()
        # ---------------- train ----------------
        # Preallocate instead of accumulating a Python list and concatenating
        # at the end: the CNN alone is 56*56 patches x 1536 dims = 9.6 MB fp16
        # per image, so a 900-image VisA category would hold ~8.7 GB in the
        # list and then double it in the concatenate. That is an OOM waiting to
        # happen, and it only shows up on the largest categories.
        files = list_img(os.path.join(root, name, "train", "good"))
        acc, offs, grids, buf, pos, P = None, [0], [], None, 0, None
        for j, f in enumerate(files):
            fs, grid, _ = extract(os.path.join(root, name, "train", "good", f))
            n_patch = int(np.prod(grid))
            if j == 0:
                P = n_patch
                dims = ({l: fs[k].shape[-1] for k, (l, _) in
                         enumerate(DINO_LAYERS)} if multi
                        else {out_layers[0]: fs.shape[-1]})
                buf = {l: np.empty((len(files) * P, dims[l]),
                                   dtype=np.float16) for l in out_layers}
            elif n_patch != P:
                raise RuntimeError(
                    f"{name}: patch count varies within train/good "
                    f"({n_patch} vs {P}); the preallocated buffer assumes a "
                    f"constant grid -- handle this explicitly before running.")
            if multi:
                for (l, _), x in zip(DINO_LAYERS, fs):
                    buf[l][pos:pos + P] = x
            else:
                buf[out_layers[0]][pos:pos + P] = fs
            pos += P
            offs.append(offs[-1] + P)
            grids.append(grid)
        offs = np.asarray(offs, dtype=np.int64)
        assert pos == offs[-1]
        for l in out_layers:
            np.savez_compressed(os.path.join(layer_dir, l, f"{nkey}_train.npz"),
                                feats=buf[l][:pos], offsets=offs,
                                names=np.array(files),
                                grids=np.asarray(grids, dtype=np.int32))
        del buf
        # ---------------- test ----------------
        tdir = os.path.join(root, name, "test")
        subs = sorted(d for d in os.listdir(tdir)
                      if os.path.isdir(os.path.join(tdir, d)))
        # Anomaly subdirectories are named by defect type on MVTec (so
        # `sub == "bad"` is never true there) and by the single label "bad" on
        # VisA. Testing `sub == "bad"` silently gave every MVTec test image an
        # all-zero GT.
        items = [(sub, f) for sub in subs
                 for f in list_img(os.path.join(tdir, sub))]
        B = {l: None for l in out_layers}
        gtflat, nm, ty, gs, offs = [], [], [], [], [0]
        pos, P = 0, None
        for j, (sub, f) in enumerate(items):
            p = os.path.join(tdir, sub, f)
            fs, grid, hw = extract(p)
            n_patch = int(np.prod(grid))
            if j == 0:
                P = n_patch
                dims = ({l: fs[k].shape[-1] for k, (l, _) in
                         enumerate(DINO_LAYERS)} if multi
                        else {out_layers[0]: fs.shape[-1]})
                B = {l: np.empty((len(items) * P, dims[l]), dtype=np.float16)
                     for l in out_layers}
            elif n_patch != P:
                raise RuntimeError(f"{name}: patch count varies in test "
                                   f"({n_patch} vs {P})")
            is_bad = sub != "good"
            # MVTec masks are "<stem>_mask.png"; VisA's prepared masks are
            # "<stem>.png". Try both rather than assume one -- picking wrong
            # silently yields an all-zero GT.
            gdir = os.path.join(root, name, "ground_truth", sub)
            gp = None
            for cand in (f[:-4] + "_mask.png", f[:-4] + ".png"):
                if os.path.exists(os.path.join(gdir, cand)):
                    gp = os.path.join(gdir, cand)
                    break
            if is_bad and gp is not None:
                frac = gt_to_grid(gp, hw, grid, edge, stride)
            else:
                frac = np.zeros(grid, dtype=np.float32)
            if multi:
                for (l, _), x in zip(DINO_LAYERS, fs):
                    B[l][pos:pos + P] = x
            else:
                B[out_layers[0]][pos:pos + P] = fs
            pos += P
            gtflat.append(frac.astype(np.float16).ravel())
            nm.append(f"{sub}/{f}")
            ty.append("good" if not is_bad else "bad")
            gs.append(grid)
            offs.append(offs[-1] + P)
        offs = np.asarray(offs, dtype=np.int64)
        GT = np.concatenate(gtflat)
        assert len(GT) == offs[-1] and pos == offs[-1]
        # Sanity: anomaly images must actually carry a mask. Zeroing the GT for
        # every anomaly image (which the `sub == "bad"` bug did on MVTec) would
        # otherwise pass silently all the way into the mechanism analysis.
        n_bad = sum(1 for t in ty if t == "bad")
        n_bad_gt = 0
        for i, t in enumerate(ty):
            if t == "bad" and GT[offs[i]:offs[i + 1]].max() > 0:
                n_bad_gt += 1
        if n_bad > 0 and n_bad_gt == 0:
            raise RuntimeError(
                f"{name}: {n_bad} anomaly images but none has a non-zero GT "
                f"mask -- the ground-truth path is wrong.")
        print(f"  {name}: {n_bad_gt}/{n_bad} anomaly images carry a GT mask",
              flush=True)
        for l in out_layers:
            np.savez_compressed(os.path.join(layer_dir, l, f"{nkey}_test.npz"),
                                feats=B[l][:pos], offsets=offs,
                                names=np.array(nm), types=np.array(ty),
                                grids=np.asarray(gs, dtype=np.int32),
                                gt_frac=GT)
        del B
        print(f"  {name:<12} train {len(files):>4} test {len(nm):>4} "
              f"grid {tuple(gs[0])} {time.time() - t0:6.1f}s", flush=True)


def main():
    global DINO_EDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr", required=True, choices=["dino672", "wrn50"])
    ap.add_argument("--dataset", default="both",
                    choices=["both", "mvtec", "visa"])
    ap.add_argument("--edge", type=int, default=0,
                    help="input smaller-edge for DINO; 0 = the repr default")
    ap.add_argument("--objects", default="",
                    help="comma-separated override; default = the M1 smoke set")
    a = ap.parse_args()
    mvt, vis = MVTEC_OBJS, VISA_CATS
    if a.objects:
        want = a.objects.split(",")
        mvt = [o for o in want if os.path.isdir(os.path.join(V1, o))]
        vis = [o for o in want if os.path.isdir(os.path.join(VISA, o))]
    if a.repr == "dino672":
        if a.edge:
            DINO_EDGE = a.edge
        layers, sub = [l for l, _ in DINO_LAYERS], f"cache_m1_dino{DINO_EDGE}"
    else:
        layers, sub = ["l23"], "cache_m1_wrn50"
    base = os.path.join(RESULTS, sub)
    print(f"=== {a.repr} edge={DINO_EDGE} -> {sub} ===")
    if a.dataset in ("both", "mvtec") and mvt:
        run_repr(a.repr, V1, mvt, layers, base)
    if a.dataset in ("both", "visa") and vis:
        run_repr(a.repr, VISA, vis, layers, base)


if __name__ == "__main__":
    main()
