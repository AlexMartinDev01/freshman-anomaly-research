# -*- coding: utf-8 -*-
"""
Stage 4 -- fixed-position 2x2 causal experiment on capsule/k1/s0.

Everything frozen EXCEPT two orthogonal factors:

    feature source : crop-DINO  vs  full-672 tokens at the SAME physical
                     positions.  A crop window is a token-aligned sub-window of
                     the globally resized image, so the full 672 pass already
                     produced exactly those tokens -- C10/C11 need no forward.
    normalization  : crop-local z-score (current implementation) vs the
                     A2-consistent per-layer global z-score (pooled over the
                     whole test set, which is exactly what A2 is built from).

    C00  crop-DINO + local    (current implementation)
    C10  full-672  + local    -> isolates "the crop loses global context"
    C01  crop-DINO + global   -> isolates "normalization scope"
    C11  full-672  + global   -> both factors removed

Held identical across all four: selector, positions, m', n, budget, support
bank, physical-overlap 672->448 pooling, affine, writeback.

Single process, no timing, output only under v3_root_cause_diagnosis/.
Never touches results/model_v0/v3/ and never changes HEAD.

Usage: python experiments/model_v0/v3_stage4_2x2.py
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, gt_file  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_run as vr  # noqa: E402
import v3_check_scale as cs  # noqa: E402

CELL = os.environ.get("RC_CELL", "mvtec/capsule/k1/s0")
OUT = os.path.join(RESULTS, "v3_root_cause_diagnosis")
VARIANTS = ["C00", "C10", "C01", "C11"]


def flat(a):
    return a.reshape(-1, a.shape[-1])


def main():
    ds, obj, kk, ss = CELL.split("/")
    shot, split = int(kk[1:]), int(ss[1:])
    d = os.path.join(RESULTS, "v3", *CELL.split("/"))
    diag = json.load(open(os.path.join(d, "diagnostics.json")))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    a_, b_ = meta["affine"]["a"], meta["affine"]["b"]
    # Stage 4B: RC_NO_AFFINE=1 sets a=1, b=0 so the 2x2 is read in the RAW
    # score domain.  Stage 4 kept the frozen affine on all four arms, so its
    # C01-C00 mixes the normalization change with whatever the old affine does
    # once the domain moves.  4B separates those.  Same code path, one switch.
    if os.environ.get("RC_NO_AFFINE") == "1":
        a_, b_ = 1.0, 0.0
        print("  RC_NO_AFFINE=1 -> affine disabled (a=1, b=0), raw domain")
    a0 = np.load(os.path.join(d, "a0.npy"))
    os.makedirs(OUT, exist_ok=True)
    print(f"cell {CELL}   affine a={a_:.5f} b={b_:.5f}")

    _, c448 = sel.load_split(obj, "train")
    _, te448 = sel.load_split(obj, "test")
    c672 = rf.load_672(obj, "train")
    te672 = rf.load_672(obj, "test")
    idx = draw_images(c448[rf.LAYERS[0]]["offsets"], shot, split, obj)
    print(f"  support ids = {sorted(int(i) for i in idx)}  (S_k)")
    o6 = c672[rf.LAYERS[0]]["offsets"]
    bank = {l: l2n(torch.from_numpy(np.concatenate(
        [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
    ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}

    teo = te672[rf.LAYERS[0]]["offsets"]
    names = [str(x) for x in te448[rf.LAYERS[0]]["names"]]
    types = [str(x) for x in te448[rf.LAYERS[0]]["types"]]
    g448 = te448[rf.LAYERS[0]]["grids"]
    y = np.array([0 if t == "good" else 1 for t in types])

    print("  A2-consistent global stats (pooled per-layer 672 distances) ...")
    gstat = {}
    for l in rf.LAYERS:
        vals = []
        for i in range(len(names)):
            q = l2n(torch.from_numpy(
                te672[l]["feats"][teo[i]:teo[i + 1]].astype(np.float32)).to(DEV))
            with torch.no_grad():
                vals.append(nn_dist(q, bank[l]).cpu().numpy())
        allv = np.concatenate(vals)
        gstat[l] = (float(allv.mean()), float(allv.std() + 1e-12))
        del vals
    print(f"    mu/sd per layer: "
          f"{ {l: (round(gstat[l][0], 3), round(gstat[l][1], 3)) for l in rf.LAYERS} }")

    model, resize, totensor, norm = cs.build()
    Ms = {k: np.array(a0, copy=True) for k in VARIANTS}
    n_ref = 0
    for i, nm in enumerate(names):
        rec = diag["a3"][i]
        if rec["m2"] == 0:
            continue
        n = int(rec["n"])
        gh, gw = int(g448[i][0]), int(g448[i][1])
        wg, ww = gw - 2, gh - 2                     # window-grid extent
        src = os.path.join(V1, obj, "test", *nm.split("/"))
        big = rf.resized_grid(cv2.imread(src, cv2.IMREAD_COLOR), resize)
        gh6, gw6 = big.size[1] // rf.STRIDE, big.size[0] // rf.STRIDE
        full = [te672[l]["feats"][teo[i]:teo[i + 1]]
                .reshape(gh6, gw6, -1) for l in rf.LAYERS]
        acc = {k: {} for k in VARIANTS}
        for p in rec.get("pos", []):
            p = int(p)
            pr, pc = p // wg, p % wg
            r0, c0 = rf.window_672(pr, pc, n, gh6, gw6)
            crop = rf.crop_feats(model, big, r0, c0, n, norm, totensor)
            for tag, feats in (("C0", crop),
                               ("C1", [f[r0:r0 + n, c0:c0 + n] for f in full])):
                d_loc, d_glo = [], []
                for lay, f in zip(rf.LAYERS, feats):
                    q = l2n(torch.from_numpy(flat(f).astype(np.float32)).to(DEV))
                    with torch.no_grad():
                        dd = nn_dist(q, bank[lay]).cpu().numpy()
                    d_loc.append((dd - dd.mean()) / (dd.std() + 1e-12))
                    mu, sd = gstat[lay]
                    d_glo.append((dd - mu) / sd)
                loc = np.mean(d_loc, axis=0).reshape(n, n)
                glo = np.mean(d_glo, axis=0).reshape(n, n)
                for vname, cmap in ((f"{tag}0", loc), (f"{tag}1", glo)):
                    V = {"C0": "C00", "C1": "C10"} if vname.endswith("0") else \
                        {"C0": "C01", "C1": "C11"}
                    k = V[tag]
                    for kk_, vv in vr.pool_crop_to_448(cmap, r0, c0, gh, gw,
                                                       gh6, gw6).items():
                        acc[k].setdefault(kk_, []).append(vv)
        for k in VARIANTS:
            for (r, c), vs in acc[k].items():
                Ms[k][i][r, c] = a_ * float(np.mean(vs)) + b_
        n_ref += 1
        if n_ref % 10 == 0:
            print(f"    refined images done: {n_ref}", flush=True)
    print(f"  refined images: {n_ref}")

    rows, tmpf = [], {}
    for k in VARIANTS:
        if not np.isfinite(Ms[k]).all():
            raise SystemExit(f"{k}: non-finite map")
        sub = os.path.join(OUT, f"stage4_{k}")
        os.makedirs(sub, exist_ok=True)
        jobs = []
        for i, nm in enumerate(names):
            p = os.path.join(sub, f"{i}.npy")
            np.save(p, Ms[k][i])
            im = cv2.imread(os.path.join(V1, obj, "test", *nm.split("/")),
                            cv2.IMREAD_COLOR)
            gp = None
            if y[i]:
                gp = gt_file(V1, obj, nm)
                if gp is not None:
                    g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                    if g is None or (g > 0).sum() == 0:
                        gp = None
            jobs.append((p, gp, im.shape[:2]))
        mm = pixel_metrics_binned(jobs, pro_limit=0.05)
        sc = np.array([float(m.mean()) for m in Ms[k]])
        rows.append(dict(cell=CELL, variant=k,
                         img_AUROC=roc_auc_score(y, sc) * 100,
                         px_AUROC=mm["px_AUROC"] * 100,
                         AUPRO=mm["AUPRO"] * 100))
    import pandas as pd
    df = pd.DataFrame(rows)
    tag = "04b_stage4_pre_affine_2x2" if os.environ.get("RC_NO_AFFINE") == "1" \
        else "04_stage4_2x2"
    df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    base = df.set_index("variant")
    print("\n  === Stage 4: fixed-position 2x2 ===")
    for k in VARIANTS:
        print(f"    {k}  img {base.loc[k,'img_AUROC']:7.2f}  "
              f"px {base.loc[k,'px_AUROC']:7.2f}  AUPRO {base.loc[k,'AUPRO']:7.2f}")
    a0v = base.loc["C00", "AUPRO"]
    print(f"\n    A0 reference AUPRO (from 01_metric_recheck.csv): see below")
    for lab, k in (("context (C10-C00)", "C10"), ("norm    (C01-C00)", "C01"),
                   ("both    (C11-C00)", "C11")):
        print(f"    {lab}  {base.loc[k,'AUPRO'] - a0v:+8.2f}")
    print(f"\n  wrote {os.path.join(OUT, '04_stage4_2x2.csv')}")


if __name__ == "__main__":
    main()
