# -*- coding: utf-8 -*-
"""
Stage 5 -- support-normal context fidelity + token-budget feasibility.

Answers ONE question for V3.1: without running the full 672 image, how large
does a halo have to be before the CENTRAL CORE of a local crop matches the
full-672 tokens at the same physical positions?

Strictly support-only.  It reads S_k (the k normal support images of each
(object, shot, split)) and the full-672 feature cache.  It never reads a test
anomaly, a GT mask, AUPRO, px_AUROC or A5 -- the halo size must be decided by
normal-only feature fidelity plus budget arithmetic, never by defect
performance.

Why NN-distance rank agreement and not just cosine: the detector consumes
1-NN DISTANCE, and features can be cosine-similar while their nearest-neighbour
ordering has already changed.

Halo only supplies attention context; only the core is ever written back, so
only the core is scored here.

Usage: OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5_context.py
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from scipy.stats import spearmanr  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, VISA  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_check_scale as cs  # noqa: E402

OUT = os.path.join(RESULTS, "v3_root_cause_diagnosis")
N_CTX = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
CORE = 5                      # 3x3 448 window == 5x5 672 tokens (addendum 3)
ANCHORS = 3                   # sampled 448 anchor windows per support image
CASES = [("mvtec", "bottle", 8), ("mvtec", "capsule", 8),
         ("visa", "pcb2", 1), ("visa", "macaroni1", 1)]


def main():
    os.makedirs(OUT, exist_ok=True)
    model, resize, totensor, norm = cs.build()
    fid, bfe, prof = [], [], []
    for ds, obj, shot in CASES:
        split = 0
        _, c448 = sel.load_split(obj, "train")
        c672 = rf.load_672(obj, "train")
        tro = c448[rf.LAYERS[0]]["offsets"]
        idx = draw_images(tro, shot, split, obj)
        # support bank: the same k images, 672 domain
        o6 = c672[rf.LAYERS[0]]["offsets"]
        bank = {l: l2n(torch.from_numpy(np.concatenate(
            [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
        ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}
        grids = c672[rf.LAYERS[0]]["grids"]
        g = {tuple(int(v) for v in grids[i]) for i in idx}
        if len(g) != 1:
            print(f"  {ds}/{obj}: support grids differ, skip"); continue
        gh6, gw6 = g.pop()
        B = sel.budget((gh6, gw6))

        for si, i in enumerate(idx):
            p = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train",
                             "good", str(c672[rf.LAYERS[0]]["names"][i]))
            big = rf.resized_grid(cv2.imread(p, cv2.IMREAD_COLOR), resize)
            full = [c672[l]["feats"][o6[i]:o6[i + 1]].reshape(gh6, gw6, -1)
                    for l in rf.LAYERS]
            # anchors spread across the valid interior
            for ax in range(ANCHORS):
                for ay in range(ANCHORS):
                    ar = int((ax + 1) / (ANCHORS + 1) * (gh6 - 8))
                    ac = int((ay + 1) / (ANCHORS + 1) * (gw6 - 8))
                    for n in N_CTX:
                        h = n // 2
                        r0 = min(max(0, ar - h), gh6 - n)
                        c0 = min(max(0, ac - h), gw6 - n)
                        crop = rf.crop_feats(model, big, r0, c0, n,
                                             norm, totensor)
                        k0 = (n - CORE) // 2
                        rec = dict(dataset=ds, object=obj, shot=shot,
                                   split=split,
                                   support_image_id=int(i), n_ctx=n,
                                   core_size=CORE, halo_tokens=n * n)
                        cors, sps, fres = [], [], []
                        # `full` is a list (one array per layer); `rf.LAYERS`
                        # holds the layer NAMES, so index it by position, not
                        # by name -- that mismatch crashed the first run.
                        for li, (lay, fc) in enumerate(zip(rf.LAYERS, crop)):
                            a = fc[k0:k0 + CORE, k0:k0 + CORE].reshape(-1, fc.shape[-1])
                            b = full[li][r0 + k0:r0 + k0 + CORE,
                                         c0 + k0:c0 + k0 + CORE].reshape(-1, fc.shape[-1])
                            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
                            bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
                            cos = (an * bn).sum(1)
                            cors.append(cos)
                            fres.append(np.linalg.norm(a - b, axis=1) /
                                        (np.linalg.norm(b, axis=1) + 1e-12))
                            qa = l2n(torch.from_numpy(a.astype(np.float32)).to(DEV))
                            qb = l2n(torch.from_numpy(b.astype(np.float32)).to(DEV))
                            with torch.no_grad():
                                da = nn_dist(qa, bank[lay]).cpu().numpy()
                                db = nn_dist(qb, bank[lay]).cpu().numpy()
                            if len(da) > 1 and np.std(da) > 0 and np.std(db) > 0:
                                sps.append(spearmanr(da, db).statistic)
                            rec[f"nn_rel_err_{lay}"] = float(
                                np.median(np.abs(da - db) / (np.abs(db) + 1e-12)))
                        c = np.concatenate(cors)
                        f = np.concatenate(fres)
                        rec.update(cos_mean=float(c.mean()),
                                   cos_median=float(np.median(c)),
                                   cos_p10=float(np.percentile(c, 10)),
                                   feature_rel_error=float(np.median(f)),
                                   nn_spearman=float(np.mean(sps)) if sps else np.nan,
                                   layer="all3")
                        fid.append(rec)
        # budget feasibility, label-free: only trigger counts from diagnostics
        v3d = os.path.join(RESULTS, "v3", ds, obj, f"k{shot}", f"s{split}")
        ms = []
        for r, _, f in os.walk(v3d):
            if "diagnostics.json" in f:
                dg = json.load(open(os.path.join(r, "diagnostics.json")))
                ms += [x["m"] for x in dg.get("a3", [])]
        if ms:
            ms = np.array(ms)
            for n in N_CTX:
                K = int(B // (n * n))
                bfe.append(dict(dataset=ds, object=obj, shot=shot, split=split,
                                n_ctx=n, B=B, Kmax=K,
                                mean_trigger_m=float(ms.mean()),
                                median_trigger_m=float(np.median(ms)),
                                pct_images_capped=float((ms > K).mean() * 100),
                                mean_retained_fraction=float(
                                    np.minimum(ms, K).sum() / max(ms.sum(), 1)),
                                token_utilization=float(
                                    np.minimum(ms, K).sum() * n * n / (len(ms) * B))))
                prof.append(dict(dataset=ds, object=obj, shot=shot, split=split,
                                 n_images=len(ms), m_mean=float(ms.mean()),
                                 m_max=int(ms.max()), m_zero=int((ms == 0).sum())))
        print(f"  {ds}/{obj} k={shot}: support {sorted(int(i) for i in idx)}  "
              f"grid {gh6}x{gw6}  B={B:.0f}", flush=True)

    import pandas as pd
    pd.DataFrame(fid).to_csv(os.path.join(OUT, "05_context_fidelity_support.csv"),
                             index=False)
    pd.DataFrame(bfe).to_csv(os.path.join(OUT, "05_context_budget_feasibility.csv"),
                             index=False)
    pd.DataFrame(prof).to_csv(os.path.join(OUT, "V3_TRIGGER_PROFILE.csv"),
                              index=False)
    d = pd.DataFrame(fid)
    if len(d):
        print("\n  === context fidelity vs n_ctx (median over samples) ===")
        agg = d.groupby("n_ctx")[["cos_median", "cos_p10", "nn_spearman"]].median()
        print(agg.round(4).to_string())
    print(f"\n  wrote 05_context_fidelity_support.csv ({len(fid)}) "
          f"05_context_budget_feasibility.csv ({len(bfe)}) "
          f"V3_TRIGGER_PROFILE.csv ({len(prof)})")


if __name__ == "__main__":
    main()
