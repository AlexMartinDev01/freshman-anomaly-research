# -*- coding: utf-8 -*-
"""R6-A1B -- Selection Error & Defect-Halo Audit.

Implements the frozen R6-A1B spec (the kit itself never reached this machine).

Three-layer oracle ladder, SAME held-out images, 1NN scores:

    L1  fixed 1%          bad: mean of its top round(.01*N)
                          good: mean of its own top round(.01*N)
    L2  extent oracle     bad: mean of its top m,  m = THAT image's GT defect count
                          good: mean of its own top m  (paired to the same m)
    L3  location oracle   bad: mean over its ACTUAL GT defect patches
                          good: mean of its own top m  (paired to the same m)

    L1 -> L2  quantifies tail-WIDTH mismatch (unknown anomaly support)
    L2 -> L3  quantifies tail-SELECTION error (top-m still taken by clean patches)

Halo audit: among the GT==0 patches selected into the top-alpha tail, how many
sit within r patches of a real GT defect, versus the same count drawn at random
from that image's clean pool (200 draws)?  A positive delta means the tail's
"clean" patches are defect halo/context, not ordinary normal nuisance.

Diagnostic only.  No method, no tuning, no threshold changes.

Usage
    python r6a1b_selection_halo/r6a1b_selection_halo.py            # 1NN arms
    python r6a1b_selection_halo/r6a1b_selection_halo.py --probe    # + probe arms
"""
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "model_v0"))
sys.path.insert(0, str(ROOT / "experiments" / "baseline"))
sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDINO"))

from phase_m1_rescue import load_obj                          # noqa: E402
from gate_r2_layer_confirm import draw_images, l2n, nn_dist    # noqa: E402

DEV = "cuda"
GT_THR = 0.10
ALPHAS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
RADII = [1, 2, 4]
N_RANDOM = 200
SEED = 20260930
SELECTION = ROOT / "r6a_supervised_smoke" / "R6A_SMOKE_SELECTION.csv"
OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1b_selection_halo"


def topk_mean(v, k):
    k = int(min(max(k, 1), len(v)))
    return float(np.partition(v, len(v) - k)[-k:].mean())


def kmax(v, alpha):
    return max(1, int(round(alpha * len(v))))


def superior_frac(bad_score, good_scores):
    g = np.asarray(good_scores)
    return float(((g < bad_score).sum() + 0.5 * (g == bad_score).sum()) / len(g))


def make_bank(c, ids):
    o = c["offsets"]
    return l2n(torch.from_numpy(np.concatenate(
        [c["feats"][int(o[i]):int(o[i + 1])] for i in ids]).astype(np.float32)).to(DEV))


def load_object(obj, shot, split):
    _, _, tr, te = load_obj("r0", obj)
    trc, tec = tr["final"], te["final"]
    ids = [int(v) for v in draw_images(trc["offsets"], shot, split, obj)]
    bank = make_bank(trc, ids)
    o = np.asarray(tec["offsets"])
    maps, labs, grids = [], [], []
    for i in range(len(tec["types"])):
        x = tec["feats"][int(o[i]):int(o[i + 1])].astype(np.float32)
        with torch.inference_mode():
            maps.append(nn_dist(l2n(torch.from_numpy(x).to(DEV)), bank)
                        .cpu().numpy().astype(np.float64))
        gt = np.asarray(tec["gt_frac"][int(o[i]):int(o[i + 1])], float)
        y = np.full(len(gt), -1, np.int8)
        y[gt == 0] = 0
        y[gt > GT_THR] = 1
        labs.append(y)
        grids.append(tuple(int(v) for v in tec["grids"][i]))
    del bank
    return maps, labs, grids, np.asarray([str(v) for v in tec["types"]])


def halo_hits(sel_idx, G, shape, r):
    """Count of sel_idx patches within Chebyshev r of any index in G."""
    if not len(sel_idx):
        return 0
    R, C = shape
    sr, sc = np.divmod(sel_idx, C)
    gr, gc = np.divmod(G, C)
    d = np.maximum(np.abs(sr[:, None] - gr[None, :]), np.abs(sc[:, None] - gc[None, :]))
    return int((d <= r).any(axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(a.selection)

    ladder, halo_rows, alpha_rows = [], [], []
    for _, row in sel.iterrows():
        obj, shot, split = str(row.object), int(row.shot), int(row["split"])
        t0 = time.perf_counter()
        maps, labs, grids, types = load_object(obj, shot, split)
        yimg = (types == "bad").astype(int)
        good = np.where(yimg == 0)[0]
        bad = np.where(yimg == 1)[0]
        n_def = {i: int((labs[i] == 1).sum()) for i in range(len(types))}
        R, C = grids[0]
        if any(g != (R, C) for g in grids):
            raise SystemExit(f"{obj}: token grid varies across images")

        # ---- three-layer ladder -------------------------------------------
        G1 = np.array([topk_mean(maps[j], kmax(maps[j], 0.01)) for j in good])
        acc = {"L1_fixed_1pct": [], "L2_extent_oracle": [], "L3_gt_location_oracle": []}
        for i in bad:
            m = n_def[i]
            if m < 1:
                continue
            G2 = np.array([topk_mean(maps[j], m) for j in good])
            acc["L1_fixed_1pct"].append(
                superior_frac(topk_mean(maps[i], kmax(maps[i], 0.01)), G1))
            acc["L2_extent_oracle"].append(superior_frac(topk_mean(maps[i], m), G2))
            acc["L3_gt_location_oracle"].append(
                superior_frac(float(maps[i][labs[i] == 1].mean()), G2))
        for k, v in acc.items():
            if v:
                ladder.append(dict(object=obj, layer=k, n_bad=len(v),
                                   pairwise_auc=float(np.mean(v) * 100)))

        # ---- halo audit ----------------------------------------------------
        # stable per-object seed -- Python's str hash is randomised per process,
        # so hash(obj) would make the 200-draw null irreproducible across runs.
        seed_obj = int(zlib.crc32(obj.encode("utf-8")) % 100000)
        rng = np.random.default_rng(SEED + seed_obj)
        for alpha in ALPHAS:
            k = kmax(maps[good[0]], alpha)
            for r_ in RADII:
                tail_ok, rand_ok, n_tot = 0, 0.0, 0
                for i in bad:
                    G = np.where(labs[i] == 1)[0]
                    if not len(G):
                        continue
                    idx = np.argpartition(maps[i], len(maps[i]) - k)[-k:]
                    clean_sel = idx[labs[i][idx] == 0]
                    if not len(clean_sel):
                        continue
                    tail_ok += halo_hits(clean_sel, G, (R, C), r_)
                    pool = np.where(labs[i] == 0)[0]
                    reps = [halo_hits(rng.choice(pool, min(len(clean_sel), len(pool)),
                                                 replace=False), G, (R, C), r_) / len(clean_sel)
                            for _ in range(N_RANDOM)]
                    rand_ok += float(np.mean(reps)) * len(clean_sel)
                    n_tot += len(clean_sel)
                if n_tot:
                    halo_rows.append(dict(
                        object=obj, alpha=alpha, radius=r_, n_clean_selected=n_tot,
                        tail_enrichment=tail_ok / n_tot,
                        random_enrichment=rand_ok / n_tot,
                        enrichment_delta=(tail_ok - rand_ok) / n_tot))

            # tail purity on bad images at this alpha (context for the ladder)
            nd = nc = 0
            for i in bad:
                idx = np.argpartition(maps[i], len(maps[i]) - k)[-k:]
                nd += int((labs[i][idx] == 1).sum())
                nc += int((labs[i][idx] == 0).sum())
            alpha_rows.append(dict(object=obj, alpha=alpha, k=k,
                                   selected_defect=nd, selected_clean=nc,
                                   gt_precision=nd / max(nd + nc, 1)))

        print(f"  {obj} done ({time.perf_counter()-t0:.1f}s)", flush=True)
        pd.DataFrame(ladder).to_csv(out / "r6a1b_oracle_ladder.csv", index=False)
        pd.DataFrame(halo_rows).to_csv(out / "r6a1b_halo_enrichment.csv", index=False)
        pd.DataFrame(alpha_rows).to_csv(out / "r6a1b_tail_purity.csv", index=False)
    print("DONE", out)


if __name__ == "__main__":
    main()
