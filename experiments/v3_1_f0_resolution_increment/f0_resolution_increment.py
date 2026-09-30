# -*- coding: utf-8 -*-
"""V3.1-F0 -- support-normal feasibility of the resolution-increment surrogate.

Stage 5/5D closed the halo route: a local 672 crop cannot re-acquire the
full-image context.  F0 tests the opposite decomposition --

    full448 supplies the GLOBAL CONTEXT
    local672 supplies only the RESOLUTION INCREMENT

by asking whether a cheap surrogate can predict the full-672 score geometry
without ever running a full 672 forward at test time.

At the SAME physical location, for each layer l:

    d_G448   full448  raw 1-NN distance      (global context, 448 resolution)
    d_L448   crop448  raw 1-NN distance      (local context,  448 resolution)
    d_L672   crop672  raw 1-NN distance      (local context,  672 resolution)
    d_G672   full672  raw 1-NN distance      (support-normal REFERENCE only)

    Delta_l   = d_L672_l - phi_l(d_L448_l)
    dhat_G672_l = psi_l(d_G448_l) + Delta_l

phi_l and psi_l are OPTIONAL AFFINE fits on support-normal only -- no MLP, no
attention, no adapter, no nonlinear search, no hyper-parameter sweep beyond the
three fixed scales.  The premise is difference-in-differences: crop448 and
crop672 suffer a SIMILAR context truncation, so their difference is closer to a
pure resolution gain than either term alone.

Arms compared against d_G672:
    B0 = d_L672                       (local672 alone -- the closed halo idea)
    B1 = psi(d_G448)                  (global448 alone, affinely mapped)
    B2 = psi(d_G448) + Delta          (global448 + resolution increment)

SUPPORT-NORMAL ONLY.  No test anomaly, no GT, no AUPRO/AUROC, no A5, no
selector, no 324-cell run, no defect performance of any kind.

Cross-fitting: phi/psi are fitted on half the support images and applied to the
other half (2-fold, by support-image parity), because the same support data is
used for fitting and for scoring.  An affine fit on ~10^3 points barely
overfits, but the asymmetry would otherwise favour B1/B2 over B0.

Usage: OMP_NUM_THREADS=2 python experiments/v3_1_f0_resolution_increment/f0_resolution_increment.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from scipy.stats import spearmanr  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, VISA  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_calibration as cal  # noqa: E402
import v3_check_scale as cs  # noqa: E402

OUT = os.path.join(RESULTS, "v3_1_f0_resolution_increment")
SCALES = [9, 13, 15]          # 672-crop side, FIXED -- no sweep beyond these
CORE448 = 3                   # probe = 3x3 448 patches == 5x5 672 tokens
CORE672 = CORE448 * 5 // 3    # 5
ANCH = 3                      # ANCH x ANCH probe positions per support image
SHOT, SPLIT = 8, 0

# ---- pre-registered GO gate (fixed BEFORE this script was run) ----
GATE_RHO = 0.50               # median_category rho(B2)
GATE_GAIN = 0.05              # median_category [rho(B2) - rho(B1)]
GATE_B2_GT_B0 = True          # B2 must beat B0 everywhere

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def side448(n):
    """448-crop side for a 672-crop side n: the nearest ODD window that can
    centre the shared 3x3-448 / 5x5-672 probe symmetrically.

    The two token grids are not commensurate (a 448 token spans 1.5 672
    tokens), so an exact physical match is impossible.  The achieved extents
    are recorded per row (physical_px_448 / physical_px_672) rather than
    smoothed over: at n=9,13,15 the 448 crop is 17% / 3.8% / 10% wider.
    """
    return CORE448 + 2 * int(round((n / 1.5 - CORE448) / 2.0))


def probe_positions(gh4, gw4, gh6, gw6):
    """448 anchors at which EVERY scale fits, unclipped, in BOTH domains."""
    out = []
    for ar in range(1, gh4 - 1):
        for ac in range(1, gw4 - 1):
            for n in SCALES:
                N, p4 = side448(n), (side448(n) - CORE448) // 2
                r0, c0 = ar - 1 - p4, ac - 1 - p4
                if r0 < 0 or r0 + N > gh4 or c0 < 0 or c0 + N > gw4:
                    break
                cy, cx = int(np.floor(1.5 * ar + 0.75)), int(np.floor(1.5 * ac + 0.75))
                h = (n - 1) // 2
                if cy - h < 0 or cy - h + n > gh6 or cx - h < 0 or cx - h + n > gw6:
                    break
            else:
                out.append((ar, ac))
    return out


def spread(items, k):
    """k evenly spaced picks from a list, deterministically."""
    if len(items) <= k:
        return list(items)
    span = len(items) - 1
    return [items[span * (i + 1) // (k + 1)] for i in range(k)]


def anchor_grid(valid, k=ANCH):
    """k x k anchor centres: spread over the distinct rows and columns of the
    valid set, keeping only combinations that are themselves valid."""
    ars = spread(sorted({a for a, _ in valid}), k)
    acs = spread(sorted({c for _, c in valid}), k)
    out = [(a, c) for a in ars for c in acs if (a, c) in set(valid)]
    if len(out) < len(ars) * len(acs):
        raise SystemExit(f"anchor grid incomplete: {len(out)} of {len(ars)*len(acs)}")
    return out


def local_weights(i, j):
    """(rows, cols, area weights) of the 672 tokens overlapping 448 patch
    (i, j).  Bounds-restricted, but the arithmetic is the frozen cal._span /
    cal._overlap -- verified against cal.pool_weights in selftest()."""
    y0, y1 = cal._span(i, cal.T448)
    x0, x1 = cal._span(j, cal.T448)
    rows, cols, w = [], [], []
    for u in range(int(y0 // cal.T672) - 1, int(y1 // cal.T672) + 2):
        oy = cal._overlap(*cal._span(u, cal.T672), y0, y1)
        if oy <= 0:
            continue
        for v in range(int(x0 // cal.T672) - 1, int(x1 // cal.T672) + 2):
            ox = cal._overlap(*cal._span(v, cal.T672), x0, x1)
            if ox <= 0:
                continue
            rows.append(u); cols.append(v); w.append(oy * ox)
    return np.array(rows, int), np.array(cols, int), np.array(w, float)


def fit_affine(x, y):
    """y ~ a*x + b by least squares.  Direction is FIXED: map the 448-domain
    quantity onto the 672-domain one.  Never reversed.  Affine only."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    if not np.isfinite(a) or not np.isfinite(b):
        raise SystemExit("affine fit produced non-finite coefficients")
    return float(a), float(b)


def selftest():
    """local_weights must agree with the frozen cal.pool_weights.

    Sampled over REAL 448 patches of a 32x32 / 48x48 pair.  cal.pool_weights is
    bounded by the 672 grid; a 448 patch index >= 32 has no business existing in
    a 32x32 grid and its footprint would fall outside the 672 grid, so feeding
    one in would test the bounds rather than the arithmetic.
    """
    import random
    random.seed(0)
    for _ in range(40):
        i, j = random.randrange(0, 32), random.randrange(0, 32)
        r1, c1, w1 = local_weights(i, j)
        r2, c2, w2 = cal.pool_weights(i, j, 48, 48)
        assert set(zip(r1.tolist(), c1.tolist())) == set(zip(r2.tolist(), c2.tolist())), (i, j)
        m1 = {(a, b): w for a, b, w in zip(r1, c1, w1)}
        for a, b, w in zip(r2, c2, w2):
            assert abs(m1[(a, b)] - w) < 1e-9, (i, j, a, b)
    print("  selftest: local_weights == cal.pool_weights (40 random patches) OK")


def positive_control(model, norm, totensor, resize448, resize672):
    """Both domains must reproduce their caches bit-exactly, or every crop
    discrepancy below would be uninterpretable."""
    rows = []
    for ds, obj in (("mvtec", "bottle"), ("visa", "pcb2")):
        c448 = sel.load_split(obj, "train")[1]
        c672 = rf.load_672(obj, "train")
        p0 = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train", "good",
                          str(c448[rf.LAYERS[0]]["names"][0]))
        img = cv2.imread(p0, cv2.IMREAD_COLOR)
        for dom, bigr, cache, offkey in (
                ("448", resize448, c448, "offsets"), ("672", resize672, c672, "offsets")):
            big = rf.resized_grid(img, bigr)
            with torch.no_grad():
                tk = model.model.get_intermediate_layers(
                    norm(totensor(big)).unsqueeze(0).to(model.device), n=[5, 8, 11])
            for li, lay in enumerate(rf.LAYERS):
                o = cache[lay][offkey]
                ck = cache[lay]["feats"].shape[-1]
                a = tk[li].squeeze(0).cpu().numpy().astype(np.float16) \
                    .astype(np.float32).reshape(-1, ck)
                b = cache[lay]["feats"][o[0]:o[1]].astype(np.float32)
                an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
                bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
                cos = float((an * bn).sum(1).mean())
                mad = float(np.abs(a - b).max())
                rows.append(dict(dataset=ds, object=obj, domain=dom, layer=lay,
                                 cos_mean=cos, max_abs_diff=mad,
                                 pass_=bool(cos > 0.999 and mad < 0.05)))
                print(f"  {ds}/{obj:8s} {dom} {lay:8s} cos={cos:.6f} "
                      f"maxdiff={mad:.5f} {'OK' if rows[-1]['pass_'] else 'FAIL'}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "f0_positive_control.csv"), index=False)
    if not df.pass_.all():
        raise SystemExit("POSITIVE CONTROL FAILED -- a domain does not reproduce "
                         "its cache; crop distances would be uninterpretable.")


def main():
    os.makedirs(OUT, exist_ok=True)
    selftest()
    model, resize672, totensor, norm = cs.build()
    resize448 = transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC,
                                  antialias=True)
    print("=== positive control: both domains vs their caches ===")
    positive_control(model, norm, totensor, resize448, resize672)

    recs = []
    for objs, ds in ((MVTEC, "mvtec"), (VISA_CATS, "visa")):
        for obj in objs:
            c448 = sel.load_split(obj, "train")[1]
            c672 = rf.load_672(obj, "train")
            o4 = {l: c448[l]["offsets"] for l in rf.LAYERS}
            o6 = {l: c672[l]["offsets"] for l in rf.LAYERS}
            g448 = {l: tuple(int(v) for v in c448[l]["grids"][0]) for l in rf.LAYERS}
            g672 = {l: tuple(int(v) for v in c672[l]["grids"][0]) for l in rf.LAYERS}
            if len(set(g448.values())) != 1 or len(set(g672.values())) != 1:
                print(f"  {ds}/{obj}: grids differ across layers, skip"); continue
            gh4, gw4 = g448[rf.LAYERS[0]]
            gh6, gw6 = g672[rf.LAYERS[0]]
            idx = draw_images(o4[rf.LAYERS[0]], SHOT, SPLIT, obj)
            if len({tuple(int(v) for v in c448[rf.LAYERS[0]]["grids"][i]) for i in idx}) != 1 \
               or len({tuple(int(v) for v in c672[rf.LAYERS[0]]["grids"][i]) for i in idx}) != 1:
                print(f"  {ds}/{obj}: support grids differ, skip"); continue
            pos = anchor_grid(probe_positions(gh4, gw4, gh6, gw6))
            if len(pos) < ANCH * ANCH:
                print(f"  {ds}/{obj}: only {len(pos)} usable anchors, skip"); continue
            # leave-one-image-out banks: the probe image is never in its own bank
            feat4 = {l: c448[l]["feats"] for l in rf.LAYERS}
            feat6 = {l: c672[l]["feats"] for l in rf.LAYERS}
            for si, i in enumerate(idx):
                others = [j for j in idx if j != i]
                bank4 = {l: l2n(torch.from_numpy(np.concatenate(
                    [feat4[l][o4[l][j]:o4[l][j + 1]] for j in others]
                ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}
                bank6 = {l: l2n(torch.from_numpy(np.concatenate(
                    [feat6[l][o6[l][j]:o6[l][j + 1]] for j in others]
                ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}
                img = cv2.imread(os.path.join(
                    V1 if ds == "mvtec" else VISA, obj, "train", "good",
                    str(c448[rf.LAYERS[0]]["names"][i])), cv2.IMREAD_COLOR)
                big4, big6 = rf.resized_grid(img, resize448), rf.resized_grid(img, resize672)
                full4 = {l: feat4[l][o4[l][i]:o4[l][i + 1]].reshape(gh4, gw4, -1)
                         for l in rf.LAYERS}
                full6 = {l: feat6[l][o6[l][i]:o6[l][i + 1]].reshape(gh6, gw6, -1)
                         for l in rf.LAYERS}
                for (ar, ac) in pos:
                    cy = int(np.floor(1.5 * ar + 0.75))
                    cx = int(np.floor(1.5 * ac + 0.75))
                    for n in SCALES:
                        N, p4 = side448(n), (side448(n) - CORE448) // 2
                        h = (n - 1) // 2
                        r4, c4 = ar - 1 - p4, ac - 1 - p4
                        r6, c6 = cy - h, cx - h
                        crop4 = rf.crop_feats(model, big4, r4, c4, N, norm, totensor)
                        crop6 = rf.crop_feats(model, big6, r6, c6, n, norm, totensor)
                        # probe region: centre 3x3 of the 448 crop == centre 5x5 of the 672 crop
                        q4 = p4
                        q6 = (n - CORE672) // 2
                        for dr in range(CORE448):
                            for dc in range(CORE448):
                                R, C = r4 + q4 + dr, c4 + q4 + dc
                                rows6, cols6, w6 = local_weights(R, C)
                                lr, lc = rows6 - r6, cols6 - c6
                                if lr.min() < 0 or lr.max() >= n or lc.min() < 0 or lc.max() >= n:
                                    raise AssertionError(
                                        f"probe 448 patch ({R},{C}) maps outside the "
                                        f"672 crop (offsets {lr.min()}..{lr.max()})")
                                for l in rf.LAYERS:
                                    def d_of(feat, bank, rr, cc):
                                        v = feat[rr, cc].reshape(1, -1)
                                        q = l2n(torch.from_numpy(v.astype(np.float32)).to(DEV))
                                        with torch.no_grad():
                                            return float(nn_dist(q, bank).cpu().numpy()[0])
                                    dG448 = d_of(full4[l], bank4[l], R, C)
                                    dL448 = d_of(crop4[rf.LAYERS.index(l)], bank4[l], q4 + dr, q4 + dc)
                                    dl6 = crop6[rf.LAYERS.index(l)]
                                    dL672 = float(np.average(
                                        [float(nn_dist(l2n(torch.from_numpy(
                                            dl6[u, v].reshape(1, -1).astype(np.float32)).to(DEV)),
                                            bank6[l]).cpu().numpy()[0])
                                         for u, v in zip(lr, lc)], weights=w6))
                                    dG672 = float(np.average(
                                        [float(nn_dist(l2n(torch.from_numpy(
                                            full6[l][u, v].reshape(1, -1).astype(np.float32)).to(DEV)),
                                            bank6[l]).cpu().numpy()[0])
                                         for u, v in zip(rows6, cols6)], weights=w6))
                                    recs.append(dict(
                                        dataset=ds, object=obj, support_id=int(i),
                                        fold=si % 2, anchor_r=ar, anchor_c=ac,
                                        patch_r=R, patch_c=C, n=n, layer=l,
                                        physical_px_448=N * 14, physical_px_672=n * 14,
                                        d_G448=dG448, d_L448=dL448,
                                        d_L672=dL672, d_G672=dG672))
            print(f"  {ds}/{obj} done (grid448 {gh4}x{gw4}, grid672 {gh6}x{gw6}, "
                  f"anchors {pos})", flush=True)
            pd.DataFrame(recs).to_csv(os.path.join(OUT, "_partial_f0.csv"), index=False)

    df = pd.DataFrame(recs)
    if not len(df):
        raise SystemExit("no probes collected")
    df.to_csv(os.path.join(OUT, "f0_probe_distances.csv"), index=False)
    part = os.path.join(OUT, "_partial_f0.csv")
    if os.path.exists(part):
        os.remove(part)
    per_layer, agg3 = analyse(df)
    per_layer.to_csv(os.path.join(OUT, "f0_per_layer_metrics.csv"), index=False)
    agg3.to_csv(os.path.join(OUT, "f0_agg3_metrics.csv"), index=False)
    decide(agg3)
    print(f"\n  wrote f0_* ({len(df)} probe samples, {len(per_layer)} per-layer rows, "
          f"{len(agg3)} agg3 rows)")


def build_arms(g, n, layer):
    """B0/B1/B2 for one (dataset, object, layer, n), phi/psi cross-fitted by
    support-image parity so the fit never scores its own samples."""
    a0 = np.full(len(g), np.nan)
    a1 = np.full(len(g), np.nan)
    a2 = np.full(len(g), np.nan)
    for test_fold in (0, 1):
        fit = g[g.fold != test_fold]
        ev = g.fold == test_fold
        if not len(fit) or not ev.any():
            raise SystemExit(f"{layer}/n={n}: a cross-fit fold is empty")
        phi = fit_affine(fit.d_L448, fit.d_L672)
        psi = fit_affine(fit.d_G448, fit.d_G672)
        dL672, dG448, dL448 = g.d_L672.values[ev], g.d_G448.values[ev], g.d_L448.values[ev]
        delta = dL672 - (phi[0] * dL448 + phi[1])
        mapped = psi[0] * dG448 + psi[1]
        a0[ev], a1[ev], a2[ev] = dL672, mapped, mapped + delta
    return a0, a1, a2


def analyse(df):
    """Per-layer and agg3 Spearman / MAE / relative error for B0/B1/B2.

    agg3 mirrors the detector's aggregation: z-score each layer with the
    REFERENCE (d_G672) per-layer statistics, apply the same transform to every
    arm, then average over the three layers.
    """
    per_layer, agg_rows = [], []
    for (ds, obj, n), g in df.groupby(["dataset", "object", "n"]):
        arms = {}
        for layer in rf.LAYERS:
            gl = g[g.layer == layer]
            a0, a1, a2 = build_arms(gl, n, layer)
            y = gl.d_G672.values
            for name, a in (("B0", a0), ("B1", a1), ("B2", a2)):
                arms[(layer, name)] = a
                per_layer.append(dict(
                    dataset=ds, object=obj, n=n, layer=layer, arm=name,
                    n_samples=len(gl), spearman=float(spearmanr(a, y).statistic),
                    mae=float(np.mean(np.abs(a - y))),
                    rel_error=float(np.mean(np.abs(a - y) / (np.abs(y) + 1e-12)))))
        # ---- agg3 over layers
        mu, sd = {}, {}
        for layer in rf.LAYERS:
            y = g[g.layer == layer].d_G672.values
            mu[layer], sd[layer] = float(y.mean()), float(y.std() + 1e-12)
        for name in ("B0", "B1", "B2"):
            z = np.mean([(arms[(l, name)] - mu[l]) / sd[l] for l in rf.LAYERS], axis=0)
            zy = np.mean([(g[g.layer == l].d_G672.values - mu[l]) / sd[l]
                          for l in rf.LAYERS], axis=0)
            agg_rows.append(dict(
                dataset=ds, object=obj, n=n, arm=name, n_samples=len(z),
                spearman=float(spearmanr(z, zy).statistic),
                mae=float(np.mean(np.abs(z - zy))),
                rel_error=float(np.mean(np.abs(z - zy) / (np.abs(zy) + 1e-12)))))
    return pd.DataFrame(per_layer), pd.DataFrame(agg_rows)


def decide(agg3):
    """Pre-registered GO gate.  No threshold is touched after seeing results."""
    piv = agg3.pivot_table(index=["dataset", "object", "n"], columns="arm",
                           values="spearman").reset_index()
    piv["gain_B2_B1"] = piv.B2 - piv.B1
    rows = []
    for n in SCALES:
        s = piv[piv.n == n]
        med = {a: float(s[a].median()) for a in ("B0", "B1", "B2")}
        gain = float(s.gain_B2_B1.median())
        per_ds = {ds: float(s[s.dataset == ds].gain_B2_B1.median())
                  for ds in ("mvtec", "visa")}
        c1 = med["B2"] >= GATE_RHO
        c2 = gain >= GATE_GAIN
        c3 = all(v > 0 for v in per_ds.values())
        c4 = med["B2"] > med["B0"]
        rows.append(dict(n=n, median_B0=med["B0"], median_B1=med["B1"],
                         median_B2=med["B2"], median_gain_B2_B1=gain,
                         gain_mvtec=per_ds["mvtec"], gain_visa=per_ds["visa"],
                         gate1_rho=bool(c1), gate2_gain=bool(c2),
                         gate3_both_datasets=bool(c3), gate4_B2_gt_B0=bool(c4),
                         GO=bool(c1 and c2 and c3 and c4)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "f0_gate_decision.csv"), index=False)
    print("\n  === V3.1-F0 pre-registered gate ===")
    print(f"    G1 median_category rho(B2) >= {GATE_RHO:.2f}")
    print(f"    G2 median_category [rho(B2)-rho(B1)] >= {GATE_GAIN:.2f}")
    print(f"    G3 MVTec and VisA B2-B1 both > 0")
    print(f"    G4 B2 > B0")
    print(d.round(3).to_string(index=False))
    win = d[d.GO]
    if len(win):
        print(f"\n    GO. Smallest passing scale: n = {int(win.n.min())}")
    else:
        print("\n    NO-GO: no scale passes all four gates.")
        print("    => stop V3 architecture exploration: do NOT extend n, do NOT")
        print("       add a more complex model, do NOT consult defect performance.")
    return d


if __name__ == "__main__":
    main()
