# -*- coding: utf-8 -*-
"""R6-A1F -- Layer-wise Causal Propagation Trace (main driver).

R6-A1E measured one layer (block 11). This traces all 12 blocks in a single
forward per image and asks: where does the pre-encoding counterfactual's
far-field consequence START, does it GROW with depth, and does its magnitude
survive an input-perturbation magnitude-matched control?

Reuses A1E's frozen manifests verbatim (donors, matched goods, survivor set), so
this is a strict extension of A1E rather than a re-randomization: block 11 of the
dense trace must reproduce A1E's cell-level drift.

Everything is diagnostic. No detector, no threshold tuning.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "model_v0"))
sys.path.insert(0, str(ROOT / "experiments" / "baseline"))
sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDINO"))
sys.path.insert(0, str(ROOT / "r6a1e_preencoding"))
sys.path.insert(0, str(ROOT / "r6a1f_kit"))

from phase_m1_rescue import load_obj                                     # noqa: E402
from gate_r2_layer_confirm import draw_images                            # noqa: E402
from r6a1e_utils import (load_anomalydino_model, make_support_bank,      # noqa: E402
                         cache_img_features, score_1nn, topmean,
                         dilate_patch_mask, patchmask_to_pixel, replace_tensor,
                         cheb_distance_from_gt, ring_label)
from r6a1f_utils import (LAYER_INDEX, LAYER_NAME, extract_dense,         # noqa: E402
                         dense_drift_full, dense_rel_l2, topmean_rows,
                         input_delta_pixel, input_delta_token)

CFG = json.loads((ROOT / "r6a1e_preencoding" / "R6A1E_CONFIG.json").read_text(encoding="utf-8"))
KIT = json.loads((ROOT / "r6a1f_kit" / "R6A1F_CONFIG.json").read_text(encoding="utf-8"))
A1E = ROOT / "results" / "model_v0" / "metrics" / "r6a1e_preencoding"

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--objects", default="", help="smoke only: comma list; empty = full config")
_ap.add_argument("--max-bad", type=int, default=0, help="smoke only: cap bad images; 0 = all")
_ap.add_argument("--outdir", default="", help="smoke only: alternate output directory")
_args = _ap.parse_args()

# precheck artifacts always come from the canonical dir; OUT only receives trace
# outputs (so a smoke run can never overwrite the gated manifests)
PRE = ROOT / "results" / "model_v0" / "metrics" / "r6a1f_layerwise"
OUT = ROOT / "results" / "model_v0" / "metrics" / (_args.outdir or "r6a1f_layerwise")
OUT.mkdir(parents=True, exist_ok=True)

OBJECTS = [o for o in _args.objects.split(",") if o] or [str(o) for o in KIT["objects"]]
RADII = [int(r) for r in KIT["radii"]]
ALPHA = float(KIT["alpha"][0])
GT_THR = 0.10
MIN_VALID_R16 = 10
RINGS = ["1-2", "3-4", "5-8", "9-16", "17+"]
NL = len(LAYER_INDEX)

if not (PRE / "parity_report.csv").exists():
    raise SystemExit(f"Missing {PRE/'parity_report.csv'}; run r6a1f_precheck.py first.")
for req in ["donor_manifest.csv", "pair_manifest.csv",
            "survivor_manifest.csv", "resolved_test_paths.csv"]:
    if not (A1E / req).exists():
        raise SystemExit(f"Missing frozen A1E artifact {A1E/req}. STOP.")
p = pd.read_csv(PRE / "parity_report.csv")
if (p.median_cos < 0.9995).any() or (p.p01_cos < 0.995).any():
    raise SystemExit("Saved parity report does not pass the frozen gate. STOP.")

PATHS = pd.read_csv(A1E / "resolved_test_paths.csv")
DONORS = pd.read_csv(A1E / "donor_manifest.csv")
PAIRS = pd.read_csv(A1E / "pair_manifest.csv")
SURV = pd.read_csv(A1E / "survivor_manifest.csv")

model = load_anomalydino_model(ROOT, CFG)
patch_size = int(model.model.patch_size)

series_rows, ring_rows, cell_rows = [], [], []


def get_path(obj, idx):
    q = PATHS[(PATHS.object == obj) & (PATHS.image_index == int(idx))]
    if len(q) != 1:
        raise RuntimeError(f"path manifest miss {obj}/{idx}")
    return str(q.iloc[0].path)


for obj in OBJECTS:
    t0 = time.perf_counter()
    print(f"\n=== OBJECT {obj} ===", flush=True)
    _, _, tr, te = load_obj("r0", obj)
    tc, ec = tr["final"], te["final"]

    ids = [int(v) for v in draw_images(tc["offsets"], 4, 0, obj)]
    bank = make_support_bank(tc, ids)

    types = np.asarray([str(x) for x in ec["types"]])
    offs = np.asarray(ec["offsets"])
    H, W = map(int, ec["grids"][0])
    N = int(offs[1] - offs[0])
    if N != H * W:
        raise RuntimeError(f"{obj}: N={N} != {H}x{W}")

    # raw features straight from the frozen cache -- this is what A1E scored
    raw_feats = {i: cache_img_features(ec, i) for i in range(len(types))}

    survivors = [int(b) for b in
                 SURV[(SURV.object == obj) & (SURV.primary_survivor)].bad_index]
    if _args.max_bad:
        survivors = survivors[:_args.max_bad]
    print(f"  survivors {len(survivors)}", flush=True)

    donor_tensors = []
    for _, dr in DONORS[DONORS.object == obj].sort_values("donor_rank").iterrows():
        t, g = model.prepare_image(str(dr.path))
        if tuple(map(int, g)) != (H, W):
            raise RuntimeError(f"{obj}: donor grid {g} != {(H,W)}")
        donor_tensors.append((int(dr.donor_rank), t, str(dr.path)))

    for bcount, bi in enumerate(survivors):
        aa, bb = int(offs[bi]), int(offs[bi + 1])
        gt = np.asarray(ec["gt_frac"][aa:bb], float).reshape(H, W)
        G = gt > GT_THR
        if not G.any():
            continue
        dist = cheb_distance_from_gt(G).reshape(-1)
        exc_masks = {r: dilate_patch_mask(G, r) for r in RADII}
        # ring membership is a property of the BAD IMAGE, not of (radius, donor):
        # build it once instead of rebuilding it with a per-token Python loop
        # inside every cell.
        ring_of = np.array([ring_label(int(d)) for d in dist])
        ring_masks = {ring: (ring_of == ring) for ring in RINGS}

        gp = PAIRS[(PAIRS.object == obj) & (PAIRS.bad_index == bi)].sort_values("good_rank")
        good_ids = [int(x) for x in gp.good_index.tolist()]

        raw_batch = [model.prepare_image(get_path(obj, bi))[0]] + \
                    [model.prepare_image(get_path(obj, gi))[0] for gi in good_ids]
        if any(tuple(t.shape) != tuple(raw_batch[0].shape) for t in raw_batch):
            raise RuntimeError(f"{obj}/{bi}: target tensor shapes differ")

        # raw dense pass: ONCE per bad image, hoisted out of radius/donor loops
        raw12, x0_raw = extract_dense(model, raw_batch)

        for r in RADII:
            exc = exc_masks[r]
            valid = (~exc).reshape(-1)               # FLAT, always
            if valid.sum() < 1:
                continue
            pixmask = patchmask_to_pixel(exc, patch_size, raw_batch[0].shape[-2:])

            for donor_rank, donor_tensor, donor_path in donor_tensors:
                cfs = [replace_tensor(t, donor_tensor, pixmask) for t in raw_batch]
                cf12, x0_cf = extract_dense(model, cfs)

                drift_all, rel_all, dpix, dtok, topo = [], [], [], [], []
                tok_l2, tok_ref = [], []
                for s in range(len(raw_batch)):
                    dr = dense_drift_full(raw12[s], cf12[s])          # (12, N)
                    drift_all.append(dr)
                    rel_all.append(dense_rel_l2(raw12[s], cf12[s], valid))
                    dpix.append(input_delta_pixel(raw_batch[s], cfs[s], pixmask))
                    dtok.append(input_delta_token(x0_raw[s], x0_cf[s], exc))
                    topo.append(topmean_rows(dr, valid, ALPHA))
                    # per-token quantities, computed ONCE; the ring profile below
                    # is then a cheap masked mean instead of a fresh norm per
                    # (ring, layer) pair.
                    tok_l2.append(np.linalg.norm(
                        np.asarray(cf12[s], np.float32) - np.asarray(raw12[s], np.float32),
                        axis=2))
                    tok_ref.append(np.linalg.norm(np.asarray(raw12[s], np.float32), axis=2))

                # ---- final-layer scores, exactly as A1E -------------------
                sc_raw = [topmean(score_1nn(raw_feats[bi], bank), valid, ALPHA)] + \
                         [topmean(score_1nn(raw_feats[gi], bank), valid, ALPHA) for gi in good_ids]
                sc_cf = [topmean(score_1nn(cf12[s][11], bank), valid, ALPHA)
                         for s in range(len(raw_batch))]

                # ---- A1E cross-check: drift from the CACHE raw vs fresh cf --
                # A1E's raw reference was the fp16 cache; ours is a fresh fp32
                # forward. Parity is 0.99999988, not 1.0, so compare with
                # tolerance and keep the difference visible.
                a1e_drift = []
                for s, idx in enumerate([bi] + good_ids):
                    cache_raw = np.asarray(raw_feats[idx], np.float32)
                    a1e_drift.append(float(np.mean(
                        dense_drift_full(cache_raw[None], cf12[s][11][None])[0][valid])))

                # ---- per-series layer rows --------------------------------
                for s in range(len(raw_batch)):
                    tag = "bad" if s == 0 else f"good{s}"
                    gi = -1 if s == 0 else good_ids[s - 1]
                    for l in LAYER_INDEX:
                        series_rows.append(dict(
                            object=obj, bad_index=bi, series=tag, good_index=gi,
                            radius=r, donor_rank=donor_rank, layer_index=l,
                            layer_name=LAYER_NAME.get(l, ""), n_valid=int(valid.sum()),
                            delta_pixel=dpix[s], delta_token=dtok[s],
                            drift_mean=float(drift_all[s][l][valid].mean()),
                            drift_top01=float(topo[s][l]),
                            rel_l2=float(rel_all[s][l]),
                            score_raw=float(sc_raw[s]), score_cf=float(sc_cf[s])))

                # ---- radius/ring profile ----------------------------------
                for s in range(len(raw_batch)):
                    tag = "bad" if s == 0 else f"good{s}"
                    for ring in ["ALL"] + RINGS:
                        m = valid if ring == "ALL" else (valid & ring_masks[ring])
                        if not m.any():
                            continue
                        n_l2 = tok_l2[s][:, m].mean(axis=1)
                        n_ref = np.clip(tok_ref[s][:, m].mean(axis=1), 1e-12, None)
                        for l in LAYER_INDEX:
                            ring_rows.append(dict(
                                object=obj, bad_index=bi, series=tag, radius=r,
                                donor_rank=donor_rank, ring=ring, layer_index=l,
                                layer_name=LAYER_NAME.get(l, ""), n_patch=int(m.sum()),
                                drift_mean=float(drift_all[s][l][m].mean()),
                                rel_l2=float(n_l2[l] / n_ref[l])))

                # ---- matched-pair rows ------------------------------------
                for j, gi in enumerate(good_ids, start=1):
                    cell_rows.append(dict(
                        object=obj, bad_index=bi, good_index=gi, radius=r,
                        donor_rank=donor_rank, n_valid=int(valid.sum()),
                        raw_score_bad=sc_raw[0], raw_score_good=sc_raw[j],
                        cf_score_bad=sc_cf[0], cf_score_good=sc_cf[j],
                        score_DiD=(sc_cf[0] - sc_raw[0]) - (sc_cf[j] - sc_raw[j]),
                        a1e_drift_bad=a1e_drift[0], a1e_drift_good=a1e_drift[j],
                        drift_diff_a1e=a1e_drift[0] - a1e_drift[j]))
                del cfs, cf12, x0_cf, drift_all, rel_all

        del raw12, x0_raw, raw_batch
        if (bcount + 1) % 10 == 0:
            print(f"  {obj} {bcount+1}/{len(survivors)}", flush=True)
            pd.DataFrame(series_rows).to_csv(OUT / "per_series_layer.csv", index=False)
            pd.DataFrame(ring_rows).to_csv(OUT / "radius_profile_raw.csv", index=False)
            pd.DataFrame(cell_rows).to_csv(OUT / "matched_cells.csv", index=False)

    print(f"  {obj} done ({time.perf_counter()-t0:.1f}s)", flush=True)
    pd.DataFrame(series_rows).to_csv(OUT / "per_series_layer.csv", index=False)
    pd.DataFrame(ring_rows).to_csv(OUT / "radius_profile_raw.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "matched_cells.csv", index=False)

print("\nR6-A1F layer trace complete:", OUT)
