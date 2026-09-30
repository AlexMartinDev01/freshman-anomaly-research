# -*- coding: utf-8 -*-
"""R6-A1F precheck -- mandatory gates. STOP on any failure; never loosen.

Beyond the A1E parity gate this adds one guard that matters specifically here:
`phase_m1_rescue.load_cached` HARDCODES `grids = (32,32)` for the stacked MVTec
test cache (line 86). It is correct for screw/cable/bottle only because those
images really are 32x32. So the token-count check below is asserted from the
FEATURE LENGTH, independently of `ec["grids"]`, and cross-checked against
`model.prepare_image`.
"""
from __future__ import annotations

import hashlib
import json
import sys
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

from phase_m1_rescue import load_obj                                  # noqa: E402
from r6a1e_utils import (load_anomalydino_model, cosine_positionwise,  # noqa: E402
                         cache_img_features)
from r6a1f_utils import LAYER_INDEX, LAYER_NAME, A1E_CACHE_BLOCK, EXPECT_N  # noqa: E402

CFG = json.loads((ROOT / "r6a1e_preencoding" / "R6A1E_CONFIG.json").read_text(encoding="utf-8"))
KIT = json.loads((ROOT / "r6a1f_kit" / "R6A1F_CONFIG.json").read_text(encoding="utf-8"))
A1E = ROOT / "results" / "model_v0" / "metrics" / "r6a1e_preencoding"
OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1f_layerwise"
OUT.mkdir(parents=True, exist_ok=True)

print("=== R6-A1F PRECHECK ===")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: STOP")
print("GPU:", torch.cuda.get_device_name(0))

# ---- 1. A1E artifacts present, and their hashes recorded -------------------
REQUIRED = ["parity_report.csv", "resolved_test_paths.csv", "donor_manifest.csv",
            "pair_manifest.csv", "survivor_manifest.csv", "per_pair.csv",
            "raw_full_good_baseline.csv", "FROZEN_OBJECT_VERDICTS.csv"]
missing = [f for f in REQUIRED if not (A1E / f).exists()]
if missing:
    raise SystemExit(f"Missing A1E artifacts: {missing}. STOP.")
with open(OUT / "a1f_inputs_sha256.txt", "w", encoding="utf-8") as fh:
    fh.write("# R6-A1F reuses these frozen R6-A1E artifacts verbatim.\n")
    for f in REQUIRED:
        h = hashlib.sha256((A1E / f).read_bytes()).hexdigest()
        fh.write(f"{h}  {f}\n")
        print("  sha256", f, h[:16])

par = pd.read_csv(A1E / "parity_report.csv")
if (par.median_cos < 0.9995).any() or (par.p01_cos < 0.995).any():
    raise SystemExit("A1E parity report does not pass the frozen gate. STOP.")

PATHS = pd.read_csv(A1E / "resolved_test_paths.csv")
DONORS = pd.read_csv(A1E / "donor_manifest.csv")
PAIRS = pd.read_csv(A1E / "pair_manifest.csv")
SURV = pd.read_csv(A1E / "survivor_manifest.csv")

OBJECTS = [str(o) for o in KIT["objects"]]
for o in OBJECTS:
    if o not in set(SURV.object):
        raise SystemExit(f"{o} missing from survivor_manifest. STOP.")
    n_surv = int(SURV[(SURV.object == o) & (SURV.primary_survivor)].shape[0])
    if n_surv < 1:
        raise SystemExit(f"{o}: zero survivors. STOP.")
    print(f"  {o:11s} survivors={n_surv:4d}  donors={int((DONORS.object==o).sum())}")
print("  objects:", OBJECTS, "| pcb2 excluded by R6A1F_CONFIG (spec)")

# ---- 2. model + block count ------------------------------------------------
model = load_anomalydino_model(ROOT, CFG)
nb = len(model.model.blocks)
if nb != 12:
    raise SystemExit(f"expected 12 blocks, got {nb}. STOP.")
patch_size = int(model.model.patch_size)
print(f"  model blocks={nb} patch_size={patch_size} "
      f"register_tokens={getattr(model.model,'num_register_tokens',0)}")

# ---- 3. per-object: token-count guard + dense identity probe + parity -------
parity_rows, layer_rows = [], []
for obj in OBJECTS:
    _, _, tr, te = load_obj("r0", obj)
    tc, ec = tr["final"], te["final"]
    types = np.asarray([str(x) for x in ec["types"]])
    offs = np.asarray(ec["offsets"])

    # token count from the FEATURE LENGTH, not from ec["grids"]
    i0 = 0
    N_feat = int(offs[i0 + 1] - offs[i0])
    H, W = map(int, ec["grids"][i0])
    if N_feat != H * W:
        raise SystemExit(f"{obj}: feature length {N_feat} != grid {H}x{W}={H*W}. STOP.")
    if obj in EXPECT_N and N_feat != EXPECT_N[obj]:
        raise SystemExit(f"{obj}: N={N_feat} != expected {EXPECT_N[obj]}. STOP.")

    good = np.where(types == "good")[0]
    bad = np.where(types == "bad")[0]
    if not len(good) or not len(bad):
        raise SystemExit(f"{obj}: missing good/bad. STOP.")

    for i in [int(good[0]), int(bad[0])]:
        row = PATHS[(PATHS.object == obj) & (PATHS.image_index == i)]
        if len(row) != 1:
            raise SystemExit(f"{obj}/{i}: path manifest miss. STOP.")
        p = str(row.iloc[0].path)
        t, g = model.prepare_image(p)
        if tuple(map(int, g)) != (H, W):
            raise SystemExit(f"{obj}/{i}: prepare_image grid {g} != cached {(H,W)}. STOP.")
        if tuple(t.shape[-2:]) != (H * patch_size, W * patch_size):
            raise SystemExit(f"{obj}/{i}: prepared tensor {tuple(t.shape)} vs grid. STOP.")

        x = t.unsqueeze(0).to(model.device)
        with torch.inference_mode():
            dense = model.model.get_intermediate_layers(x, n=LAYER_INDEX)
            default = model.model.get_intermediate_layers(x)[0]

        # dense[11] must be the audited A1E quantity, bitwise
        d11 = dense[11]
        same = bool(torch.equal(d11, default))
        maxdiff = float((d11 - default).abs().max().item())

        # parity of blocks 5/8/11 against the frozen cache
        for name, blk in A1E_CACHE_BLOCK.items():
            cached = cache_img_features(te[name], i)
            got = dense[blk].squeeze(0).float().cpu().numpy()
            if got.shape != cached.shape:
                raise SystemExit(f"{obj}/{i}/{name}: shape {got.shape} vs {cached.shape}. STOP.")
            cs = cosine_positionwise(cached, got)
            med, p01 = float(np.median(cs)), float(np.quantile(cs, .01))
            parity_rows.append(dict(object=obj, image_index=i, image_type=types[i],
                                    layer_name=name, layer_index=blk,
                                    median_cos=med, p01_cos=p01, min_cos=float(cs.min()),
                                    n_patches=len(cs), feature_dim=int(cached.shape[1])))
            if med < 0.9995 or p01 < 0.995:
                pd.DataFrame(parity_rows).to_csv(OUT / "parity_report_FAILED.csv", index=False)
                raise SystemExit(f"PARITY FAIL {obj}/{i}/{name}: med={med:.7f} p01={p01:.7f}. STOP.")
        print(f"  {obj:11s} idx{i:<4d} {types[i]:4s} dense[11]==default:{same} "
              f"maxdiff={maxdiff:.2e} N={N_feat}")

        if not same and maxdiff > 0:
            raise SystemExit(f"{obj}/{i}: dense[11] != default call. STOP.")

    for blk in LAYER_INDEX:
        layer_rows.append(dict(object=obj, layer_index=blk,
                               layer_name=LAYER_NAME.get(blk, ""),
                               depth_frac=round((blk + 1) / nb, 4),
                               layer_name_status="fixed_by_25pct_depth_grid",
                               a1e_cache_name=next((k for k, v in A1E_CACHE_BLOCK.items() if v == blk), ""),
                               is_config_named=blk in LAYER_NAME,
                               dense_matches_default=(blk == 11),
                               n_tokens=N_feat, grid_h=H, grid_w=W))

# ---- 4. donor paths + replace_tensor shape contract ------------------------
for _, d in DONORS[DONORS.object.isin(OBJECTS)].iterrows():
    if not Path(str(d.path)).exists():
        raise SystemExit(f"donor path missing: {d.path}. STOP.")
    if bool(d.is_support):
        raise SystemExit(f"donor is a support image: {d.path}. STOP.")
print("  donor paths exist, none is a support image")

# ---- 5. layer index freeze ------------------------------------------------
df = pd.DataFrame(layer_rows)
df.to_csv(OUT / "r6a1f_LAYER_INDEX.csv", index=False)
pd.DataFrame(parity_rows).to_csv(OUT / "parity_report.csv", index=False)

print("\nPARITY GATE PASS")
print("parity rows:", len(parity_rows),
      "| min median_cos:", round(float(pd.DataFrame(parity_rows).median_cos.min()), 8),
      "| min p01:", round(float(pd.DataFrame(parity_rows).p01_cos.min()), 8))
print("wrote", OUT / "r6a1f_LAYER_INDEX.csv", "and", OUT / "parity_report.csv")
