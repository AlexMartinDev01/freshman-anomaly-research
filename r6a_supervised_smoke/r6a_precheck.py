# -*- coding: utf-8 -*-
from pathlib import Path
import sys, numpy as np, pandas as pd, torch, sklearn

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

print("=== R6-A SMOKE PRECHECK ===")
required=[
    ROOT/"experiments"/"model_v0"/"phase_m1_rescue.py",
    ROOT/"experiments"/"model_v0"/"gate_r2_layer_confirm.py",
    ROOT/"r6a_supervised_smoke"/"R6A_SMOKE_SELECTION.csv",
]
for p in required:
    print("[OK]" if p.exists() else "[MISSING]",p)
    if not p.exists(): raise SystemExit(2)

print("CUDA:",torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
print("GPU:",torch.cuda.get_device_name(0))
print("torch:",torch.__version__,"sklearn:",sklearn.__version__)

from phase_m1_rescue import load_obj
sel=pd.read_csv(ROOT/"r6a_supervised_smoke"/"R6A_SMOKE_SELECTION.csv")
for obj in sel.object:
    _,layers,tr,te=load_obj("r0",obj)
    c=te["final"]
    for k in ["feats","offsets","gt_frac","types","names"]:
        if k not in c: raise RuntimeError(f"{obj}: test final missing {k}")
    if "final" not in tr: raise RuntimeError(f"{obj}: train final missing")
    print("[CACHE OK]",obj,
          "images=",len(c["types"]),
          "patches=",len(c["feats"]),
          "dim=",np.asarray(c["feats"]).shape[-1])
print("PRECHECK PASS")
