# -*- coding: utf-8 -*-
from pathlib import Path
import sys, numpy as np, pandas as pd, torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

print("=== R5 PRECHECK ===")
required=[
    ROOT/"experiments"/"model_v0"/"phase_m1_rescue.py",
    ROOT/"experiments"/"model_v0"/"gate_r2_layer_confirm.py",
    ROOT/"experiments"/"model_v0"/"tail_calib.py",
    ROOT/"r5_rawmap_causal_interventions"/"R5_MAIN_SELECTION.csv",
]
for p in required:
    print("[OK]" if p.exists() else "[MISSING]",p)
    if not p.exists(): raise SystemExit(2)

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:",torch.cuda.get_device_name(0))
print("Torch:",torch.__version__,"CUDA runtime:",torch.version.cuda)

from phase_m1_rescue import load_obj
sel=pd.read_csv(ROOT/"r5_rawmap_causal_interventions"/"R5_MAIN_SELECTION.csv")
bad=[]
for obj in sorted(sel.object.unique()):
    try:
        root,layers,tr,te=load_obj("r0",obj)
        ftr=tr["final"]; fte=te["final"]
        for name,c in [("train",ftr),("test",fte)]:
            for k in ["feats","offsets"]:
                if k not in c: raise RuntimeError(f"{name} missing {k}")
        for k in ["types","grids","gt_frac","names"]:
            if k not in fte: raise RuntimeError(f"test missing {k}")
        if int(fte["offsets"][-1]) != len(fte["feats"]):
            raise RuntimeError("test offsets/features mismatch")
        print("[CACHE OK]",obj,"test_images=",len(fte["types"]))
        del tr,te
    except Exception as e:
        print("[CACHE FAIL]",obj,repr(e))
        bad.append((obj,repr(e)))
if bad:
    raise SystemExit("PRECHECK FAIL")
print("PRECHECK PASS")
