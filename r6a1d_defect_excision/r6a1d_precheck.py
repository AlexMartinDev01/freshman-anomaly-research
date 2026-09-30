# -*- coding: utf-8 -*-
from pathlib import Path
import sys,pandas as pd,torch,numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj

sel=pd.read_csv(ROOT/"r6a1d_defect_excision"/"R6A1D_SELECTION.csv")
print("=== R6-A1D PRECHECK ===")
print("CUDA:",torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
print("GPU:",torch.cuda.get_device_name(0))

for obj in sel.object:
    _,_,tr,te=load_obj("r0",obj)
    c=te["final"]
    grids=[tuple(map(int,g)) for g in c["grids"]]
    if len(set(grids))!=1:
        raise RuntimeError(f"{obj}: multiple grids {set(grids)}")
    if "gt_frac" not in c:
        raise RuntimeError(f"{obj}: missing gt_frac")
    print("[OK]",obj,"grid",grids[0],"images",len(c["types"]))
print("PRECHECK PASS")
