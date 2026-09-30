# -*- coding: utf-8 -*-
from pathlib import Path
import sys, os, numpy as np, pandas as pd, torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

print("=== TSN ROOT-CAUSE LOCAL PRECHECK ===")
print("ROOT:",ROOT)

required=[
    ROOT/"experiments"/"model_v0"/"phase_m1_rescue.py",
    ROOT/"experiments"/"model_v0"/"gate_r2_layer_confirm.py",
    ROOT/"experiments"/"model_v0"/"tail_calib.py",
    ROOT/"results"/"model_v0"/"metrics"/"tsn_formal_v2.csv",
    ROOT/"tsn_rootcause_local_kit_v2"/"11_causal_selection_v2.csv",
]
for p in required:
    ok=p.exists()
    print("[OK]" if ok else "[MISSING]",p)
    if not ok:
        raise SystemExit(2)

from phase_m1_rescue import load_obj
from tail_calib import RESULTS

print("CUDA:",torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:",torch.cuda.get_device_name(0))
print("Torch:",torch.__version__,"CUDA runtime:",torch.version.cuda)

sel=pd.read_csv(ROOT/"tsn_rootcause_local_kit_v2"/"11_causal_selection_v2.csv")
print("Selected causal cells:",len(sel))
print(sel[["dataset","object","shot","split"]].to_string(index=False))

# Validate that every selected object can load raw train/test caches with final layer features.
bad=[]
for obj in sorted(sel.object.unique()):
    try:
        root,layers,tr,te=load_obj("r0",obj)
        if "final" not in tr or "final" not in te:
            bad.append((obj,"missing final layer"))
            continue
        for splitname,c in [("train",tr["final"]),("test",te["final"])]:
            if "feats" not in c or "offsets" not in c:
                bad.append((obj,f"{splitname} missing feats/offsets"))
                continue
            offs=np.asarray(c["offsets"])
            if len(offs)<2 or int(offs[-1])!=len(c["feats"]):
                bad.append((obj,f"{splitname} invalid offsets/features"))
        print("[CACHE OK]",obj)
    except Exception as e:
        bad.append((obj,repr(e)))
        print("[CACHE FAIL]",obj,repr(e))

if bad:
    print("\nFAILED OBJECTS:")
    for x in bad: print(x)
    raise SystemExit("PRECHECK FAIL: raw feature cache is incomplete.")

print("\nPRECHECK PASS: selected objects have train/test final-layer raw features and CUDA is ready.")
