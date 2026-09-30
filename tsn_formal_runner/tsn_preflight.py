# -*- coding: utf-8 -*-
from pathlib import Path
import os,sys
import numpy as np
import torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

print("=== TSN PREFLIGHT ===")
print("ROOT:",ROOT)
need=[
 ROOT/"experiments"/"model_v0"/"phase_m1_rescue.py",
 ROOT/"experiments"/"model_v0"/"gate_r2_layer_confirm.py",
 ROOT/"experiments"/"model_v0"/"tail_calib.py",
]
for p in need:
    print("[OK]" if p.exists() else "[MISSING]",p)
    if not p.exists(): raise SystemExit(2)

from phase_m1_rescue import load_obj,V1,VISA
from tail_calib import RESULTS
print("RESULTS:",RESULTS)
print("MVTec root:",V1, os.path.isdir(V1))
print("VisA root :",VISA, os.path.isdir(VISA))
print("CUDA available:",torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
print("GPU:",torch.cuda.get_device_name(0))
print("Torch:",torch.__version__,"CUDA runtime:",torch.version.cuda)

mv=Path(RESULTS)/"cache_ml_img"/"final"
vi=Path(RESULTS)/"cache_visa"/"final"
print("MVTec final cache dir:",mv, mv.exists())
print("VisA final cache dir :",vi, vi.exists())
mvobjs=sorted(p.name[:-len("_train.npz")] for p in mv.glob("*_train.npz"))
viobjs=sorted(p.name[:-len("_train.npz")] for p in vi.glob("*_train.npz"))
print("MVTec cached objects:",len(mvobjs),mvobjs)
print("VisA cached objects :",len(viobjs),viobjs)
objs=sorted(set(mvobjs+viobjs))
print("TOTAL cached objects:",len(objs))
if len(mvobjs)!=15 or len(viobjs)!=12 or len(objs)!=27:
    raise SystemExit("Expected 15 MVTec + 12 VisA = 27 cached objects.")

for obj in objs:
    root,layers,tr,te=load_obj("r0",obj)
    assert list(layers)==["mid","midlate","final"],(obj,layers)
    for l in layers:
        for splitname,c in [("train",tr[l]),("test",te[l])]:
            for key in ["feats","offsets"]:
                if key not in c: raise RuntimeError(f"{obj}/{l}/{splitname}: missing {key}")
            offs=np.asarray(c["offsets"])
            if len(offs)<2 or offs[0]!=0 or np.any(np.diff(offs)<0):
                raise RuntimeError(f"{obj}/{l}/{splitname}: invalid offsets")
            if int(offs[-1])!=len(c["feats"]):
                raise RuntimeError(f"{obj}/{l}/{splitname}: offsets[-1] != feats len")
    print("[CACHE OK]",obj)

print("PREFLIGHT PASS: 27 objects, three layers, train/test caches and CUDA are available.")
