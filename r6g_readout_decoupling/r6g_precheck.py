# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))
sys.path.insert(0,str(ROOT/"r6g_readout_decoupling"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images
from r6g_utils import *

CFG=json.loads((ROOT/"r6g_readout_decoupling"/"R6G_CONFIG.json").read_text(encoding="utf-8"))
SEL=pd.read_csv(ROOT/"r6g_readout_decoupling"/"R6G_SELECTION.csv")
OUT=ROOT/"results"/"model_v0"/"metrics"/"r6g_readout_decoupling"
OUT.mkdir(parents=True,exist_ok=True)

A1E=ROOT/CFG["a1e_dir"]
A1F=ROOT/CFG["a1f_dir"]
SMOKE=ROOT/CFG["r6a_smoke_dir"]

required=[
    A1E/"resolved_test_paths.csv",
    A1F/"dense_layer_curve.csv",
    A1F/"object_verdict.csv",
    SMOKE/"folds.csv",
    SMOKE/"summary.csv",
]
for p in required:
    if not p.exists(): raise SystemExit(f"Missing required frozen artifact: {p}. STOP.")

with open(OUT/"R6G_INPUT_SHA256.txt","w",encoding="utf-8") as fh:
    for p in required:
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        fh.write(f"{h}  {p}\n")
        print("sha",p.name,h[:16])

# Need the exact A1E config/model settings.
a1e_cfg_path=ROOT/CFG["a1e_code_dir"]/"R6A1E_CONFIG.json"
if not a1e_cfg_path.exists():
    raise SystemExit(f"Missing {a1e_cfg_path}. STOP.")
A1ECFG=json.loads(a1e_cfg_path.read_text(encoding="utf-8"))

PATHS=pd.read_csv(A1E/"resolved_test_paths.csv")
FOLDS=pd.read_csv(SMOKE/"folds.csv")
SUM=pd.read_csv(SMOKE/"summary.csv")

model=load_anomalydino_model(ROOT,A1ECFG)
if len(model.model.blocks)!=12:
    raise SystemExit(f"Expected 12 blocks, got {len(model.model.blocks)}. STOP.")

# dataset roots for support-path resolution
roots={}
roots["mvtec"]=resolve_existing_root(ROOT,A1ECFG.get("mvtec_root",""),A1ECFG["mvtec_root_candidates"])
roots["visa"]=resolve_existing_root(ROOT,A1ECFG.get("visa_root",""),A1ECFG["visa_root_candidates"])

# object -> dataset, frozen by known benchmark membership
dataset_of={"bottle":"mvtec","cable":"mvtec","screw":"mvtec",
            "chewinggum":"visa","macaroni2":"visa","pcb2":"visa"}

support_rows=[]
parity_rows=[]
for _,rr in SEL.iterrows():
    obj=str(rr.object); shot=int(rr.shot); split=int(rr["split"])
    _,_,tr,te=load_obj("r0",obj)
    tc,ec=tr["final"],te["final"]
    ids=[int(v) for v in draw_images(tc["offsets"],shot,split,obj)]

    # Support IDs must exactly equal the earlier smoke summary.
    ss=SUM[(SUM.object==obj)&(SUM.probe=="logreg")]
    if len(ss)!=1: raise SystemExit(f"{obj}: missing/duplicate smoke summary. STOP.")
    frozen=[int(x) for x in str(ss.iloc[0].support_ids).split(";") if str(x)!=""]
    if ids!=frozen:
        raise SystemExit(f"{obj}: support mismatch current={ids}, frozen={frozen}. STOP.")

    # Test image names/folds exact alignment.
    fo=FOLDS[FOLDS.object==obj].sort_values("image_id")
    names=[str(x) for x in ec["names"]]
    if len(fo)!=len(names):
        raise SystemExit(f"{obj}: fold count {len(fo)} != images {len(names)}. STOP.")
    if [str(x) for x in fo.image.tolist()] != names:
        # fallback exact basename comparison is NOT allowed: names must be identical
        raise SystemExit(f"{obj}: frozen fold image names do not exactly align. STOP.")

    # Resolved test paths exact coverage.
    po=PATHS[PATHS.object==obj].sort_values("image_index")
    if len(po)!=len(names):
        raise SystemExit(f"{obj}: A1E test paths {len(po)} != images {len(names)}. STOP.")
    if po.image_index.tolist()!=list(range(len(names))):
        raise SystemExit(f"{obj}: test path indexes not contiguous. STOP.")

    # Resolve support train paths.
    ds=dataset_of[obj]
    trainroot=roots[ds]/obj/"train"
    idx=build_image_index(trainroot)
    if "names" not in tc:
        raise SystemExit(f"{obj}: train cache has no names; cannot resolve support. STOP.")
    # Every support image is part of the 1NN bank, so parity is mandatory there too.
    toff=np.asarray(tc["offsets"])
    for sid in ids:
        sp=resolve_name(str(tc["names"][sid]),idx)
        t,g=model.prepare_image(str(sp))
        dense=extract_dense(model,[t])[0]
        got=dense[11]
        a,b=int(toff[sid]),int(toff[sid+1])
        cached=np.asarray(tc["feats"][a:b],np.float32)
        if got.shape!=cached.shape:
            raise SystemExit(f"{obj}/support{sid}: dense final shape {got.shape} != train cache {cached.shape}. STOP.")
        cs=cosine_positionwise(cached,got)
        med=float(np.median(cs)); p01=float(np.quantile(cs,.01))
        if med<0.9995 or p01<0.995:
            raise SystemExit(f"SUPPORT PARITY FAIL {obj}/{sid}: med={med} p01={p01}. STOP.")
        support_rows.append(dict(object=obj,support_id=sid,cached_name=str(tc["names"][sid]),
                                 path=str(sp),median_cos=med,p01_cos=p01,min_cos=float(cs.min()),
                                 n_patches=len(cs),dim=cached.shape[1]))

    # Dense final parity on first good + first bad test image.
    types=np.asarray([str(x) for x in ec["types"]])
    check=[int(np.where(types=="good")[0][0]),int(np.where(types=="bad")[0][0])]
    off=np.asarray(ec["offsets"])
    for i in check:
        p=str(po[po.image_index==i].iloc[0].path)
        t,g=model.prepare_image(p)
        dense=extract_dense(model,[t])[0]
        got=dense[11]
        a,b=int(off[i]),int(off[i+1])
        cached=np.asarray(ec["feats"][a:b],np.float32)
        if got.shape!=cached.shape:
            raise SystemExit(f"{obj}/{i}: dense final shape {got.shape} != cache {cached.shape}. STOP.")
        cs=cosine_positionwise(cached,got)
        med=float(np.median(cs)); p01=float(np.quantile(cs,.01))
        parity_rows.append(dict(object=obj,image_index=i,image_type=types[i],
                                median_cos=med,p01_cos=p01,min_cos=float(cs.min()),
                                n_patches=len(cs),dim=cached.shape[1]))
        if med<0.9995 or p01<0.995:
            pd.DataFrame(parity_rows).to_csv(OUT/"parity_FAILED.csv",index=False)
            raise SystemExit(f"PARITY FAIL {obj}/{i} med={med} p01={p01}. STOP.")

pd.DataFrame(support_rows).to_csv(OUT/"support_path_manifest.csv",index=False)
pd.DataFrame(parity_rows).to_csv(OUT/"parity_report.csv",index=False)
(OUT/"dataset_roots.json").write_text(json.dumps({k:str(v) for k,v in roots.items()},indent=2),encoding="utf-8")
print("R6-G PRECHECK PASS")
