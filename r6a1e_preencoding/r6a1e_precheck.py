# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))
sys.path.insert(0,str(ROOT/"r6a1e_preencoding"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images
from r6a1e_utils import *

CFG=json.loads((ROOT/"r6a1e_preencoding"/"R6A1E_CONFIG.json").read_text(encoding="utf-8"))
SEL=pd.read_csv(ROOT/"r6a1e_preencoding"/"R6A1E_SELECTION.csv")
OUT=ROOT/"results"/"model_v0"/"metrics"/"r6a1e_preencoding"
OUT.mkdir(parents=True,exist_ok=True)

print("=== R6-A1E PRECHECK ===")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: STOP")
print("GPU:",torch.cuda.get_device_name(0))

roots={}
roots["mvtec"]=resolve_existing_root(ROOT,CFG.get("mvtec_root",""),CFG["mvtec_root_candidates"])
roots["visa"]=resolve_existing_root(ROOT,CFG.get("visa_root",""),CFG["visa_root_candidates"])
print("MVTec root:",roots["mvtec"])
print("VisA root:",roots["visa"])

model=load_anomalydino_model(ROOT,CFG)
patch_size=int(model.model.patch_size)
print("Model:",CFG["model_name"],"patch_size:",patch_size)

resolved=[]
donors=[]
pairs=[]
parity=[]

for _,rr in SEL.iterrows():
    ds=str(rr.dataset); obj=str(rr.object); shot=int(rr.shot); split=int(rr["split"])
    print("\nOBJECT",obj,flush=True)
    _,_,tr,te=load_obj("r0",obj)
    tc,ec=tr["final"],te["final"]
    ids=[int(v) for v in draw_images(tc["offsets"],shot,split,obj)]
    support_names=set()
    if "names" in tc:
        support_names={str(tc["names"][i]).lower() for i in ids}

    objroot=roots[ds]/obj
    trainroot=objroot/"train"
    testroot=objroot/"test"
    if not trainroot.exists() or not testroot.exists():
        raise RuntimeError(f"{obj}: expected train/test under {objroot}")

    train_idx=build_image_index(trainroot)
    test_idx=build_image_index(testroot)

    types=np.asarray([str(x) for x in ec["types"]])
    names=np.asarray([str(x) for x in ec["names"]])
    grids=[tuple(map(int,g)) for g in ec["grids"]]
    if len(set(grids))!=1:
        raise RuntimeError(f"{obj}: multiple cached grids {set(grids)}; STOP")
    cache_grid=grids[0]

    # Resolve every test image path.
    paths={}
    for i,nm in enumerate(names):
        p=resolve_name(nm,test_idx)
        paths[i]=p
        resolved.append(dict(dataset=ds,object=obj,image_index=i,
                             image_type=types[i],cached_name=nm,path=str(p)))

    # Resolve all train names when available, choose non-support donors.
    cand=[]
    if "names" in tc:
        for i,nm in enumerate(tc["names"]):
            if i in ids: continue
            try:
                p=resolve_name(nm,train_idx)
                cand.append((i,str(nm),p))
            except Exception:
                pass
    if len(cand)<int(CFG["n_normal_donors"]):
        raise RuntimeError(
            f"{obj}: only {len(cand)} resolvable non-support train-good donors. "
            "Protocol forbids support/test donor fallback. STOP."
        )

    rng=np.random.default_rng(stable_seed(obj))
    choose=rng.choice(len(cand),size=int(CFG["n_normal_donors"]),replace=False)
    chosen=[cand[int(j)] for j in np.atleast_1d(choose)]

    # Check donor prepared grid/shape.
    donor_tensors=[]
    for rank,(i,nm,p) in enumerate(chosen):
        t,g=model.prepare_image(str(p))
        if tuple(map(int,g))!=cache_grid:
            raise RuntimeError(f"{obj}: donor grid {g} != cache grid {cache_grid}: {p}")
        donor_tensors.append(t)
        donors.append(dict(dataset=ds,object=obj,donor_rank=rank,
                           train_index=i,cached_name=nm,path=str(p),
                           is_support=False,grid=str(tuple(g))))

    # Test image prepared-grid check.
    for i,p in paths.items():
        t,g=model.prepare_image(str(p))
        if tuple(map(int,g))!=cache_grid:
            raise RuntimeError(f"{obj}: test idx {i} grid {g} != cache {cache_grid}: {p}")
        if tuple(t.shape[-2:])!=(cache_grid[0]*patch_size,cache_grid[1]*patch_size):
            raise RuntimeError(f"{obj}: prepared tensor / patch-grid mismatch at {p}")

    # Mandatory parity: first good + first bad.
    diag=[]
    good=np.where(types=="good")[0]
    bad=np.where(types=="bad")[0]
    if not len(good) or not len(bad):
        raise RuntimeError(f"{obj}: missing good or bad test images")
    diag=[int(good[0]),int(bad[0])]
    for i in diag:
        t,g=model.prepare_image(str(paths[i]))
        z=model.extract_features(t)
        c=cache_img_features(ec,i)
        if z.shape!=c.shape:
            raise RuntimeError(f"{obj}/{i}: parity shape mismatch model={z.shape}, cache={c.shape}")
        cs=cosine_positionwise(c,z)
        row=dict(dataset=ds,object=obj,image_index=i,image_type=types[i],
                 cached_name=names[i],median_cos=float(np.median(cs)),
                 p01_cos=float(np.quantile(cs,.01)),min_cos=float(np.min(cs)),
                 n_patches=len(cs),feature_dim=c.shape[1])
        parity.append(row)
        print(" parity",types[i],"median",row["median_cos"],"p01",row["p01_cos"])
        if row["median_cos"]<0.9995 or row["p01_cos"]<0.995:
            pd.DataFrame(parity).to_csv(OUT/"parity_report_FAILED.csv",index=False)
            raise RuntimeError(
                f"PARITY GATE FAIL {obj}/{i}: median={row['median_cos']:.6f}, "
                f"p01={row['p01_cos']:.6f}. STOP; do not run intervention."
            )

    # Pair 3 good images per bad, frozen independently of scores.
    ng=int(CFG["n_good_pairs_per_bad"])
    rng=np.random.default_rng(stable_seed(obj)+77)
    for bi in bad:
        picks=rng.choice(good,size=min(ng,len(good)),replace=False)
        for rank,gi in enumerate(np.atleast_1d(picks)):
            pairs.append(dict(dataset=ds,object=obj,bad_index=int(bi),
                              good_rank=rank,good_index=int(gi)))

pd.DataFrame(resolved).to_csv(OUT/"resolved_test_paths.csv",index=False)
pd.DataFrame(donors).to_csv(OUT/"donor_manifest.csv",index=False)
pd.DataFrame(pairs).to_csv(OUT/"pair_manifest.csv",index=False)
pd.DataFrame(parity).to_csv(OUT/"parity_report.csv",index=False)
(OUT/"resolved_roots.json").write_text(
    json.dumps({k:str(v) for k,v in roots.items()},indent=2),encoding="utf-8")

print("\nPARITY GATE PASS")
print("Resolved manifests written to",OUT)
