# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys,json,shutil,gc
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))
sys.path.insert(0,str(ROOT/"r6g_readout_decoupling"))

from phase_m1_rescue import load_obj
from r6g_utils import *

CFG=json.loads((ROOT/"r6g_readout_decoupling"/"R6G_CONFIG.json").read_text(encoding="utf-8"))
SEL=pd.read_csv(ROOT/"r6g_readout_decoupling"/"R6G_SELECTION.csv")
OUT=ROOT/"results"/"model_v0"/"metrics"/"r6g_readout_decoupling"
OUT.mkdir(parents=True,exist_ok=True)

A1E=ROOT/CFG["a1e_dir"]
SMOKE=ROOT/CFG["r6a_smoke_dir"]
PATHS=pd.read_csv(A1E/"resolved_test_paths.csv")
FOLDS=pd.read_csv(SMOKE/"folds.csv")
SUPPORT=pd.read_csv(OUT/"support_path_manifest.csv")

a1e_cfg_path=ROOT/CFG["a1e_code_dir"]/"R6A1E_CONFIG.json"
A1ECFG=json.loads(a1e_cfg_path.read_text(encoding="utf-8"))
model=load_anomalydino_model(ROOT,A1ECFG)

LAYERS=[int(x) for x in CFG["layers"]]
GT=float(CFG["gt_threshold"])
MAXN=int(CFG["max_per_class_per_image"])
SEED=int(CFG["seed"])
TOP=float(CFG["image_top_fraction"])
BATCH=int(CFG["batch_size_extract"])

fold_metric_rows=[]
image_rows=[]
object_layer_rows=[]

def deterministic_sample_indices(y,image_id,obj):
    rng=np.random.default_rng(stable_seed(f"{obj}:{image_id}",SEED))
    keep=[]
    for cls in [0,1]:
        ids=np.where(y==cls)[0]
        if len(ids)>MAXN: ids=rng.choice(ids,MAXN,replace=False)
        keep.extend(ids.tolist())
    return np.array(sorted(keep),dtype=int)

for _,rr in SEL.iterrows():
    obj=str(rr.object)
    print("\n=== R6-G",obj,"===",flush=True)
    _,_,tr,te=load_obj("r0",obj)
    ec=te["final"]; offs=np.asarray(ec["offsets"])
    names=np.asarray([str(x) for x in ec["names"]])
    types=np.asarray([str(x) for x in ec["types"]])
    nimg=len(names)

    fo=FOLDS[FOLDS.object==obj].sort_values("image_id")
    fold_of=fo.fold.to_numpy(dtype=int)
    nfold=int(fold_of.max()+1)

    po=PATHS[PATHS.object==obj].sort_values("image_index")
    test_paths=[str(x) for x in po.path.tolist()]

    # Patch labels + deterministic sampled training patch indexes.
    labels=[]; sample_idx=[]; sample_count=0
    for i in range(nimg):
        a,b=int(offs[i]),int(offs[i+1])
        gt=np.asarray(ec["gt_frac"][a:b],float)
        y=np.full(len(gt),-1,np.int8)
        y[gt==0]=0
        y[gt>GT]=1
        labels.append(y)
        si=deterministic_sample_indices(y,i,obj)
        sample_idx.append(si); sample_count+=len(si)

    # Resolve/extract support banks for all 12 layers.
    so=SUPPORT[SUPPORT.object==obj].sort_values("support_id")
    stensors=[]
    for p in so.path:
        t,g=model.prepare_image(str(p)); stensors.append(t)
    sdense=extract_dense(model,stensors,LAYERS)
    banks={}
    for li,l in enumerate(LAYERS):
        banks[l]=np.concatenate([z[li] for z in sdense],axis=0).astype(np.float32)
    del sdense,stensors

    # ---------- PASS 1: sampled features for fitting all layer/fold probes ----------
    tmp=OUT/f"_tmp_{obj}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()

    # one memmap per layer: only sampled patches; float16 to bound disk.
    mm={}
    for l in LAYERS:
        mm[l]=np.memmap(tmp/f"L{l}.f16",dtype=np.float16,mode="w+",shape=(sample_count,384))
    meta_img=np.empty(sample_count,np.int32)
    meta_y=np.empty(sample_count,np.int8)

    pos=0
    for s in range(0,nimg,BATCH):
        ids=list(range(s,min(s+BATCH,nimg)))
        tensors=[]
        for i in ids:
            t,g=model.prepare_image(test_paths[i]); tensors.append(t)
        dens=extract_dense(model,tensors,LAYERS)
        for j,i in enumerate(ids):
            si=sample_idx[i]; n=len(si)
            for li,l in enumerate(LAYERS):
                mm[l][pos:pos+n]=dens[j][li][si].astype(np.float16)
            meta_img[pos:pos+n]=i
            meta_y[pos:pos+n]=labels[i][si]
            pos+=n
        if (s//BATCH)%10==0: print(" pass1",min(s+BATCH,nimg),"/",nimg,flush=True)
        del dens,tensors
    for l in LAYERS: mm[l].flush()
    np.save(tmp/"sample_image.npy",meta_img)
    np.save(tmp/"sample_y.npy",meta_y)

    # Fit exact frozen models.
    models={}
    for l in LAYERS:
        Xall=np.asarray(mm[l],dtype=np.float32)
        for f in range(nfold):
            trmask=np.array([fold_of[i]!=f for i in meta_img],dtype=bool)
            Xtr=Xall[trmask]; ytr=meta_y[trmask]
            if len(np.unique(ytr))<2:
                raise RuntimeError(f"{obj} L{l} fold{f}: train missing class")
            scaler=StandardScaler().fit(Xtr)
            Xz=scaler.transform(Xtr)
            lr=LogisticRegression(C=1.0,class_weight="balanced",max_iter=3000,
                                  solver="liblinear",random_state=SEED)
            svm=LinearSVC(C=1.0,class_weight="balanced",max_iter=10000,random_state=SEED)
            lr.fit(Xz,ytr); svm.fit(Xz,ytr)
            models[(l,f,"logreg")]=(scaler,lr)
            models[(l,f,"linsvm")]=(scaler,svm)
        del Xall
    del mm
    gc.collect()

    # accumulators: per layer/probe/fold
    acc={}
    for l in LAYERS:
        for f in range(nfold):
            acc[(l,f,"1nn")]={"y":[],"s":[],"img_y":[],"img_s":[]}
            for p in CFG["probes"]:
                acc[(l,f,p)]={"y":[],"s":[],"img_y":[],"img_s":[]}

    # ---------- PASS 2: OOF evaluation, all eligible patches ----------
    for s in range(0,nimg,BATCH):
        ids=list(range(s,min(s+BATCH,nimg)))
        tensors=[]
        for i in ids:
            t,g=model.prepare_image(test_paths[i]); tensors.append(t)
        dens=extract_dense(model,tensors,LAYERS)

        for j,i in enumerate(ids):
            f=int(fold_of[i]); yfull=labels[i]; keep=yfull>=0; y=yfull[keep]
            img_y=int(types[i]=="bad")
            for li,l in enumerate(LAYERS):
                X=dens[j][li]
                Xi=X[keep]
                bscore=score_1nn(X,banks[l])[keep]
                prec_b,rec_b,k_b=tail_precision_recall(y,bscore,TOP)
                im_b=topmean(bscore,TOP)
                a=acc[(l,f,"1nn")]
                a["y"].append(y); a["s"].append(bscore); a["img_y"].append(img_y); a["img_s"].append(im_b)
                image_rows.append(dict(object=obj,image_id=i,image=names[i],fold=f,layer=l,
                    method="1nn",probe="",image_y=img_y,n_eligible=len(y),
                    n_defect=int((y==1).sum()),score=im_b,
                    tail_precision=prec_b,tail_recall=rec_b,tail_k=k_b))

                for p in CFG["probes"]:
                    scaler,mdl=models[(l,f,p)]
                    Xz=scaler.transform(Xi)
                    ps=mdl.predict_proba(Xz)[:,1] if p=="logreg" else mdl.decision_function(Xz)
                    prec,rec,k=tail_precision_recall(y,ps,TOP)
                    ims=topmean(ps,TOP)
                    a=acc[(l,f,p)]
                    a["y"].append(y); a["s"].append(ps); a["img_y"].append(img_y); a["img_s"].append(ims)
                    image_rows.append(dict(object=obj,image_id=i,image=names[i],fold=f,layer=l,
                        method="probe",probe=p,image_y=img_y,n_eligible=len(y),
                        n_defect=int((y==1).sum()),score=ims,
                        tail_precision=prec,tail_recall=rec,tail_k=k))
        if (s//BATCH)%10==0: print(" pass2",min(s+BATCH,nimg),"/",nimg,flush=True)
        del dens,tensors

    # fold metrics
    for l in LAYERS:
        for f in range(nfold):
            for method in ["1nn"]+list(CFG["probes"]):
                a=acc[(l,f,method)]
                y=np.concatenate(a["y"]); sc=np.concatenate(a["s"])
                iy=np.asarray(a["img_y"],int); isc=np.asarray(a["img_s"],float)
                if len(np.unique(y))<2 or len(np.unique(iy))<2:
                    raise RuntimeError(f"{obj} L{l} fold{f} {method}: invalid heldout classes")
                ii=pd.DataFrame(image_rows)
                if method=="1nn":
                    sub=ii[(ii.object==obj)&(ii.layer==l)&(ii.fold==f)&(ii.method=="1nn")]
                    probe=""
                else:
                    sub=ii[(ii.object==obj)&(ii.layer==l)&(ii.fold==f)&(ii.method=="probe")&(ii.probe==method)]
                    probe=method
                bad=sub[sub.image_y==1]
                fold_metric_rows.append(dict(
                    object=obj,layer=l,fold=f,method=("1nn" if method=="1nn" else "probe"),probe=probe,
                    n_patch=len(y),n_defect_patch=int(y.sum()),n_images=len(iy),
                    patch_auc=roc_auc_score(y,sc),
                    patch_ap=average_precision_score(y,sc),
                    image_auc=roc_auc_score(iy,isc),
                    tail_precision_bad=float(bad.tail_precision.mean()),
                    tail_recall_bad=float(bad.tail_recall.mean())
                ))

    # object-layer macro summaries
    fm=pd.DataFrame([x for x in fold_metric_rows if x["object"]==obj])
    for l in LAYERS:
        base=fm[(fm.layer==l)&(fm.method=="1nn")]
        br={m:float(base[m].mean()) for m in ["patch_auc","patch_ap","image_auc","tail_precision_bad","tail_recall_bad"]}
        object_layer_rows.append(dict(object=obj,layer=l,method="1nn",probe="",
                                       **br,n_folds=len(base)))
        for p in CFG["probes"]:
            q=fm[(fm.layer==l)&(fm.method=="probe")&(fm.probe==p)]
            rr={m:float(q[m].mean()) for m in ["patch_auc","patch_ap","image_auc","tail_precision_bad","tail_recall_bad"]}
            object_layer_rows.append(dict(object=obj,layer=l,method="probe",probe=p,
                                           **rr,n_folds=len(q)))

    pd.DataFrame(fold_metric_rows).to_csv(OUT/"fold_metrics.csv",index=False)
    pd.DataFrame(image_rows).to_csv(OUT/"per_image_metrics.csv",index=False)
    pd.DataFrame(object_layer_rows).to_csv(OUT/"object_layer_metrics.csv",index=False)

    # Close/delete all Python references before removing memmap files on Windows.
    del models,banks,acc
    gc.collect()
    if not CFG.get("keep_temp_features",False):
        shutil.rmtree(tmp)
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\nR6-G main experiment complete:",OUT)
