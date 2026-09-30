# -*- coding: utf-8 -*-
"""
R5 raw-map causal interventions.
MAIN protocol is frozen in R5_PREREGISTRATION_FROZEN.md.

Requires original E:\\work\\freshman raw DINO caches.
Diagnostic use of test GT is allowed only for causal intervention, not for deployable scoring.
"""
from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images, l2n, nn_dist
from tail_calib import mean_top1p

DEV="cuda"
EPS=1e-12
STRENGTH_DOSES=np.array([0,.25,.5,1,2,4],float)
EXTENT_ALPHA=np.array([.001,.0025,.005,.01,.02,.05,.10],float)
COH_ALPHA=np.array([.005,.01,.02,.05],float)

def qshape(x):
    x=np.asarray(x,float).ravel()
    return float(np.log((x.max()+EPS)/(float(mean_top1p(x))+EPS)))

def score3(x):
    x=np.asarray(x,float).ravel()
    return float(mean_top1p(x)), float(x.max()), qshape(x)

def robust_scale(clean):
    clean=np.asarray(clean,float).ravel()
    med=float(np.median(clean))
    mad=float(np.median(np.abs(clean-med)))
    s=1.4826*mad
    if not np.isfinite(s) or s<=1e-10:
        s=float(np.std(clean))
    return max(s,1e-8)

def make_bank(feats,offs,ids):
    a=np.concatenate([feats[int(offs[i]):int(offs[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(a).to(DEV))

def raw_maps(final_tr, final_te, shot, split, obj):
    ids=[int(v) for v in draw_images(final_tr["offsets"],shot,split,obj)]
    bank=make_bank(final_tr["feats"],final_tr["offsets"],ids)
    offs=final_te["offsets"]
    maps=[]
    for i in range(len(final_te["types"])):
        q=torch.from_numpy(final_te["feats"][int(offs[i]):int(offs[i+1])].astype(np.float32)).to(DEV)
        with torch.inference_mode():
            dd=nn_dist(l2n(q),bank).cpu().numpy().astype(np.float64)
        gh,gw=[int(v) for v in final_te["grids"][i]]
        maps.append(dd.reshape(gh,gw))
    del bank
    return ids,maps

def gt_mask_for(final_te,i):
    o=final_te["offsets"]; a=int(o[i]); b=int(o[i+1])
    gh,gw=[int(v) for v in final_te["grids"][i]]
    gt=np.asarray(final_te["gt_frac"][a:b],float).reshape(gh,gw)
    return gt>0.10, gt==0.0

def safe_rho(x,y):
    r=spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else np.nan

def top_mask(x,alpha):
    flat=x.ravel()
    k=max(1,int(round(alpha*flat.size)))
    idx=np.argpartition(flat,-k)[-k:]
    m=np.zeros(flat.size,bool); m[idx]=True
    return m.reshape(x.shape),k

def spatial_stats(x,alpha):
    m,k=top_mask(x,alpha)
    lab,n=label(m,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],int))
    if n:
        counts=np.bincount(lab.ravel())[1:]
        lcc=float(counts.max()/k)
    else:
        lcc=0.0
    # fraction of possible selected-node 4-neighbor half-edges realized among selected nodes
    e=int((m[:,:-1]&m[:,1:]).sum() + (m[:-1,:]&m[1:,:]).sum())
    deg=int((m[:,:-1]).sum()+(m[:,1:]).sum()+(m[:-1,:]).sum()+(m[1:,:]).sum())
    adj=float(2*e/deg) if deg else 0.0
    return lcc,adj

def auc(y,s):
    return float(roc_auc_score(y,np.asarray(s))*100)

def run_cell(obj,shot,split,final_tr,final_te,shuffle_reps,seed):
    t0=time.perf_counter()
    support,maps=raw_maps(final_tr,final_te,shot,split,obj)
    names=np.asarray([str(v) for v in final_te["names"]])
    types=np.asarray([str(v) for v in final_te["types"]])
    y=(types=="bad").astype(int)

    base_top1=np.array([score3(m)[0] for m in maps])
    base_max=np.array([score3(m)[1] for m in maps])
    base_q=np.array([score3(m)[2] for m in maps])

    strength_rows=[]
    for i in np.where(y==1)[0]:
        G,C=gt_mask_for(final_te,int(i))
        if not G.any() or not C.any(): continue
        s=robust_scale(maps[i][C])
        vals=[]
        for lam in STRENGTH_DOSES:
            z=maps[i].copy()
            z[G]=z[G]+float(lam)*s
            t,mx,q=score3(z)
            vals.append((t,mx,q))
            strength_rows.append(dict(
                object=obj,shot=shot,split=split,image=names[i],dose=float(lam),
                n_defect=int(G.sum()),scale=s,top1=t,max=mx,q=q
            ))
        vv=np.asarray(vals)
        rho_t=safe_rho(STRENGTH_DOSES,vv[:,0])
        rho_m=safe_rho(STRENGTH_DOSES,vv[:,1])
        rho_q=safe_rho(STRENGTH_DOSES,vv[:,2])
        # duplicate image-level summary at dose=-1 for easy downstream parsing
        strength_rows.append(dict(
            object=obj,shot=shot,split=split,image=names[i],dose=-1.0,
            n_defect=int(G.sum()),scale=s,top1=rho_t,max=rho_m,q=rho_q
        ))

    # AUROC under strengthening: good images unchanged, bad images GT-strengthened.
    strength_auc=[]
    for lam in STRENGTH_DOSES:
        st=base_top1.copy(); sm=base_max.copy(); sq=base_q.copy()
        for i in np.where(y==1)[0]:
            G,C=gt_mask_for(final_te,int(i))
            if not G.any() or not C.any(): continue
            s=robust_scale(maps[i][C]); z=maps[i].copy(); z[G]+=float(lam)*s
            st[i],sm[i],sq[i]=score3(z)
        strength_auc.append(dict(
            object=obj,shot=shot,split=split,dose=float(lam),
            auc_top1=auc(y,st),auc_max=auc(y,sm),auc_q=auc(y,sq)
        ))

    # R5-B on REAL NORMAL backgrounds.
    extent_rows=[]
    for i in np.where(y==0)[0]:
        x=maps[i].copy(); flat=x.ravel(); N=flat.size
        s=robust_scale(flat)
        h=float(flat.max()+s)  # fixed strength, strictly above original normal max
        order=np.argsort(flat) # coordinatewise increase: replace lowest scores
        vals=[]
        for a in EXTENT_ALPHA:
            k=max(1,int(round(float(a)*N)))
            z=flat.copy()
            idx=order[:k]
            z[idx]=h
            t,mx,q=score3(z)
            vals.append((t,mx,q,k))
            extent_rows.append(dict(
                object=obj,shot=shot,split=split,image=names[i],alpha=float(a),m=k,
                N=N,implant_strength=h,normal_scale=s,top1=t,max=mx,q=q
            ))
        vv=np.asarray(vals,float)
        extent_rows.append(dict(
            object=obj,shot=shot,split=split,image=names[i],alpha=-1.0,m=-1,N=N,
            implant_strength=h,normal_scale=s,
            top1=safe_rho(EXTENT_ALPHA,vv[:,0]),
            max=safe_rho(EXTENT_ALPHA,vv[:,1]),
            q=safe_rho(EXTENT_ALPHA,vv[:,2])
        ))

    # R5-C: rank-position spatial signal, MAIN split 0 only.
    coherence_rows=[]; shuffle_rows=[]
    if int(split)==0:
        rng=np.random.default_rng(seed + abs(hash(obj))%100000 + shot*101)
        for a in COH_ALPHA:
            orig_l=[]; orig_adj=[]
            for x in maps:
                l,ad=spatial_stats(x,float(a)); orig_l.append(l); orig_adj.append(ad)
            obs_l=auc(y,orig_l); obs_a=auc(y,orig_adj)
            coherence_rows.append(dict(
                object=obj,shot=shot,split=split,alpha=float(a),
                auc_lcc=obs_l,auc_adjacency=obs_a,n_images=len(y)
            ))
            sh_l=[]; sh_a=[]
            for r in range(shuffle_reps):
                sl=[]; sa=[]
                for x in maps:
                    p=rng.permutation(x.size).reshape(x.shape)
                    # permutation indexes; preserve exact histogram
                    xp=x.ravel()[p.ravel()].reshape(x.shape)
                    l,ad=spatial_stats(xp,float(a)); sl.append(l); sa.append(ad)
                sh_l.append(auc(y,sl)); sh_a.append(auc(y,sa))
            shuffle_rows.append(dict(
                object=obj,shot=shot,split=split,alpha=float(a),
                shuffle_reps=shuffle_reps,
                lcc_mean=float(np.mean(sh_l)),lcc_q05=float(np.quantile(sh_l,.05)),
                lcc_q95=float(np.quantile(sh_l,.95)),
                adjacency_mean=float(np.mean(sh_a)),adjacency_q05=float(np.quantile(sh_a,.05)),
                adjacency_q95=float(np.quantile(sh_a,.95)),
                obs_lcc=obs_l,obs_adjacency=obs_a
            ))

    cell=dict(
        object=obj,shot=shot,split=split,n_test=len(y),n_bad=int(y.sum()),n_good=int((y==0).sum()),
        baseline_auc_top1=auc(y,base_top1),baseline_auc_max=auc(y,base_max),
        baseline_auc_q=auc(y,base_q),elapsed_sec=time.perf_counter()-t0
    )
    return cell,strength_rows,strength_auc,extent_rows,coherence_rows,shuffle_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default=r"r5_rawmap_causal_interventions\R5_MAIN_SELECTION.csv")
    ap.add_argument("--outdir",default=r"results\model_v0\metrics\r5_rawmap")
    ap.add_argument("--shuffle-reps",type=int,default=100)
    ap.add_argument("--seed",type=int,default=20260930)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")

    sel=pd.read_csv(a.selection)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    files={
        "cells":out/"R5_cells.csv",
        "strength":out/"R5A_strength_per_image.csv",
        "strength_auc":out/"R5A_strength_auc.csv",
        "extent":out/"R5B_extent_normal_background.csv",
        "coherence":out/"R5C_coherence_original.csv",
        "shuffle":out/"R5C_coherence_shuffle.csv",
    }
    data={k:[] for k in files}
    done=set()
    if a.resume and files["cells"].exists():
        for k,p in files.items():
            if p.exists(): data[k]=pd.read_csv(p).to_dict("records")
        old=pd.read_csv(files["cells"])
        done={(str(r.object),int(r.shot),int(r["split"])) for _,r in old.iterrows()}

    for obj,gg in sel.groupby("object",sort=True):
        print("\nLOAD",obj,flush=True)
        _,layers,tr,te=load_obj("r0",obj)
        final_tr=tr["final"]; final_te=te["final"]
        for _,r in gg.sort_values(["shot","split"]).iterrows():
            key=(str(obj),int(r.shot),int(r["split"]))
            if key in done:
                print("SKIP",key); continue
            print("RUN",key,flush=True)
            res=run_cell(obj,key[1],key[2],final_tr,final_te,a.shuffle_reps,a.seed)
            for k,v in zip(["cells","strength","strength_auc","extent","coherence","shuffle"],res):
                if isinstance(v,list): data[k].extend(v)
                else: data[k].append(v)
                pd.DataFrame(data[k]).to_csv(files[k],index=False)
            c=res[0]
            print("  top1 %.2f | max %.2f | q %.2f | %.1fs" %
                  (c["baseline_auc_top1"],c["baseline_auc_max"],c["baseline_auc_q"],c["elapsed_sec"]),flush=True)
        del tr,te
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nDONE",out)

if __name__=="__main__":
    main()
