# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.ndimage import binary_dilation

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images,l2n,nn_dist

DEV="cuda"
GT_THR=.10
RADII=[0,2,4,8,16]
ALPHAS=[.001,.0025,.005,.01,.02,.05]
RANDOM_REPS=100
SEED=20260930

def make_bank(c,ids):
    o=np.asarray(c["offsets"])
    x=np.concatenate([c["feats"][int(o[i]):int(o[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))

def raw_maps(trc,tec,shot,split,obj):
    ids=[int(v) for v in draw_images(trc["offsets"],shot,split,obj)]
    bank=make_bank(trc,ids)
    o=np.asarray(tec["offsets"]); maps=[]
    for i in range(len(tec["types"])):
        x=tec["feats"][int(o[i]):int(o[i+1])].astype(np.float32)
        q=l2n(torch.from_numpy(x).to(DEV))
        with torch.inference_mode():
            d=nn_dist(q,bank).cpu().numpy().astype(np.float64)
        gh,gw=[int(v) for v in tec["grids"][i]]
        if gh*gw!=len(d):
            raise RuntimeError(f"{obj}/{i}: grid mismatch")
        maps.append(d.reshape(gh,gw))
    del bank
    return ids,maps

def topmean_masked(x,valid,alpha):
    v=np.asarray(x[valid],float).ravel()
    if len(v)==0:
        return np.nan
    k=max(1,int(round(alpha*len(v))))
    k=min(k,len(v))
    return float(np.partition(v,len(v)-k)[-k:].mean())

def pairwise_auc_from_scores(bad_scores,good_scores_lists):
    wins=ties=den=0
    for b,gs in zip(bad_scores,good_scores_lists):
        gs=np.asarray(gs,float)
        wins += int((b>gs).sum())
        ties += int((b==gs).sum())
        den += len(gs)
    return 100.0*(wins+0.5*ties)/den if den else np.nan

def dilate_chebyshev(mask,r):
    if r==0:
        return mask.copy()
    # square structuring element = Chebyshev radius
    st=np.ones((2*r+1,2*r+1),dtype=bool)
    return binary_dilation(mask,structure=st)

def translated_masks(mask,n,rng):
    ys,xs=np.where(mask)
    if len(ys)==0:
        return []
    H,W=mask.shape
    miny,maxy=int(ys.min()),int(ys.max())
    minx,maxx=int(xs.min()),int(xs.max())
    dy_lo=-miny; dy_hi=(H-1)-maxy
    dx_lo=-minx; dx_hi=(W-1)-maxx

    candidates=[(dy,dx) for dy in range(dy_lo,dy_hi+1)
                        for dx in range(dx_lo,dx_hi+1)
                        if not (dy==0 and dx==0)]
    if not candidates:
        return []
    take=min(n,len(candidates))
    sel=rng.choice(len(candidates),size=take,replace=False)
    out=[]
    for ii in np.atleast_1d(sel):
        dy,dx=candidates[int(ii)]
        m=np.zeros_like(mask,dtype=bool)
        m[ys+dy,xs+dx]=True
        out.append(m)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default=r"r6a1d_defect_excision\R6A1D_SELECTION.csv")
    ap.add_argument("--outdir",default=r"results\model_v0\metrics\r6a1d_defect_excision")
    a=ap.parse_args()

    sel=pd.read_csv(a.selection)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)

    per_bad=[]; object_rows=[]; random_rows=[]

    for _,rr in sel.iterrows():
        obj=str(rr.object); shot=int(rr.shot); split=int(rr["split"])
        print("\nRUN",obj,flush=True)

        _,_,tr,te=load_obj("r0",obj)
        trc,tec=tr["final"],te["final"]
        _,maps=raw_maps(trc,tec,shot,split,obj)

        types=np.asarray([str(v) for v in tec["types"]])
        names=np.asarray([str(v) for v in tec["names"]])
        offs=np.asarray(tec["offsets"])
        bad_ids=np.where(types=="bad")[0]
        good_ids=np.where(types=="good")[0]

        # Require one common grid per object for exact same-coordinate masking.
        grids=[tuple(map(int,g)) for g in tec["grids"]]
        if len(set(grids))!=1:
            raise RuntimeError(f"{obj}: multiple test grids found; protocol says STOP.")
        H,W=grids[0]

        for rad in RADII:
            centered_by_alpha={a: ([],[]) for a in ALPHAS}  # bad scores, list of good score lists
            random_aucs={a: [] for a in ALPHAS}
            removed_fracs=[]
            valid_counts=[]

            # Store bad-specific centered masks and scores first
            bad_payload=[]
            for bi in bad_ids:
                aa,bb=int(offs[bi]),int(offs[bi+1])
                gt=np.asarray(tec["gt_frac"][aa:bb],float).reshape(H,W)
                G=gt>GT_THR
                if not G.any():
                    continue
                exc=dilate_chebyshev(G,rad)
                valid=~exc
                if valid.sum()<10:
                    continue

                removed_fracs.append(float(exc.mean()))
                valid_counts.append(int(valid.sum()))

                payload={"bi":int(bi),"G":G,"exc":exc,"valid":valid}
                bad_payload.append(payload)

                for alpha in ALPHAS:
                    bs=topmean_masked(maps[bi],valid,alpha)
                    gs=[topmean_masked(maps[gi],valid,alpha) for gi in good_ids]
                    centered_by_alpha[alpha][0].append(bs)
                    centered_by_alpha[alpha][1].append(gs)
                    per_bad.append(dict(
                        object=obj,radius=rad,alpha=alpha,
                        bad_image=names[bi],
                        n_defect=int(G.sum()),
                        n_removed=int(exc.sum()),
                        n_valid=int(valid.sum()),
                        removed_fraction=float(exc.mean()),
                        bad_score=bs,
                        mean_good_score=float(np.mean(gs)),
                        pairwise_win_rate=float(np.mean(
                            (bs>np.asarray(gs,float)) + .5*(bs==np.asarray(gs,float))
                        ))
                    ))

            # Centered pairwise AUC
            centered_auc={}
            for alpha in ALPHAS:
                bs,gs=centered_by_alpha[alpha]
                centered_auc[alpha]=pairwise_auc_from_scores(bs,gs)

            # Shape/area-matched translated mask null.
            # Replicate r uses one independently translated mask per bad image.
            rng=np.random.default_rng(SEED + int(zlib.crc32(obj.encode()))%100000 + rad*1009)
            translated_by_bad=[]
            for payload in bad_payload:
                ms=translated_masks(payload["exc"],RANDOM_REPS,rng)
                translated_by_bad.append(ms)

            nrep=min([len(x) for x in translated_by_bad], default=0)
            if nrep>0:
                for rep in range(nrep):
                    for alpha in ALPHAS:
                        bss=[]; gss=[]
                        for payload,ms in zip(bad_payload,translated_by_bad):
                            valid=~ms[rep]
                            bi=payload["bi"]
                            bs=topmean_masked(maps[bi],valid,alpha)
                            gs=[topmean_masked(maps[gi],valid,alpha) for gi in good_ids]
                            bss.append(bs); gss.append(gs)
                        ra=pairwise_auc_from_scores(bss,gss)
                        random_aucs[alpha].append(ra)
                        random_rows.append(dict(
                            object=obj,radius=rad,alpha=alpha,rep=rep,
                            random_pairwise_auc=ra
                        ))

            for alpha in ALPHAS:
                arr=np.asarray(random_aucs[alpha],float)
                ca=centered_auc[alpha]
                object_rows.append(dict(
                    object=obj,radius=rad,alpha=alpha,
                    pairwise_auc_centered=ca,
                    random_n=int(len(arr)),
                    random_mean=float(np.mean(arr)) if len(arr) else np.nan,
                    random_q05=float(np.quantile(arr,.05)) if len(arr) else np.nan,
                    random_q95=float(np.quantile(arr,.95)) if len(arr) else np.nan,
                    centered_minus_random_mean=ca-float(np.mean(arr)) if len(arr) else np.nan,
                    mean_removed_fraction=float(np.mean(removed_fracs)) if removed_fracs else np.nan,
                    mean_valid_count=float(np.mean(valid_counts)) if valid_counts else np.nan,
                    n_bad_used=len(bad_payload),
                    n_good=len(good_ids)
                ))

            pd.DataFrame(per_bad).to_csv(out/"per_bad_image.csv",index=False)
            pd.DataFrame(object_rows).to_csv(out/"object_radius_alpha_summary.csv",index=False)
            pd.DataFrame(random_rows).to_csv(out/"random_translation_null.csv",index=False)

    print("\nDONE",out)

if __name__=="__main__":
    main()
