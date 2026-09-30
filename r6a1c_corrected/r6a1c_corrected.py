# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images,l2n

DEV="cuda"
GT_THR=.10
FAR=4
ALPHA=.01
KS=[1,2,5,10]
N_NEIGH=10
LAYERS=["mid","midlate","final"]
CHUNK=512
SEED=20260930

def bank_of(c,ids):
    o=np.asarray(c["offsets"])
    x=np.concatenate([c["feats"][int(o[i]):int(o[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))

def knn_exact(q,bank,ks,want_idx=0):
    kmax=max(max(ks),want_idx)
    vals={k:np.empty(len(q),np.float64) for k in ks}
    ix=np.empty((len(q),want_idx),np.int64) if want_idx else None
    for s in range(0,len(q),CHUNK):
        qb=q[s:s+CHUNK]
        with torch.inference_mode():
            sim=qb@bank.T
            # largest cosine similarities are nearest in cosine distance
            topv,topi=torch.topk(sim,k=kmax,dim=1,largest=True,sorted=True)
            dist=(1.0-topv).cpu().numpy().astype(np.float64)
            inds=topi.cpu().numpy()
        for k in ks:
            vals[k][s:s+len(qb)]=dist[:,k-1]
        if want_idx:
            ix[s:s+len(qb)]=inds[:,:want_idx]
    return vals,ix

def nnd(q,bank):
    return knn_exact(q,bank,[1])[0][1]

def cheb(a,b,C):
    if not len(a) or not len(b): return np.full(len(a),np.inf)
    ar,ac=np.divmod(a,C); br,bc=np.divmod(b,C)
    return np.maximum(np.abs(ar[:,None]-br[None,:]),np.abs(ac[:,None]-bc[None,:])).min(axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default=r"r6a_supervised_smoke\R6A_SMOKE_SELECTION.csv")
    ap.add_argument("--outdir",default=r"results\model_v0\metrics\r6a1c_corrected")
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    sel=pd.read_csv(a.selection)
    cov_rows=[]; div_rows=[]; layer_rows=[]; det_rows=[]; good_rows=[]

    for _,rr in sel.iterrows():
        obj=str(rr.object); shot=int(rr.shot); split=int(rr["split"])
        print("RUN",obj,flush=True)
        _,_,tr,te=load_obj("r0",obj)
        tf=tr["final"]; ef=te["final"]
        ids=[int(v) for v in draw_images(tf["offsets"],shot,split,obj)]
        oe=np.asarray(ef["offsets"]); ot=np.asarray(tf["offsets"])
        types=np.asarray([str(v) for v in ef["types"]])
        names=np.asarray([str(v) for v in ef["names"]])
        R,C=[int(v) for v in ef["grids"][0]]
        n_tok=int(ot[1]-ot[0])

        # ---------- define groups with exact final-layer 1NN ----------
        bank=bank_of(tf,ids)
        groups={}
        final_d1={}
        for i in range(len(types)):
            if types[i]!="bad": continue
            gt=np.asarray(ef["gt_frac"][int(oe[i]):int(oe[i+1])],float)
            G=np.where(gt>GT_THR)[0]; clean=np.where(gt==0)[0]
            if not len(G) or not len(clean): continue
            far=clean[cheb(clean,G,C)>FAR]
            if not len(far): continue
            q=l2n(torch.from_numpy(ef["feats"][int(oe[i]):int(oe[i+1])].astype(np.float32)).to(DEV))
            dk,ix=knn_exact(q,bank,KS,N_NEIGH)
            k=max(1,int(round(ALPHA*len(q))))
            tail=np.argpartition(dk[1],len(dk[1])-k)[-k:]
            NH=np.intersect1d(tail,far)
            if not len(NH): continue
            rng=np.random.default_rng(SEED+int(zlib.crc32(obj.encode()))%100000+i)
            pool=np.setdiff1d(far,tail)
            if len(pool)<len(NH): continue
            grp={
                "D":G,
                "N_H":NH,
                "N_R_far":rng.choice(pool,len(NH),replace=False),
                "N_R_all":rng.choice(clean,min(len(NH),len(clean)),replace=False)
            }
            groups[i]=grp; final_d1[i]=dk[1]
            for gname,idx in grp.items():
                cov_rows.append(dict(object=obj,image=names[i],group=gname,n=len(idx),
                                     **{f"d{k}":float(np.mean(dk[k][idx])) for k in KS}))
                slots=ix[idx]//n_tok
                pos=ix[idx]%n_tok
                pr,pc=pos//C,pos%C
                div_rows.append(dict(
                    object=obj,image=names[i],group=gname,n=len(idx),
                    distinct_support_images=float(np.mean([len(np.unique(s)) for s in slots])),
                    neighbour_spatial_std=float(np.mean(np.std(pr,axis=1)+np.std(pc,axis=1))),
                    frac_from_one_image=float(np.mean(
                        [np.bincount(s,minlength=len(ids)).max()/N_NEIGH for s in slots]))
                ))
            del q
        del bank

        # ---------- corrected cross-layer query: TEST layer features ----------
        for layer in LAYERS:
            bt=bank_of(tr[layer],ids)
            oe_l=np.asarray(te[layer]["offsets"])
            for i,grp in groups.items():
                # validate patch index compatibility
                n_test=int(oe_l[i+1]-oe_l[i])
                maxidx=max(int(np.max(x)) for x in grp.values() if len(x))
                if maxidx>=n_test:
                    raise RuntimeError(f"{obj}/{layer}/{i}: patch-index mismatch")
                allidx=np.concatenate(list(grp.values()))
                owners=np.concatenate([[g]*len(idx) for g,idx in grp.items()])
                x=te[layer]["feats"][int(oe_l[i]):int(oe_l[i+1])][allidx].astype(np.float32)
                q=l2n(torch.from_numpy(x).to(DEV))
                d=nnd(q,bt); del q
                for gname in grp:
                    m=owners==gname
                    layer_rows.append(dict(object=obj,layer=layer,image=names[i],group=gname,
                                           n=int(m.sum()),d1_mean=float(d[m].mean())))
            del bt

        # ---------- corrected matched true-1NN detectability ----------
        sub={}
        for drop in range(len(ids)):
            keep=[j for j in ids if j!=ids[drop]]
            sub[drop]=bank_of(tf,keep)

        null=[]
        for si,sid in enumerate(ids):
            q=l2n(torch.from_numpy(tf["feats"][int(ot[sid]):int(ot[sid+1])].astype(np.float32)).to(DEV))
            null.append(nnd(q,sub[si])); del q
        null=np.concatenate(null)
        thr=float(np.percentile(null,99.0))

        # bad N_H using matched average over 3-image subbanks
        nh_image_rates=[]
        for i,grp in groups.items():
            q=l2n(torch.from_numpy(ef["feats"][int(oe[i]):int(oe[i+1])].astype(np.float32)).to(DEV))
            dbar=np.mean([nnd(q,sub[d]) for d in range(len(ids))],axis=0); del q
            for gname,idx in grp.items():
                det_rows.append(dict(object=obj,image=names[i],group=gname,n=len(idx),
                                     p99_threshold=thr,frac_over_p99=float(np.mean(dbar[idx]>thr)),
                                     mean_distance=float(np.mean(dbar[idx]))))
            nh=grp["N_H"]
            nh_image_rates.append(float(np.mean(dbar[nh]>thr)))

        # good-image top1% natural control under exactly same matched score
        good_rates=[]; good_means=[]
        for i in np.where(types=="good")[0]:
            q=l2n(torch.from_numpy(ef["feats"][int(oe[i]):int(oe[i+1])].astype(np.float32)).to(DEV))
            dbar=np.mean([nnd(q,sub[d]) for d in range(len(ids))],axis=0); del q
            k=max(1,int(round(ALPHA*len(dbar))))
            tail=np.argpartition(dbar,len(dbar)-k)[-k:]
            rate=float(np.mean(dbar[tail]>thr))
            md=float(np.mean(dbar[tail]))
            good_rows.append(dict(object=obj,image=names[i],group="good_tail",n=k,
                                  p99_threshold=thr,frac_over_p99=rate,mean_distance=md))
            good_rates.append(rate); good_means.append(md)

        # image-level scalar distinguishability diagnostic
        bad_rates=np.asarray(nh_image_rates); gr=np.asarray(good_rates)
        if len(bad_rates) and len(gr):
            yy=np.r_[np.ones(len(bad_rates)),np.zeros(len(gr))]
            ss=np.r_[bad_rates,gr]
            auc_rate=roc_auc_score(yy,ss)
        else:
            auc_rate=np.nan
        print(" ",obj,"p99-rate AUC NH-vs-good =",round(float(auc_rate),3),flush=True)

        for b in sub.values(): del b

        pd.DataFrame(cov_rows).to_csv(out/"support_coverage_corrected.csv",index=False)
        pd.DataFrame(div_rows).to_csv(out/"neighbour_diversity_corrected.csv",index=False)
        pd.DataFrame(layer_rows).to_csv(out/"layer_consistency_corrected.csv",index=False)
        pd.DataFrame(det_rows).to_csv(out/"detectability_bad_groups_corrected.csv",index=False)
        pd.DataFrame(good_rows).to_csv(out/"good_tail_control_corrected.csv",index=False)

    print("DONE",out)

if __name__=="__main__":
    main()
