# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,sys,zlib,json,gc,time
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images

DEV="cuda" if torch.cuda.is_available() else "cpu"
GT_THR=.10
SEED=20260930
OBJECTS=["screw","macaroni2","pcb2","bottle","cable","chewinggum"]
HARD=["screw","macaroni2","pcb2"]
OUT=ROOT/"results"/"model_v0"/"metrics"/"rc_final_direct"
OUT.mkdir(parents=True,exist_ok=True)

def l2_np(x):
    x=np.asarray(x,np.float32)
    return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),1e-12,None)

def build_bank(c,ids):
    offs=np.asarray(c["offsets"]); feats=c["feats"]; dim=int(feats.shape[-1])
    lengths=[int(offs[i+1]-offs[i]) for i in ids]
    total=sum(lengths)
    dtype=torch.float16 if DEV=="cuda" else torch.float32
    bank=torch.empty((total,dim),device=DEV,dtype=dtype)
    slots=np.empty(total,np.int16); poss=np.empty(total,np.int32)
    cur=0
    for slot,i in enumerate(ids):
        a,b=int(offs[i]),int(offs[i+1]); arr=np.asarray(feats[a:b],np.float32)
        for s in range(0,len(arr),32768):
            z=torch.from_numpy(arr[s:s+32768]).to(DEV)
            z=F.normalize(z.float(),dim=1)
            bank[cur+s:cur+s+len(z)]=z.to(dtype)
        slots[cur:cur+len(arr)]=slot
        poss[cur:cur+len(arr)]=np.arange(len(arr),dtype=np.int32)
        cur+=len(arr)
    return bank,slots,poss

@torch.inference_mode()
def knn_topk(arr,bank,k=10,qchunk=256,bchunk=16384):
    arr=np.asarray(arr,np.float32)
    allv=[]; alli=[]
    for qs in range(0,len(arr),qchunk):
        q=torch.from_numpy(arr[qs:qs+qchunk]).to(DEV)
        q=F.normalize(q.float(),dim=1).to(bank.dtype)
        m=len(q)
        bv=torch.full((m,k),-2.0,device=DEV,dtype=torch.float32)
        bi=torch.full((m,k),-1,device=DEV,dtype=torch.long)
        for bs in range(0,len(bank),bchunk):
            b=bank[bs:bs+bchunk]
            sim=(q@b.T).float()
            kk=min(k,sim.shape[1])
            tv,ti=torch.topk(sim,k=kk,dim=1,largest=True,sorted=True)
            ti=ti+bs
            cv=torch.cat([bv,tv],dim=1); ci=torch.cat([bi,ti],dim=1)
            bv,sel=torch.topk(cv,k=k,dim=1,largest=True,sorted=True)
            bi=torch.gather(ci,1,sel)
        allv.append((1.0-bv).cpu().numpy())
        alli.append(bi.cpu().numpy())
    return np.concatenate(allv),np.concatenate(alli)

def top1mean(v):
    v=np.asarray(v,float)
    k=max(1,int(np.ceil(.01*len(v))))
    return float(np.partition(v,len(v)-k)[-k:].mean())

def eval_bank(te,bank):
    offs=np.asarray(te["offsets"]); types=np.asarray([str(x) for x in te["types"]])
    patch_y=[]; patch_s=[]; image_y=[]; image_s=[]
    good_scores=[]; bad_scores=[]
    for i in range(len(types)):
        a,b=int(offs[i]),int(offs[i+1])
        d,_=knn_topk(te["feats"][a:b],bank,k=1)
        d=d[:,0]
        gt=np.asarray(te["gt_frac"][a:b],float)
        y=np.full(len(gt),-1,np.int8); y[gt==0]=0; y[gt>GT_THR]=1
        keep=y>=0
        patch_y.append(y[keep]); patch_s.append(d[keep])
        sc=top1mean(d[keep])
        iy=int(types[i]=="bad"); image_y.append(iy); image_s.append(sc)
        (bad_scores if iy else good_scores).append(sc)
    y=np.concatenate(patch_y); s=np.concatenate(patch_s)
    return dict(
        patch_auc=float(roc_auc_score(y,s)),
        patch_ap=float(average_precision_score(y,s)),
        image_auc=float(roc_auc_score(image_y,image_s)),
        good_tail_median=float(np.median(good_scores)),
        good_tail_p95=float(np.quantile(good_scores,.95)),
        bad_tail_median=float(np.median(bad_scores))
    )

def nested_ids(n,obj,seed):
    rng=np.random.default_rng(SEED+zlib.crc32(obj.encode())+seed*1009)
    return rng.permutation(n)

def run_cov2():
    rows=[]; dup=[]
    shots=[1,2,4,8,16,32,64]
    for obj in OBJECTS:
        print("COV2",obj,flush=True)
        _,_,tr,te=load_obj("r0",obj); tc=tr["final"]; ec=te["final"]
        ntr=len(tc["offsets"])-1
        full_done=False
        for seed in range(5):
            perm=nested_ids(ntr,obj,seed)
            for shot in shots:
                if shot>ntr: continue
                ids=np.sort(perm[:shot])
                bank,_,_=build_bank(tc,ids)
                m=eval_bank(ec,bank)
                rows.append(dict(object=obj,seed=seed,support=str(shot),n_support=shot,**m))
                if shot==1:
                    # Exact-cardinality negative control by literal bank repetition.
                    target=min(4,ntr)
                    n1=len(bank)
                    rep=max(1,int(np.ceil(target*n1/max(n1,1))))
                    dbank=bank.repeat((rep,1))
                    oe=np.asarray(ec["offsets"]); diffs=[]
                    for i in range(min(3,len(ec["types"]))):
                        a,b=int(oe[i]),int(oe[i+1])
                        d1,_=knn_topk(ec["feats"][a:b],bank,k=1)
                        dd,_=knn_topk(ec["feats"][a:b],dbank,k=1)
                        diffs.append(float(np.max(np.abs(d1-dd))))
                    dup.append(dict(object=obj,seed=seed,max_abs_diff=max(diffs),repeat_factor=rep))
                    del dbank
                del bank
                if DEV=="cuda": torch.cuda.empty_cache()
            if not full_done:
                bank,_,_=build_bank(tc,np.arange(ntr))
                m=eval_bank(ec,bank)
                rows.append(dict(object=obj,seed=-1,support="all",n_support=ntr,**m))
                del bank; full_done=True
                if DEV=="cuda": torch.cuda.empty_cache()
        del tr,te; gc.collect()
    R=pd.DataFrame(rows); D=pd.DataFrame(dup)
    R.to_csv(OUT/"RC_COV2_curve.csv",index=False); D.to_csv(OUT/"RC_COV2_duplicate_control.csv",index=False)
    if (D.max_abs_diff>1e-7).any():
        raise RuntimeError("COV2 duplicate-bank negative control failed")
    summ=[]
    for obj in OBJECTS:
        q=R[R.object==obj]
        s4=q[q.support=="4"]; full=q[q.support=="all"]
        if len(s4)==0 or len(full)!=1: continue
        di=float(full.iloc[0].image_auc-s4.image_auc.mean())
        da=float(full.iloc[0].patch_ap-s4.patch_ap.mean())
        rescue=bool(obj in HARD and di>=.10 and da>=.20 and full.iloc[0].image_auc>=.90)
        summ.append(dict(object=obj,mean4_image_auc=float(s4.image_auc.mean()),
                         full_image_auc=float(full.iloc[0].image_auc),delta_image_auc=di,
                         mean4_patch_ap=float(s4.patch_ap.mean()),full_patch_ap=float(full.iloc[0].patch_ap),
                         delta_patch_ap=da,coverage_rescue=rescue))
    S=pd.DataFrame(summ)
    hard_n=int(S[S.object.isin(HARD)].coverage_rescue.sum())
    if hard_n>=2: verdict="UNDERCOVERAGE_ROOT_SUPPORTED"
    elif ((S[S.object.isin(HARD)].delta_image_auc>=.05)|(S[S.object.isin(HARD)].delta_patch_ap>=.10)).any():
        verdict="UNDERCOVERAGE_CONTRIBUTOR_ONLY"
    else: verdict="UNDERCOVERAGE_NOT_SUPPORTED"
    S.to_csv(OUT/"RC_COV2_summary.csv",index=False)
    pd.DataFrame([dict(verdict=verdict,hard_rescue_count=hard_n)]).to_csv(OUT/"RC_COV2_VERDICT.csv",index=False)
    return verdict

def cheb(indices,G,C):
    if not len(indices) or not len(G): return np.full(len(indices),np.inf)
    ar,ac=np.divmod(indices,C); br,bc=np.divmod(G,C)
    return np.maximum(np.abs(ar[:,None]-br[None,:]),np.abs(ac[:,None]-bc[None,:])).min(1)

def collect_dir_candidates(obj):
    _,_,tr,te=load_obj("r0",obj); tc=tr["final"]; ec=te["final"]
    ids=np.asarray(draw_images(tc["offsets"],4,0,obj),int)
    bank,slots,_=build_bank(tc,ids)
    oe=np.asarray(ec["offsets"]); names=np.asarray([str(x) for x in ec["names"]]); types=np.asarray([str(x) for x in ec["types"]])
    scal=[]; qvec=[]; avec=[]; uvec=[]
    for i in range(len(types)):
        if types[i]!="bad": continue
        a,b=int(oe[i]),int(oe[i+1]); arr=np.asarray(ec["feats"][a:b],np.float32)
        gt=np.asarray(ec["gt_frac"][a:b],float)
        G=np.where(gt>GT_THR)[0]; clean=np.where(gt==0)[0]
        if not len(G) or not len(clean): continue
        grid=tuple(map(int,ec["grids"][i])); C=grid[1]
        far=clean[cheb(clean,G,C)>4]
        if not len(far): continue
        dist,idx=knn_topk(arr,bank,k=10)
        k=max(1,int(round(.01*len(arr))))
        tail=np.argpartition(dist[:,0],len(arr)-k)[-k:]
        NH=np.intersect1d(tail,far)
        if not len(NH): continue
        qn=l2_np(arr)
        use=np.r_[G,NH]; labs=np.r_[np.ones(len(G),int),np.zeros(len(NH),int)]
        anc_idx=idx[use,0]
        an=bank[torch.as_tensor(anc_idx,device=DEV)].float().cpu().numpy()
        uv=qn[use]-an; uv=l2_np(uv)
        defect_type=names[i].split("/")[0]
        for j,pidx in enumerate(use):
            scal.append(dict(row=len(scal),object=obj,image_id=i,image=names[i],defect_type=defect_type,
                             label=int(labs[j]),patch_index=int(pidx),anchor_slot=int(slots[anc_idx[j]]),
                             d1=float(dist[pidx,0]),d2=float(dist[pidx,1]),d5=float(dist[pidx,4]),d10=float(dist[pidx,9])))
            qvec.append(qn[pidx]); avec.append(an[j]); uvec.append(uv[j])
    del bank,tr,te; gc.collect()
    if DEV=="cuda": torch.cuda.empty_cache()
    return pd.DataFrame(scal),np.asarray(qvec,np.float32),np.asarray(avec,np.float32),np.asarray(uvec,np.float32)

def make_pairs(df):
    cols=["d1","d2","d5","d10"]
    sd=df[cols].std(ddof=1).replace(0,np.nan)
    pairs=[]; pid=0
    for (img,slot),g in df.groupby(["image_id","anchor_slot"]):
        D=g[g.label==1]; N=g[g.label==0]
        if len(D)==0 or len(N)==0: continue
        zd=D[cols].to_numpy()/sd.to_numpy(); zn=N[cols].to_numpy()/sd.to_numpy()
        cost=np.sqrt(((zd[:,None,:]-zn[None,:,:])**2).sum(2))
        rr,cc=linear_sum_assignment(cost)
        for r,c in zip(rr,cc):
            dif=np.abs(zd[r]-zn[c])
            if np.all(dif<=.20):
                pairs.append(dict(pair_id=pid,d_row=int(D.iloc[r].row),n_row=int(N.iloc[c].row),
                                  image_id=int(img),defect_type=str(D.iloc[r].defect_type),
                                  **{f"abs_z_{col}":float(dif[ii]) for ii,col in enumerate(cols)}))
                pid+=1
    return pd.DataFrame(pairs)

def matched_arrays(df,Q,A,U,pairs):
    rid=np.ravel(np.c_[pairs.d_row.to_numpy(int),pairs.n_row.to_numpy(int)])
    # np.c_ flattened row-major => D,N per pair
    Xrad=df.set_index("row").loc[rid,["d1","d2","d5","d10"]].to_numpy(float)
    y=np.tile([1,0],len(pairs))
    groups=np.repeat(pairs.image_id.to_numpy(int),2)
    types=np.repeat(pairs.defect_type.astype(str).to_numpy(),2)
    return rid,y,groups,types,Xrad,Q[rid],A[rid],U[rid]

def smds(X,y):
    out=[]
    for j in range(X.shape[1]):
        a=X[y==1,j]; b=X[y==0,j]
        sp=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
        out.append(float((np.mean(a)-np.mean(b))/sp) if sp>0 else np.inf)
    return np.asarray(out)

def oof_scores(X,y,groups,kind="logreg",folds=None):
    if folds is None:
        gkf=GroupKFold(n_splits=5)
        folds=list(gkf.split(X,y,groups))
    pred=np.full(len(y),np.nan,float)
    for tr,te in folds:
        sc=StandardScaler().fit(X[tr]); ztr=sc.transform(X[tr]); zte=sc.transform(X[te])
        if kind=="logreg":
            m=LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=3000,random_state=SEED)
        else:
            m=LinearSVC(C=1.0,class_weight="balanced",max_iter=10000,random_state=SEED)
        m.fit(ztr,y[tr])
        pred[te]=m.decision_function(zte)
    return pred,folds

def metrics(y,s):
    return float(roc_auc_score(y,s)),float(average_precision_score(y,s))

def run_dir():
    allsum=[]; pair_files={}
    for obj in OBJECTS:
        print("DIR",obj,flush=True)
        df,Q,A,U=collect_dir_candidates(obj)
        pairs=make_pairs(df)
        if len(pairs)==0:
            allsum.append(dict(object=obj,n_pairs=0,matching_valid=False))
            continue
        rid,y,groups,types,Xrad,Xq,Xa,Xu=matched_arrays(df,Q,A,U,pairs)
        s=smds(Xrad,y); valid=bool((obj!="screw") or (len(pairs)>=50 and np.all(np.abs(s)<=.10)))
        n_groups=len(np.unique(groups))
        if n_groups<2:
            allsum.append(dict(object=obj,n_pairs=len(pairs),matching_valid=False,n_groups=n_groups))
            pairs.to_csv(OUT/f"RC_DIR_pairs_{obj}.csv",index=False)
            continue
        folds=list(GroupKFold(n_splits=min(5,n_groups)).split(Xrad,y,groups))
        arms={"d1":Xrad[:,[0]],"radial4":Xrad,"anchor":Xa,"direction":Xu,"raw":Xq}
        rec={"object":obj,"n_pairs":len(pairs),"matching_valid":valid,
             "smd_d1":s[0],"smd_d2":s[1],"smd_d5":s[2],"smd_d10":s[3]}
        for arm,X in arms.items():
            for kind in ["logreg","linsvm"]:
                pr,_=oof_scores(X,y,groups,kind,folds)
                au,ap=metrics(y,pr); rec[f"{arm}_{kind}_auc"]=au; rec[f"{arm}_{kind}_ap"]=ap
        # permutation only for screw
        if obj=="screw":
            obs=rec["direction_logreg_auc"]; rng=np.random.default_rng(SEED); ge=0
            # row positions 2p,2p+1 correspond to pair D,N
            for rep in range(500):
                Up=Xu.copy()
                flips=rng.random(len(pairs))<.5
                for p,fl in enumerate(flips):
                    if fl:
                        i,j=2*p,2*p+1
                        Up[[i,j]]=Up[[j,i]]
                ps,_=oof_scores(Up,y,groups,"logreg",folds)
                au,_=metrics(y,ps)
                ge+=int(au>=obs)
            rec["pairswap_p"]=(1+ge)/501
            # leave-one-defect-type-out
            pred=np.full(len(y),np.nan); eligible=0
            for typ in sorted(set(types)):
                tei=np.where(types==typ)[0]; tri=np.where(types!=typ)[0]
                npairs=len(tei)//2
                if npairs<10 or len(np.unique(y[tri]))<2: continue
                sc=StandardScaler().fit(Xu[tri])
                m=LogisticRegression(C=1.0,class_weight="balanced",solver="liblinear",max_iter=3000,random_state=SEED)
                m.fit(sc.transform(Xu[tri]),y[tri]); pred[tei]=m.decision_function(sc.transform(Xu[tei]))
                eligible+=1
            keep=np.isfinite(pred)
            rec["loto_types"]=eligible
            rec["loto_auc"]=float(roc_auc_score(y[keep],pred[keep])) if eligible>=1 else np.nan
        allsum.append(rec)
        pairs.to_csv(OUT/f"RC_DIR_pairs_{obj}.csv",index=False)
        del df,Q,A,U; gc.collect()
    S=pd.DataFrame(allsum); S.to_csv(OUT/"RC_DIR_summary.csv",index=False)
    q=S[S.object=="screw"]
    if len(q)!=1: verdict="MATCHING_INSUFFICIENT"
    else:
        r=q.iloc[0]
        Aok=bool(r.n_pairs>=50 and max(abs(r.smd_d1),abs(r.smd_d2),abs(r.smd_d5),abs(r.smd_d10))<=.10)
        if not Aok: verdict="MATCHING_INSUFFICIENT"
        else:
            B=bool(r.direction_logreg_auc>=.75 and r.direction_linsvm_auc>=.70)
            C=bool(r.direction_logreg_auc-r.radial4_logreg_auc>=.15)
            D=bool(r.anchor_logreg_auc<=.65 or r.direction_logreg_auc-r.anchor_logreg_auc>=.10)
            E=bool(r.pairswap_p<.01)
            Ff=bool(r.loto_types>=3 and r.loto_auc>=.70)
            if Aok and B and C and D and E and Ff: verdict="DIRECTIONAL_IDENTIFIABILITY_SUPPORTED"
            elif not D: verdict="ANCHOR_SEMANTIC_CONFOUNDED"
            else: verdict="DIRECTIONAL_HYPOTHESIS_REJECTED"
    pd.DataFrame([dict(verdict=verdict)]).to_csv(OUT/"RC_DIR_VERDICT.csv",index=False)
    return verdict

def finalize(covv,dirv):
    s2=ROOT/"rootcause_rc"/"results_stage2"/"RC_STAGE2_VERDICT.csv"
    arch="UNKNOWN"
    if s2.exists(): arch=str(pd.read_csv(s2).iloc[0].architecture_verdict)
    if covv=="UNDERCOVERAGE_ROOT_SUPPORTED":
        final="NORMAL_SUPPORT_UNDERCOVERAGE_MAJOR_ROOT"
    elif dirv=="DIRECTIONAL_IDENTIFIABILITY_SUPPORTED" and arch=="DOWNSTREAM_READOUT_BOTTLENECK_SUPPORTED":
        final="TWO_STAGE_NORMAL_ONLY_READOUT_BOTTLENECK_ROOT_SUPPORTED"
    elif arch=="DOWNSTREAM_READOUT_BOTTLENECK_SUPPORTED":
        final="READOUT_BOTTLENECK_ROOT_SUPPORTED_EXACT_PATCH_VARIABLE_OPEN"
    else:
        final="ROOT_CAUSE_OPEN"
    pd.DataFrame([dict(cov2_verdict=covv,dir_verdict=dirv,architecture_verdict=arch,
                       final_verdict=final)]).to_csv(OUT/"RC_FINAL_VERDICT.csv",index=False)
    return final

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["cov2","dir","all"],default="all")
    a=ap.parse_args()
    covv="NOT_RUN"; dirv="NOT_RUN"
    if a.mode in ("cov2","all"): covv=run_cov2()
    elif (OUT/"RC_COV2_VERDICT.csv").exists(): covv=str(pd.read_csv(OUT/"RC_COV2_VERDICT.csv").iloc[0].verdict)
    if a.mode in ("dir","all"): dirv=run_dir()
    elif (OUT/"RC_DIR_VERDICT.csv").exists(): dirv=str(pd.read_csv(OUT/"RC_DIR_VERDICT.csv").iloc[0].verdict)
    final=finalize(covv,dirv)
    print("COV2",covv); print("DIR",dirv); print("FINAL",final)

if __name__=="__main__":
    main()
