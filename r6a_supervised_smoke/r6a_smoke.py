# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0,str(ROOT/"experiments"/"baseline"))
sys.path.insert(0,str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images, l2n, nn_dist

DEV="cuda"
GT_THR=.10
MAX_PER_CLASS_PER_IMAGE=512
SEED=20260930

def top1mean(v):
    v=np.asarray(v,float).ravel()
    k=max(1,int(np.ceil(.01*len(v))))
    return float(np.partition(v,len(v)-k)[-k:].mean())

def make_bank(c,ids):
    offs=c["offsets"]
    x=np.concatenate([c["feats"][int(offs[i]):int(offs[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))

def baseline_scores(trc,tec,shot,split,obj):
    ids=[int(v) for v in draw_images(trc["offsets"],shot,split,obj)]
    bank=make_bank(trc,ids)
    offs=tec["offsets"]; out=[]
    for i in range(len(tec["types"])):
        x=tec["feats"][int(offs[i]):int(offs[i+1])].astype(np.float32)
        q=l2n(torch.from_numpy(x).to(DEV))
        with torch.inference_mode():
            d=nn_dist(q,bank).cpu().numpy().astype(np.float64)
        out.append(d)
    del bank
    return ids,out

def extract_images(c):
    offs=np.asarray(c["offsets"])
    rows=[]
    for i in range(len(c["types"])):
        a,b=int(offs[i]),int(offs[i+1])
        X=np.asarray(c["feats"][a:b],np.float32)
        gt=np.asarray(c["gt_frac"][a:b],float)
        y=np.full(len(gt),-1,np.int8)
        y[gt==0]=0
        y[gt>GT_THR]=1
        rows.append(dict(
            image_id=i,name=str(c["names"][i]),type=str(c["types"][i]),
            X=X,y=y
        ))
    return rows

def choose_folds(images):
    # Stratify at IMAGE level by good/bad. This guarantees no patch leakage.
    yimg=np.array([1 if r["type"]=="bad" else 0 for r in images])
    idx=np.arange(len(images))
    for nfold in [5,3]:
        if np.bincount(yimg,minlength=2).min() < nfold:
            continue
        skf=StratifiedKFold(n_splits=nfold,shuffle=True,random_state=SEED)
        folds=np.empty(len(images),int)
        good=True
        for f,(_,te) in enumerate(skf.split(idx,yimg)):
            folds[te]=f
            # verify heldout fold has eligible defect and clean patches overall
            yy=np.concatenate([images[i]["y"] for i in te])
            yy=yy[yy>=0]
            if len(np.unique(yy))<2:
                good=False
        if good: return folds,nfold
    raise RuntimeError("Could not create leakage-free 5/3-fold image-level CV with both patch classes.")

def sample_train(images,train_ids,rng):
    Xs=[]; ys=[]
    for i in train_ids:
        X=images[i]["X"]; y=images[i]["y"]
        for cls in [0,1]:
            ids=np.where(y==cls)[0]
            if not len(ids): continue
            if len(ids)>MAX_PER_CLASS_PER_IMAGE:
                ids=rng.choice(ids,MAX_PER_CLASS_PER_IMAGE,replace=False)
            Xs.append(X[ids]); ys.append(np.full(len(ids),cls,np.int8))
    X=np.concatenate(Xs); y=np.concatenate(ys)
    if len(np.unique(y))<2: raise RuntimeError("Training fold missing a patch class.")
    return X,y

def eval_fold(images,test_ids,baseline_maps,model,scaler,probe,obj,fold,patch_rows,image_rows):
    pY=[]; pS=[]; pB=[]
    for i in test_ids:
        X=images[i]["X"]; y=images[i]["y"]; keep=y>=0
        Xt=scaler.transform(X[keep])
        if probe=="logreg":
            s=model.predict_proba(Xt)[:,1]
        else:
            s=model.decision_function(Xt)
        yy=y[keep]
        bb=np.asarray(baseline_maps[i])[keep]
        pY.append(yy); pS.append(s); pB.append(bb)
        patch_rows.append(dict(
            object=obj,fold=fold,probe=probe,image=images[i]["name"],
            n_eligible=int(keep.sum()),n_defect=int((yy==1).sum()),n_clean=int((yy==0).sum()),
            patch_auc_probe=np.nan if len(np.unique(yy))<2 else roc_auc_score(yy,s),
            patch_ap_probe=np.nan if yy.sum()==0 else average_precision_score(yy,s),
            patch_auc_1nn=np.nan if len(np.unique(yy))<2 else roc_auc_score(yy,bb),
            patch_ap_1nn=np.nan if yy.sum()==0 else average_precision_score(yy,bb)
        ))
        image_rows.append(dict(
            object=obj,fold=fold,probe=probe,image=images[i]["name"],
            image_y=int(images[i]["type"]=="bad"),
            score_probe=top1mean(s),
            score_1nn=top1mean(bb)
        ))
    return np.concatenate(pY),np.concatenate(pS),np.concatenate(pB)

def run_object(obj,shot,split):
    _,layers,tr,te=load_obj("r0",obj)
    trc,tec=tr["final"],te["final"]
    support,baseline_maps=baseline_scores(trc,tec,shot,split,obj)
    images=extract_images(tec)
    folds,nfold=choose_folds(images)

    fold_rows=[]
    for i,r in enumerate(images):
        fold_rows.append(dict(object=obj,image_id=i,image=r["name"],type=r["type"],fold=int(folds[i])))

    patch_rows=[]; image_rows=[]; summary=[]
    for probe in ["logreg","linsvm"]:
        oof_y=[]; oof_s=[]; oof_b=[]
        for f in range(nfold):
            te_ids=np.where(folds==f)[0]; tr_ids=np.where(folds!=f)[0]
            rng=np.random.default_rng(SEED+f)
            Xtr,ytr=sample_train(images,tr_ids,rng)
            scaler=StandardScaler().fit(Xtr)
            Xz=scaler.transform(Xtr)
            if probe=="logreg":
                model=LogisticRegression(C=1.0,class_weight="balanced",max_iter=3000,
                                         solver="liblinear",random_state=SEED)
            else:
                model=LinearSVC(C=1.0,class_weight="balanced",max_iter=10000,random_state=SEED)
            model.fit(Xz,ytr)
            yy,ss,bb=eval_fold(images,te_ids,baseline_maps,model,scaler,probe,obj,f,patch_rows,image_rows)
            oof_y.append(yy); oof_s.append(ss); oof_b.append(bb)

        y=np.concatenate(oof_y); s=np.concatenate(oof_s); b=np.concatenate(oof_b)
        im=pd.DataFrame([r for r in image_rows if r["object"]==obj and r["probe"]==probe])
        summary.append(dict(
            object=obj,probe=probe,n_folds=nfold,n_patch=len(y),n_defect_patch=int(y.sum()),
            patch_auc_probe=roc_auc_score(y,s),
            patch_ap_probe=average_precision_score(y,s),
            patch_auc_1nn=roc_auc_score(y,b),
            patch_ap_1nn=average_precision_score(y,b),
            patch_auc_gain=(roc_auc_score(y,s)-roc_auc_score(y,b))*100,
            image_auc_probe=roc_auc_score(im.image_y,im.score_probe),
            image_auc_1nn=roc_auc_score(im.image_y,im.score_1nn),
            image_auc_gain=(roc_auc_score(im.image_y,im.score_probe)-roc_auc_score(im.image_y,im.score_1nn))*100,
            support_ids=";".join(map(str,support))
        ))
    return fold_rows,patch_rows,image_rows,summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",default=r"r6a_supervised_smoke\R6A_SMOKE_SELECTION.csv")
    ap.add_argument("--outdir",default=r"results\model_v0\metrics\r6a_smoke")
    a=ap.parse_args()
    sel=pd.read_csv(a.selection)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    F=[];P=[];I=[];S=[]
    for _,r in sel.iterrows():
        print("RUN",r.object,flush=True)
        f,p,i,s=run_object(str(r.object),int(r.shot),int(r["split"]))
        F+=f;P+=p;I+=i;S+=s
        pd.DataFrame(F).to_csv(out/"folds.csv",index=False)
        pd.DataFrame(P).to_csv(out/"per_image_patch_metrics.csv",index=False)
        pd.DataFrame(I).to_csv(out/"image_scores.csv",index=False)
        pd.DataFrame(S).to_csv(out/"summary.csv",index=False)
        for z in s:
            print(" ",z["probe"],
                  "patch probe %.3f 1NN %.3f gain %+5.1f | image probe %.3f 1NN %.3f gain %+5.1f" %
                  (z["patch_auc_probe"],z["patch_auc_1nn"],z["patch_auc_gain"],
                   z["image_auc_probe"],z["image_auc_1nn"],z["image_auc_gain"]),flush=True)
    print("DONE",out)

if __name__=="__main__":
    main()
