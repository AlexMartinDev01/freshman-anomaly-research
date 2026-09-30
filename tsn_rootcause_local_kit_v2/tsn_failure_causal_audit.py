# -*- coding: utf-8 -*-
"""
TSN formal failure causal audit.
Run from ORIGINAL project root E:\work\freshman.

This is DIAGNOSTIC ONLY. It does not alter the frozen 243-cell result.
It uses test labels only for oracle diagnosis, never for a deployable method.

Outputs:
- causal_audit_cells.csv
- causal_audit_per_image.csv
- causal_audit_bank_level.csv
"""
from __future__ import annotations
import argparse, os, sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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

def qshape(d):
    d=np.asarray(d,dtype=np.float64).ravel()
    eps=1e-12
    return float(np.log((float(d.max())+eps)/(float(mean_top1p(d))+eps)))

def make_bank(feats,offs,ids):
    x=np.concatenate([feats[int(offs[i]):int(offs[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))

def sl(cache,i):
    o=cache["offsets"]
    return cache["feats"][int(o[i]):int(o[i+1])]

def score_q(cache,i,bank):
    q=l2n(torch.from_numpy(np.asarray(sl(cache,i),np.float32)).to(DEV))
    with torch.inference_mode():
        d=nn_dist(q,bank).cpu().numpy().astype(np.float64)
    return qshape(d), float(mean_top1p(d))

def auc(y,s):
    return float(roc_auc_score(y,np.asarray(s))*100.0)

def safe_median(x):
    x=np.asarray(x,float)
    return float(np.median(x))

def run_cell(obj,shot,split,formal_row,per_image_rows,bank_rows,shuffle_reps=200):
    t0=time.perf_counter()
    _,layers,tr,te=load_obj("r0",obj)
    final=tr["final"]; tf=te["final"]
    support=[int(v) for v in draw_images(final["offsets"],shot,split,obj)]
    types=np.asarray([str(v) for v in tf["types"]])
    y=(types=="bad").astype(int)
    good=(y==0)

    # FULL-k bank: exact formal TSN domain for test.
    B=make_bank(final["feats"],final["offsets"],support)
    q_full=[]; top1_full=[]
    for j in range(len(y)):
        q,t=score_q(tf,j,B)
        q_full.append(q); top1_full.append(t)
    q_full=np.asarray(q_full); top1_full=np.asarray(top1_full)
    del B

    # LOO reference + test scored under the SAME LOO bank.
    refs=[]
    q_loo=[]
    good_centers_loo=[]
    gaps_matched=[]
    for ii,sid in enumerate(support):
        ids=[u for u in support if u!=sid]
        Bi=make_bank(final["feats"],final["offsets"],ids)
        qref_i,_=score_q(final,sid,Bi)
        refs.append(qref_i)

        qt=[]
        for j in range(len(y)):
            q,_=score_q(tf,j,Bi)
            qt.append(q)
        qt=np.asarray(qt)
        q_loo.append(qt)

        gc=safe_median(qt[good])
        good_centers_loo.append(gc)
        gaps_matched.append(qref_i-gc)
        bank_rows.append(dict(
            object=obj,shot=shot,split=split,bank_i=ii,
            support_id=sid,n_bank=len(ids),
            q_ref_i=qref_i,test_good_center_i=gc,
            matched_calibration_gap=qref_i-gc,
            full_good_center=safe_median(q_full[good]),
            full_minus_loo_good_center=safe_median(q_full[good])-gc
        ))
        del Bi

    refs=np.asarray(refs)
    q_loo=np.stack(q_loo,axis=0) # [k, ntest]
    good_centers_loo=np.asarray(good_centers_loo)

    qref_naive=safe_median(refs)
    good_full=safe_median(q_full[good])

    # Scores / interventions
    score_naive=np.abs(q_full-qref_naive)
    score_oracle_full=np.abs(q_full-good_full)

    # Exact bank identity matching.
    score_matched=np.median(np.abs(q_loo-refs[:,None]),axis=0)

    # Same k-1 bank size, but replace bank-specific ref with common ref.
    common_ref=safe_median(refs)
    score_size_only=np.median(np.abs(q_loo-common_ref),axis=0)

    # Oracle per-bank centers: upper bound for train-vs-test normal calibration.
    score_matched_oracle=np.median(np.abs(q_loo-good_centers_loo[:,None]),axis=0)

    # Cyclic wrong pairing + repeated random wrong pairing.
    rolled=np.roll(refs,1)
    score_wrong_cyclic=np.median(np.abs(q_loo-rolled[:,None]),axis=0)
    rng=np.random.default_rng(20260930 + shot*100 + split)
    shuf_aucs=[]
    if shot>1:
        for _ in range(shuffle_reps):
            perm=rng.permutation(shot)
            if np.all(perm==np.arange(shot)):
                perm=np.roll(perm,1)
            s=np.median(np.abs(q_loo-refs[perm,None]),axis=0)
            shuf_aucs.append(auc(y,s))

    # Direction diagnostics
    res=dict(
        object=obj,shot=shot,split=split,n_test=len(y),n_bad=int(y.sum()),n_good=int(good.sum()),
        formal_baseline_final=float(formal_row.baseline_final_img_AUROC),
        formal_baseline_agg=float(formal_row.baseline_agg_img_AUROC),
        formal_tsn=float(formal_row.tsn_img_AUROC),
        q_ref_formal=float(formal_row.q_ref),
        q_ref_recomputed=qref_naive,
        q_ref_abs_diff=abs(qref_naive-float(formal_row.q_ref)),
        test_good_center_full=good_full,
        calibration_gap_full=qref_naive-good_full,
        matched_gap_abs_mean=float(np.mean(np.abs(gaps_matched))),
        matched_gap_abs_median=float(np.median(np.abs(gaps_matched))),
        auc_final_raw_replay=auc(y,top1_full),
        auc_naive_replay=auc(y,score_naive),
        auc_q=auc(y,q_full),
        auc_minus_q=auc(y,-q_full),
        auc_abs_oracle_full=auc(y,score_oracle_full),
        auc_matched_support=auc(y,score_matched),
        auc_size_only_common_ref=auc(y,score_size_only),
        auc_wrong_cyclic=auc(y,score_wrong_cyclic),
        auc_shuffle_mean=float(np.mean(shuf_aucs)) if shuf_aucs else np.nan,
        auc_shuffle_q05=float(np.quantile(shuf_aucs,.05)) if shuf_aucs else np.nan,
        auc_shuffle_q95=float(np.quantile(shuf_aucs,.95)) if shuf_aucs else np.nan,
        auc_matched_oracle=auc(y,score_matched_oracle),
        gain_oracle_full_vs_naive=auc(y,score_oracle_full)-auc(y,score_naive),
        gain_matched_vs_naive=auc(y,score_matched)-auc(y,score_naive),
        gain_matched_vs_shuffle=auc(y,score_matched)-(float(np.mean(shuf_aucs)) if shuf_aucs else np.nan),
        gain_matched_oracle_vs_matched=auc(y,score_matched_oracle)-auc(y,score_matched),
        bank_center_shift_abs_mean=float(np.mean(np.abs(
            safe_median(q_full[good])-good_centers_loo))),
        elapsed_sec=time.perf_counter()-t0
    )

    names=[str(v) for v in tf.get("names",np.arange(len(y)))]
    for j in range(len(y)):
        per_image_rows.append(dict(
            object=obj,shot=shot,split=split,image=names[j],
            type=types[j],y=int(y[j]),
            q_full=float(q_full[j]),
            score_naive=float(score_naive[j]),
            score_oracle_full=float(score_oracle_full[j]),
            score_matched=float(score_matched[j]),
            score_size_only=float(score_size_only[j]),
            score_wrong_cyclic=float(score_wrong_cyclic[j]),
            score_matched_oracle=float(score_matched_oracle[j])
        ))
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--formal",default=r"results\model_v0\metrics\tsn_formal_v2.csv")
    ap.add_argument("--selection",default=r"tsn_failure_audit\04_selected_causal_cells.csv")
    ap.add_argument("--outdir",default=r"results\model_v0\metrics\tsn_failure_audit")
    ap.add_argument("--shuffle-reps",type=int,default=200)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    print("GPU:",torch.cuda.get_device_name(0))

    formal=pd.read_csv(args.formal)
    sel=pd.read_csv(args.selection)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    pcell=out/"causal_audit_cells.csv"
    pimg=out/"causal_audit_per_image.csv"
    pbank=out/"causal_audit_bank_level.csv"

    cells=[]; images=[]; banks=[]; done=set()
    if args.resume and pcell.exists():
        old=pd.read_csv(pcell)
        cells=old.to_dict("records")
        done={(str(r.object),int(r.shot),int(r["split"])) for _,r in old.iterrows()}
        if pimg.exists(): images=pd.read_csv(pimg).to_dict("records")
        if pbank.exists(): banks=pd.read_csv(pbank).to_dict("records")

    for n,r in sel.iterrows():
        key=(str(r.object),int(r.shot),int(r["split"]))
        if key in done:
            print("SKIP",key); continue
        fr=formal[(formal.object==key[0])&(formal.shot==key[1])&(formal["split"]==key[2])]
        if len(fr)!=1: raise RuntimeError(f"formal cell not unique: {key}")
        print(f"[{n+1}/{len(sel)}] {key} reason={r.reason}",flush=True)
        rr=run_cell(key[0],key[1],key[2],fr.iloc[0],images,banks,args.shuffle_reps)
        rr["reason"]=r.reason
        cells.append(rr)
        pd.DataFrame(cells).to_csv(pcell,index=False)
        pd.DataFrame(images).to_csv(pimg,index=False)
        pd.DataFrame(banks).to_csv(pbank,index=False)
        print("  naive %.2f | oracle %.2f | matched %.2f | matched-oracle %.2f | q %.2f | -q %.2f" %
              (rr["auc_naive_replay"],rr["auc_abs_oracle_full"],rr["auc_matched_support"],
               rr["auc_matched_oracle"],rr["auc_q"],rr["auc_minus_q"]),flush=True)

    print("\nDONE:",pcell)

if __name__=="__main__":
    main()
