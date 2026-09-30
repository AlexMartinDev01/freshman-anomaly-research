# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys,json,math
import numpy as np,pandas as pd,torch
from PIL import Image
from scipy.stats import wilcoxon

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

# Mandatory precheck artifacts
for req in ["parity_report.csv","resolved_test_paths.csv","donor_manifest.csv","pair_manifest.csv"]:
    if not (OUT/req).exists():
        raise RuntimeError(f"Missing precheck artifact {OUT/req}; run r6a1e_precheck.py first.")

par=pd.read_csv(OUT/"parity_report.csv")
if (par.median_cos<0.9995).any() or (par.p01_cos<0.995).any():
    raise RuntimeError("Saved parity report does not pass frozen gate. STOP.")

PATHS=pd.read_csv(OUT/"resolved_test_paths.csv")
DONORS=pd.read_csv(OUT/"donor_manifest.csv")
PAIRS=pd.read_csv(OUT/"pair_manifest.csv")

RADII=[0,2,4,8,16]
ALPHAS=[.001,.0025,.005,.01,.02,.05]
PRIMARY_ALPHA=.01
MIN_VALID_R16=10
RINGS=["1-2","3-4","5-8","9-16","17+"]

model=load_anomalydino_model(ROOT,CFG)
patch_size=int(model.model.patch_size)
batch_size=int(CFG["batch_size"])

pair_rows=[]
ring_rows=[]
survivor_rows=[]
raw_full_rows=[]

example_dir=OUT/"examples"
example_dir.mkdir(exist_ok=True)

def get_path(obj,idx):
    x=PATHS[(PATHS.object==obj)&(PATHS.image_index==int(idx))]
    if len(x)!=1: raise RuntimeError(f"path manifest miss {obj}/{idx}")
    return Path(x.iloc[0].path)

def save_example(t,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(denormalize_to_uint8(t)).save(path)

for _,rr in SEL.iterrows():
    obj=str(rr.object); shot=int(rr.shot); split=int(rr["split"])
    print("\n=== OBJECT",obj,"===",flush=True)
    _,_,tr,te=load_obj("r0",obj)
    tc,ec=tr["final"],te["final"]

    ids=[int(v) for v in draw_images(tc["offsets"],shot,split,obj)]
    bank=make_support_bank(tc,ids)

    types=np.asarray([str(x) for x in ec["types"]])
    names=np.asarray([str(x) for x in ec["names"]])
    offs=np.asarray(ec["offsets"])
    H,W=map(int,ec["grids"][0])

    bad_all=np.where(types=="bad")[0]
    good_all=np.where(types=="good")[0]

    # Cached raw features and maps; this anchors the experiment to the audited baseline.
    raw_feats={i:cache_img_features(ec,int(i)) for i in range(len(types))}
    raw_maps={i:score_1nn(raw_feats[i],bank).reshape(H,W) for i in range(len(types))}

    # Freeze survivor-matched bad set using geometry only.
    survivors=[]
    gt_masks={}
    exc_masks={}
    for bi in bad_all:
        aa,bb=int(offs[bi]),int(offs[bi+1])
        gt=np.asarray(ec["gt_frac"][aa:bb],float).reshape(H,W)
        G=gt>.10
        if not G.any(): continue
        gt_masks[int(bi)]=G
        exc_masks[int(bi)]={}
        for r in RADII:
            exc_masks[int(bi)][r]=dilate_patch_mask(G,r)
        nvalid=int((~exc_masks[int(bi)][16]).sum())
        survivor_rows.append(dict(object=obj,bad_index=int(bi),bad_name=names[bi],
                                  valid_r16=nvalid,primary_survivor=nvalid>=MIN_VALID_R16))
        if nvalid>=MIN_VALID_R16:
            survivors.append(int(bi))
    print(" primary survivors",len(survivors),"/",len(bad_all))

    # Load donor prepared tensors.
    dtab=DONORS[DONORS.object==obj].sort_values("donor_rank")
    donor_tensors=[]
    for _,dr in dtab.iterrows():
        t,g=model.prepare_image(str(dr.path))
        if tuple(map(int,g))!=(H,W): raise RuntimeError(f"{obj}: donor grid mismatch")
        donor_tensors.append((int(dr.donor_rank),t,str(dr.path)))

    # Prepared target tensor cache (CPU).
    prep_cache={}
    def prep(i):
        i=int(i)
        if i not in prep_cache:
            t,g=model.prepare_image(str(get_path(obj,i)))
            if tuple(map(int,g))!=(H,W): raise RuntimeError(f"{obj}/{i}: grid mismatch")
            prep_cache[i]=t
        return prep_cache[i]

    # External raw full-good AUC using same survivor set, useful to compare to A1D.
    for r in RADII:
        for alpha in ALPHAS:
            wins=ties=den=0
            for bi in survivors:
                valid=~exc_masks[bi][r]
                bs=topmean(raw_maps[bi],valid,alpha)
                for gi in good_all:
                    gs=topmean(raw_maps[int(gi)],valid,alpha)
                    wins += int(bs>gs); ties += int(bs==gs); den += 1
            auc=100*(wins+.5*ties)/den if den else np.nan
            raw_full_rows.append(dict(object=obj,radius=r,alpha=alpha,
                                      raw_full_good_pairwise_auc=auc,
                                      n_bad=len(survivors),n_good=len(good_all)))

    # Run matched counterfactual pairs.
    obj_pairs=PAIRS[PAIRS.object==obj]
    for bcount,bi in enumerate(survivors):
        G=gt_masks[bi]
        dist=cheb_distance_from_gt(G).reshape(-1)

        p=obj_pairs[obj_pairs.bad_index==bi].sort_values("good_rank")
        good_ids=[int(x) for x in p.good_index.tolist()]
        if not good_ids:
            raise RuntimeError(f"{obj}/{bi}: no frozen good pairs")

        target_tensors=[prep(bi)]+[prep(gi) for gi in good_ids]

        for r in RADII:
            exc=exc_masks[bi][r]
            valid=(~exc).reshape(-1)
            pixmask=patchmask_to_pixel(exc,patch_size,target_tensors[0].shape[-2:])

            # raw scores/features on exact same matched targets.
            raw_target_feats=[raw_feats[bi]]+[raw_feats[gi] for gi in good_ids]
            raw_target_maps=[raw_maps[bi]]+[raw_maps[gi] for gi in good_ids]

            for donor_rank,donor_tensor,donor_path in donor_tensors:
                cfs=[replace_tensor(t,donor_tensor,pixmask) for t in target_tensors]
                zlist=extract_batch(model,cfs,batch_size=batch_size)
                cf_maps=[score_1nn(z,bank).reshape(H,W) for z in zlist]

                # Save a small visual audit set.
                if CFG.get("save_examples",True) and bcount==0 and donor_rank==0 and r in [0,8,16]:
                    save_example(target_tensors[0],example_dir/f"{obj}_raw_bad_r{r}.png")
                    save_example(cfs[0],example_dir/f"{obj}_cf_bad_r{r}.png")

                # feature drift arrays for bad and goods
                drift_means=[]; drift_arrays=[]
                for rf,cf in zip(raw_target_feats,zlist):
                    dm,da=feature_drift(rf,cf,valid)
                    drift_means.append(dm); drift_arrays.append(da)

                for alpha in ALPHAS:
                    raw_scores=[topmean(m.reshape(-1),valid,alpha) for m in raw_target_maps]
                    cf_scores=[topmean(m.reshape(-1),valid,alpha) for m in cf_maps]

                    for j,gi in enumerate(good_ids, start=1):
                        rb,rg=raw_scores[0],raw_scores[j]
                        cb,cg=cf_scores[0],cf_scores[j]
                        raw_win=float(rb>rg)+.5*float(rb==rg)
                        cf_win=float(cb>cg)+.5*float(cb==cg)
                        pair_rows.append(dict(
                            object=obj,bad_index=bi,bad_name=names[bi],
                            good_index=gi,good_name=names[gi],
                            radius=r,alpha=alpha,donor_rank=donor_rank,
                            donor_path=donor_path,n_valid=int(valid.sum()),
                            raw_bad=rb,raw_good=rg,cf_bad=cb,cf_good=cg,
                            raw_win=raw_win,cf_win=cf_win,
                            pwr_delta=cf_win-raw_win,
                            bad_score_shift=cb-rb,
                            good_score_shift=cg-rg,
                            score_DiD=(cb-rb)-(cg-rg),
                            bad_feature_drift=drift_means[0],
                            good_feature_drift=drift_means[j],
                            drift_diff=drift_means[0]-drift_means[j]
                        ))

                # Distance-stratified mechanism profile.
                for j,label in enumerate(["bad"]+[f"good{k}" for k in range(len(good_ids))]):
                    raw_score_flat=raw_target_maps[j].reshape(-1)
                    cf_score_flat=cf_maps[j].reshape(-1)
                    shift=cf_score_flat-raw_score_flat
                    drift=drift_arrays[j]
                    for ring in RINGS:
                        idx=np.array([ring_label(int(d))==ring for d in dist]) & valid
                        if not idx.any(): continue
                        ring_rows.append(dict(
                            object=obj,bad_index=bi,target_label=label,
                            radius=r,donor_rank=donor_rank,ring=ring,
                            n_patch=int(idx.sum()),
                            feature_drift=float(np.mean(drift[idx])),
                            score_shift=float(np.mean(shift[idx]))
                        ))

        if (bcount+1)%10==0:
            print(" ",obj,bcount+1,"/",len(survivors),flush=True)
            pd.DataFrame(pair_rows).to_csv(OUT/"per_pair.csv",index=False)
            pd.DataFrame(ring_rows).to_csv(OUT/"ring_profile.csv",index=False)

    pd.DataFrame(pair_rows).to_csv(OUT/"per_pair.csv",index=False)
    pd.DataFrame(ring_rows).to_csv(OUT/"ring_profile.csv",index=False)
    pd.DataFrame(survivor_rows).to_csv(OUT/"survivor_manifest.csv",index=False)
    pd.DataFrame(raw_full_rows).to_csv(OUT/"raw_full_good_baseline.csv",index=False)

# Aggregate object summary
d=pd.DataFrame(pair_rows)
summ=[]
perbad=[]
for (obj,r,a,donor),g in d.groupby(["object","radius","alpha","donor_rank"]):
    # per bad first, avoiding overweighting any duplicated matched pairs
    pb=g.groupby("bad_index").agg(
        raw_pwr=("raw_win","mean"),
        cf_pwr=("cf_win","mean"),
        drift_diff=("drift_diff","mean"),
        score_DiD=("score_DiD","mean"),
        bad_drift=("bad_feature_drift","mean"),
        good_drift=("good_feature_drift","mean")
    ).reset_index()
    pb["object"]=obj; pb["radius"]=r; pb["alpha"]=a; pb["donor_rank"]=donor
    perbad.append(pb)
    vals=pb.drift_diff.to_numpy()
    try:
        wp=float(wilcoxon(vals,alternative="two-sided",zero_method="wilcox").pvalue) if np.any(vals!=0) else 1.0
    except Exception:
        wp=np.nan
    summ.append(dict(
        object=obj,radius=r,alpha=a,donor_rank=donor,
        n_bad=len(pb),
        raw_matched_pwr=float(pb.raw_pwr.mean()),
        cf_matched_pwr=float(pb.cf_pwr.mean()),
        pwr_change=float(pb.cf_pwr.mean()-pb.raw_pwr.mean()),
        mean_bad_drift=float(pb.bad_drift.mean()),
        mean_good_drift=float(pb.good_drift.mean()),
        mean_drift_diff=float(pb.drift_diff.mean()),
        drift_diff_wilcoxon_p=wp,
        mean_score_DiD=float(pb.score_DiD.mean())
    ))
pd.concat(perbad,ignore_index=True).to_csv(OUT/"per_bad_summary.csv",index=False)
pd.DataFrame(summ).to_csv(OUT/"object_donor_summary.csv",index=False)

print("\nR6-A1E main run complete:",OUT)
