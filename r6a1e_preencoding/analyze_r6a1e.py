# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, pandas as pd, numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--indir",default=r"results\model_v0\metrics\r6a1e_preencoding")
a=ap.parse_args()
d=Path(a.indir)

s=pd.read_csv(d/"object_donor_summary.csv")
rawfull=pd.read_csv(d/"raw_full_good_baseline.csv")
rings=pd.read_csv(d/"ring_profile.csv")

# donor-mean table
dm=s.groupby(["object","radius","alpha"]).agg(
    raw_pwr=("raw_matched_pwr","mean"),
    cf_pwr=("cf_matched_pwr","mean"),
    pwr_change=("pwr_change","mean"),
    drift_diff=("mean_drift_diff","mean"),
    max_drift_p=("drift_diff_wilcoxon_p","max"),
    score_DiD=("mean_score_DiD","mean"),
    donor_sign_agree=("pwr_change",lambda x: bool((x<0).all()) if len(x) else False),
    n_donors=("donor_rank","nunique")
).reset_index()
dm.to_csv(d/"donor_mean_summary.csv",index=False)

print("=== PRIMARY alpha=1%, radii 8/16 ===")
x=dm[(dm.alpha==.01)&(dm.radius.isin([8,16]))].copy()
print(x[["object","radius","raw_pwr","cf_pwr","pwr_change","drift_diff",
         "max_drift_p","score_DiD","donor_sign_agree"]].round(4).to_string(index=False))

# Frozen decision
dec=[]
for obj,g in dm[dm.alpha==.01].groupby("object"):
    gg=g[g.radius.isin([8,16])].sort_values("radius")
    # propagation supported if either headline radius passes
    prop=False
    for _,r in gg.iterrows():
        if (r.raw_pwr>=.60 and r.pwr_change<=-.05 and
            bool(r.donor_sign_agree) and r.drift_diff>0 and r.max_drift_p<.05):
            prop=True
    persistent=False
    if set(gg.radius)=={8,16}:
        persistent=bool(
            (gg.raw_pwr>=.60).all() and
            (gg.cf_pwr>=.60).all() and
            (gg.pwr_change.abs()<.03).all()
        )
    verdict="PROPAGATION_SUPPORTED" if prop else ("FAR_FIELD_PERSISTENT" if persistent else "MIXED_INDETERMINATE")
    dec.append(dict(object=obj,verdict=verdict))
v=pd.DataFrame(dec)
v.to_csv(d/"FROZEN_OBJECT_VERDICTS.csv",index=False)
print("\n=== FROZEN VERDICTS ===")
print(v.to_string(index=False))

print("\n=== RAW FULL-GOOD baseline (alpha=1%) ===")
z=rawfull[rawfull.alpha==.01].pivot(index="object",columns="radius",values="raw_full_good_pairwise_auc")
print(z.round(2).to_string())

# ring mechanism summary: bad vs good mean
rr=rings.groupby(["object","radius","donor_rank","ring","target_label"])[["feature_drift","score_shift"]].mean().reset_index()
rr["is_bad"]=rr.target_label.eq("bad")
bad=rr[rr.is_bad].groupby(["object","radius","donor_rank","ring"])[["feature_drift","score_shift"]].mean()
good=rr[~rr.is_bad].groupby(["object","radius","donor_rank","ring"])[["feature_drift","score_shift"]].mean()
q=bad.join(good,lsuffix="_bad",rsuffix="_good").reset_index()
q["drift_diff"]=q.feature_drift_bad-q.feature_drift_good
q["score_shift_DiD"]=q.score_shift_bad-q.score_shift_good
q.to_csv(d/"ring_bad_minus_good.csv",index=False)

print("\nInterpretation boundary:")
print("- PWR drop + excess bad far-field drift supports propagation from the replaced region.")
print("- Persistence after pre-encoding replacement does not identify its source; it triggers global/acquisition audit.")
print("- Do not convert this 6-object audit into a final detector.")
