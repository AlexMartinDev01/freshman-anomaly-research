# -*- coding: utf-8 -*-
from pathlib import Path
import argparse,pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
ap=argparse.ArgumentParser()
ap.add_argument("--indir",default=r"results\model_v0\metrics\r6a1c_corrected")
a=ap.parse_args(); d=Path(a.indir)

c=pd.read_csv(d/"support_coverage_corrected.csv")
l=pd.read_csv(d/"layer_consistency_corrected.csv")
b=pd.read_csv(d/"detectability_bad_groups_corrected.csv")
g=pd.read_csv(d/"good_tail_control_corrected.csv")

print("=== COVERAGE: object/group mean true kth-NN distance ===")
print(c.groupby(["object","group"])[["d1","d2","d5","d10"]].mean().round(4).to_string())

print("\n=== LAYER: true test-query 1NN ===")
print(l.groupby(["object","layer","group"]).d1_mean.mean().unstack("layer").round(4).to_string())

print("\n=== TRUE-1NN NORMAL-ONLY DETECTABILITY: N_H vs GOOD TAIL ===")
rows=[]
for obj in sorted(set(b.object)&set(g.object)):
    x=b[(b.object==obj)&(b.group=="N_H")]
    y=g[g.object==obj]
    a1=x.frac_over_p99.mean(); a0=y.frac_over_p99.mean()
    yy=np.r_[np.ones(len(x)),np.zeros(len(y))]
    ss=np.r_[x.frac_over_p99.values,y.frac_over_p99.values]
    auc=roc_auc_score(yy,ss)
    rows.append((obj,a1,a0,a1-a0,auc))
r=pd.DataFrame(rows,columns=["object","NH_over_p99","goodtail_over_p99","diff","AUC_NH_vs_goodtail"])
print(r.round(4).to_string(index=False))
r.to_csv(d/"corrected_object_verdict_table.csv",index=False)

print("\nIMPORTANT: AUC~0.5 here only rejects discrimination by this true-1NN tail statistic;")
print("it does NOT prove impossibility of all normal-only statistics.")
