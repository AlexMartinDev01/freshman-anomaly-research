# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon

ap=argparse.ArgumentParser()
ap.add_argument("--indir",default=r"results\model_v0\metrics\r5_rawmap")
ap.add_argument("--selection",default=r"r5_rawmap_causal_interventions\R5_MAIN_SELECTION.csv")
a=ap.parse_args()
d=Path(a.indir)

cells=pd.read_csv(d/"R5_cells.csv")
sa=pd.read_csv(d/"R5A_strength_per_image.csv")
sauc=pd.read_csv(d/"R5A_strength_auc.csv")
ex=pd.read_csv(d/"R5B_extent_normal_background.csv")
co=pd.read_csv(d/"R5C_coherence_original.csv") if (d/"R5C_coherence_original.csv").exists() else pd.DataFrame()
sh=pd.read_csv(d/"R5C_coherence_shuffle.csv") if (d/"R5C_coherence_shuffle.csv").exists() else pd.DataFrame()
sel=pd.read_csv(a.selection)
dsmap=sel[["object","dataset"]].drop_duplicates()
out=[]

# A: dose=-1 rows carry per-image Spearman rhos in top1/max/q columns.
ar=sa[sa.dose==-1].copy()
aobj=ar.groupby("object").agg(
    n_bad=("image","count"),
    rho_top1=("top1","mean"),
    rho_max=("max","mean"),
    rho_q=("q","mean"),
).reset_index().merge(dsmap,on="object",how="left")
aobj.to_csv(d/"R5A_object_summary.csv",index=False)
n=len(aobj)
nq=int((aobj.rho_q<0).sum())
nt=int((aobj.rho_top1>=0).sum())
pa=binomtest(nq,n,.5,alternative="greater").pvalue
pt=binomtest(nt,n,.5,alternative="greater").pvalue
diff=aobj.rho_top1-aobj.rho_q
pw=wilcoxon(diff,alternative="greater").pvalue if np.any(diff!=0) else 1.0

# dataset replication
ads=aobj.groupby("dataset").agg(n=("object","count"),mean_rho_top1=("rho_top1","mean"),
                               mean_rho_q=("rho_q","mean"),
                               frac_q_negative=("rho_q",lambda x:(x<0).mean())).reset_index()
ads.to_csv(d/"R5A_dataset_summary.csv",index=False)

# B
br=ex[ex.alpha==-1].copy()
bobj=br.groupby("object").agg(
    n_good=("image","count"),
    rho_top1=("top1","mean"),
    rho_max=("max","mean"),
    rho_q=("q","mean"),
    frac_image_q_negative=("q",lambda x:(x<0).mean()),
    frac_image_top1_nonnegative=("top1",lambda x:(x>=0).mean())
).reset_index().merge(dsmap,on="object",how="left")
bobj.to_csv(d/"R5B_object_summary.csv",index=False)
nb=len(bobj); nbq=int((bobj.rho_q<0).sum())
pb=binomtest(nbq,nb,.5,alternative="greater").pvalue
all_image_qneg=float((br.q<0).mean())
all_image_topok=float((br.top1>=0).mean())

# C
if len(co):
    cx=co.merge(sh,on=["object","shot","split","alpha"],suffixes=("","_sh"))
    cx["lcc_over_q95"]=cx.auc_lcc>cx.lcc_q95
    cx["adj_over_q95"]=cx.auc_adjacency>cx.adjacency_q95
    cx["lcc_effect"]=cx.auc_lcc-cx.lcc_mean
    cx["adj_effect"]=cx.auc_adjacency-cx.adjacency_mean
    cx.to_csv(d/"R5C_vs_shuffle.csv",index=False)
    # Best fixed alpha is NOT selected post-hoc. Report each alpha separately.
    csum=cx.groupby("alpha").agg(
        n_objects=("object","nunique"),
        lcc_objects_over_q95=("lcc_over_q95","sum"),
        adj_objects_over_q95=("adj_over_q95","sum"),
        mean_lcc_effect=("lcc_effect","mean"),
        mean_adj_effect=("adj_effect","mean")
    ).reset_index()
    csum.to_csv(d/"R5C_alpha_summary.csv",index=False)
else:
    csum=pd.DataFrame()

report=f"""# R5 MAIN AUTOMATIC REPORT

## R5-A Strength-only
Objects: {n}
Objects with mean rho_q < 0: {nq}/{n}; exact sign p={pa:.6g}
Objects with mean rho_top1 >= 0: {nt}/{n}; exact sign p={pt:.6g}
Paired object-level (rho_top1-rho_q) one-sided Wilcoxon p={pw:.6g}

Dataset summary:
{ads.to_string(index=False)}

Frozen A verdict:
A1 positive-control top1 >=0 in >=70% objects: {nt/n>=.70}
A2 q negative in >=70% objects and p<.05: {(nq/n>=.70) and (pa<.05)}
A3 top1-q paired Wilcoxon p<.05: {pw<.05}
A4 both datasets mean rho_q<0: {bool((ads.mean_rho_q<0).all())}

## R5-B Extent-only on real normal backgrounds
Objects with mean rho_q <0: {nbq}/{nb}; sign p={pb:.6g}
All good-image fraction rho_q<0: {all_image_qneg:.4f}
All good-image fraction rho_top1>=0: {all_image_topok:.4f}

Frozen B verdict:
B1 top1 nonnegative in 100% images: {all_image_topok>=.999999}
B2 >=80% images q-negative + every object median/mean q-negative + object sign p<.05:
{(all_image_qneg>=.80) and (nbq==nb) and (pb<.05)}

## R5-C Spatial signal
No post-hoc best-alpha selection is used.
{csum.to_string(index=False) if len(csum) else "No split-0 coherence rows found."}

Interpretation is NOT automated beyond the frozen gates.
Object is the independent block.
Do not run LOCKED replication until this MAIN report is reviewed and frozen.
"""
(d/"R5_MAIN_REPORT.md").write_text(report,encoding="utf-8")
print(report)
