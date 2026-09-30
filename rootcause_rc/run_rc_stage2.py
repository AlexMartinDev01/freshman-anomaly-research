# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib,json,subprocess
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"rootcause_rc"/"results_stage2"
OUT.mkdir(parents=True,exist_ok=True)

P={
 "patch":ROOT/"results/model_v0/metrics/r6a_smoke/per_image_patch_metrics.csv",
 "summary":ROOT/"results/model_v0/metrics/r6a_smoke/summary.csv",
 "oracle":ROOT/"results/model_v0/metrics/r6a1b_selection_halo/r6a1b_oracle_ladder.csv",
 "a1d":ROOT/"results/model_v0/metrics/r6a1d_defect_excision/object_radius_alpha_summary.csv",
 "prereg":ROOT/"rootcause_rc/RC_STAGE2_PREREG_FROZEN.md"
}
for k,p in P.items():
    if not p.exists(): raise SystemExit(f"missing {k}: {p}")

patch=pd.read_csv(P["patch"])
summary=pd.read_csv(P["summary"])
oracle=pd.read_csv(P["oracle"])
a1d=pd.read_csv(P["a1d"])

# One copy of 1NN per-image metrics.
pp=patch[patch.probe=="logreg"].copy()
bad=pp[(pp.object=="macaroni2") & pp.patch_auc_1nn.notna()].copy()
if len(bad)==0: raise SystemExit("no macaroni2 bad patch rows")

med_auc=float(bad.patch_auc_1nn.median())
frac90=float((bad.patch_auc_1nn>=.90).mean())
A=bool(med_auc>=.95 and frac90>=.80)

s=summary[(summary.object=="macaroni2")&(summary.probe=="logreg")]
if len(s)!=1: raise SystemExit("macaroni2 summary mismatch")
img_auc=float(s.iloc[0].image_auc_1nn)
B=bool(img_auc<=.75)

bad["tail_k"]=np.maximum(1,np.ceil(.01*bad.n_eligible).astype(int))
bad["extent_ratio"]=bad.n_defect/bad.tail_k
med_extent=float(bad.extent_ratio.median())
frac_less=float((bad.n_defect<bad.tail_k).mean())
C=bool(med_extent<=.25 and frac_less>=.90)

o=oracle[oracle.object=="macaroni2"].set_index("layer")
if not {"L1_fixed_1pct","L2_extent_oracle"}.issubset(o.index):
    raise SystemExit("oracle ladder missing rows")
l1=float(o.loc["L1_fixed_1pct","pairwise_auc"])
l2=float(o.loc["L2_extent_oracle","pairwise_auc"])
extent_gain=l2-l1
D=bool(extent_gain>=5.0)

q=a1d[(a1d.object=="macaroni2")&(a1d.radius==0)&np.isclose(a1d.alpha,.01)]
if len(q)!=1: raise SystemExit("A1D macaroni2 r0 alpha1 mismatch")
center=float(q.iloc[0].pairwise_auc_centered)
rmean=float(q.iloc[0].random_mean)
specificity=rmean-center
E=bool(center<=60 and specificity>=5)

loc2="AGGREGATION_DILUTION_ROOT_SUPPORTED" if all([A,B,C,D,E]) else "NOT_ESTABLISHED"

# controls / object diagnostics
control=[]
for obj,g in pp.groupby("object"):
    gb=g[g.patch_auc_1nn.notna()]
    if len(gb)==0: continue
    ss=summary[(summary.object==obj)&(summary.probe=="logreg")]
    if len(ss)!=1: continue
    k=np.maximum(1,np.ceil(.01*gb.n_eligible).astype(int))
    control.append(dict(object=obj,n_bad=len(gb),
        median_within_image_patch_auc_1nn=float(gb.patch_auc_1nn.median()),
        frac_patch_auc_ge90=float((gb.patch_auc_1nn>=.90).mean()),
        median_extent_ratio=float(np.median(gb.n_defect/k)),
        frac_defect_lt_tail_k=float(np.mean(gb.n_defect<k)),
        image_auc_1nn=float(ss.iloc[0].image_auc_1nn),
        patch_ap_1nn=float(ss.iloc[0].patch_ap_1nn),
        patch_ap_probe=float(ss.iloc[0].patch_ap_probe),
        patch_ap_gap=float(ss.iloc[0].patch_ap_probe-ss.iloc[0].patch_ap_1nn),
        patch_auc_probe=float(ss.iloc[0].patch_auc_probe)))
controls=pd.DataFrame(control)

hard_other=controls[controls.object.isin(["screw","pcb2"])]
cond2=bool(((hard_other.patch_ap_gap>=.30)&(hard_other.patch_auc_probe>=.95)).any())
arch=bool(loc2=="AGGREGATION_DILUTION_ROOT_SUPPORTED" and cond2)
arch_verdict="DOWNSTREAM_READOUT_BOTTLENECK_SUPPORTED" if arch else "NOT_ESTABLISHED"

ver=pd.DataFrame([dict(
    loc2_A=A,loc2_B=B,loc2_C=C,loc2_D=D,loc2_E=E,
    macaroni2_median_patch_auc=med_auc,
    macaroni2_frac_patch_auc_ge90=frac90,
    macaroni2_image_auc_1nn=img_auc,
    macaroni2_median_extent_ratio=med_extent,
    macaroni2_frac_defect_lt_tail=frac_less,
    extent_oracle_gain=extent_gain,
    defect_excision_centered_auc=center,
    translated_null_mean=rmean,
    excision_specificity_gap=specificity,
    loc2_verdict=loc2,
    other_hard_patch_stage_condition=cond2,
    architecture_verdict=arch_verdict
)])
ver.to_csv(OUT/"RC_STAGE2_VERDICT.csv",index=False)
bad[["object","image","n_eligible","n_defect","tail_k","extent_ratio",
     "patch_auc_1nn","patch_ap_1nn"]].to_csv(OUT/"RC_LOC2_macaroni2_images.csv",index=False)
controls.to_csv(OUT/"RC_ARCH2_object_diagnostics.csv",index=False)

def sha(p):
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    return h
manifest={k:{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for k,p in P.items()}
try: manifest["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
except: manifest["git_commit"]="UNKNOWN"
(OUT/"INPUT_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

report=[
"# RC Stage2 Results","",
"Frozen verdict:",ver.to_string(index=False),"",
"Object diagnostics:",controls.round(6).to_string(index=False),"",
"Interpretation boundary:",
"LOC2 is object-specific. ARCH2 establishes a downstream readout bottleneck at two different stages, but does not yet identify the exact high-dimensional variable lost by the patch-level scalar readout."
]
(OUT/"RC_STAGE2_REPORT.md").write_text("\n".join(report),encoding="utf-8")
print(ver.to_string(index=False))
print(controls.round(6).to_string(index=False))
