# -*- coding: utf-8 -*-
"""
RC root-cause Stage 1.
Implements only the frozen gates in rootcause_rc/RC_PREREG_FROZEN.md:
RC-GEO1, revised RC-SUFF1 (Amendment A1), RC-COV1 supporting analysis.
R6-G outputs are deliberately not read.
"""
from pathlib import Path
import hashlib, json, subprocess, zlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rootcause_rc" / "results_stage1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260930
B_BOOT = 20000
B_PERM = 20000

P = {
 "coverage": ROOT/"results/model_v0/metrics/r6a1c_corrected/support_coverage_corrected.csv",
 "layers": ROOT/"results/model_v0/metrics/r6a1c_corrected/layer_consistency_corrected.csv",
 "diversity": ROOT/"results/model_v0/metrics/r6a1c_corrected/neighbour_diversity_corrected.csv",
 "smoke": ROOT/"results/model_v0/metrics/r6a_smoke/summary.csv",
 "scores": ROOT/"results/model_v0/metrics/r6a_smoke/image_scores.csv",
 "folds": ROOT/"results/model_v0/metrics/r6a_smoke/folds.csv",
 "gate17": ROOT/"results/model_v0/metrics/gate17a_fusion.csv",
 "prereg": ROOT/"rootcause_rc/RC_PREREG_FROZEN.md",
}
for k,p in P.items():
    if not p.exists():
        raise SystemExit(f"Missing required input {k}: {p}")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def stable_seed(*parts):
    s="|".join(map(str,parts))
    return (SEED + zlib.crc32(s.encode("utf-8"))) % (2**32-1)

def wtest(x, alternative="greater"):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if len(x)==0 or np.all(np.abs(x)<1e-15):
        return 1.0
    try:
        return float(wilcoxon(x, alternative=alternative, zero_method="wilcox", method="auto").pvalue)
    except ValueError:
        return 1.0

def bootstrap_ci_median(x, seed, B=B_BOOT):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    vals=[]
    step=1000
    for s in range(0,B,step):
        n=min(step,B-s)
        idx=rng.integers(0,len(x),size=(n,len(x)))
        vals.append(np.median(x[idx],axis=1))
    v=np.concatenate(vals)
    return tuple(np.quantile(v,[.025,.975]))

# RC-GEO1
cov=pd.read_csv(P["coverage"])
req={"object","image","group","d1","d2","d5","d10"}
if not req.issubset(cov.columns): raise SystemExit(f"coverage schema mismatch: {cov.columns.tolist()}")
if cov.duplicated(["object","image","group"]).any():
    raise SystemExit("GEO1 coverage has duplicate object/image/group rows")

geo_rows=[]
for obj,g in cov.groupby("object"):
    pv=g[g.group.isin(["D","N_H"])].set_index(["image","group"])
    for metric in ["d1","d2","d5","d10"]:
        wide=pv[metric].unstack("group").dropna()
        if not {"D","N_H"}.issubset(wide.columns): continue
        d=(wide["N_H"]-wide["D"]).to_numpy(float)
        lo,hi=bootstrap_ci_median(d,stable_seed("geo",obj,metric))
        geo_rows.append(dict(object=obj,axis="knn",metric=metric,n_images=len(d),
            median_delta=float(np.median(d)),mean_delta=float(np.mean(d)),
            sign_fraction=float(np.mean(d>0)),p_greater=wtest(d,"greater"),
            ci95_lo=lo,ci95_hi=hi))
geo=pd.DataFrame(geo_rows)

lay=pd.read_csv(P["layers"])
req={"object","layer","image","group","d1_mean"}
if not req.issubset(lay.columns): raise SystemExit(f"layer schema mismatch: {lay.columns.tolist()}")
if lay.duplicated(["object","layer","image","group"]).any():
    raise SystemExit("GEO1 layer file has duplicate object/layer/image/group rows")
layer_rows=[]
for (obj,l),g in lay.groupby(["object","layer"]):
    wide=g[g.group.isin(["D","N_H"])].pivot(index="image",columns="group",values="d1_mean").dropna()
    if not {"D","N_H"}.issubset(wide.columns): continue
    d=(wide["N_H"]-wide["D"]).to_numpy(float)
    lo,hi=bootstrap_ci_median(d,stable_seed("layer",obj,l))
    layer_rows.append(dict(object=obj,axis="layer",metric=l,n_images=len(d),
        median_delta=float(np.median(d)),mean_delta=float(np.mean(d)),
        sign_fraction=float(np.mean(d>0)),p_greater=wtest(d,"greater"),
        ci95_lo=lo,ci95_hi=hi))
geo_layer=pd.DataFrame(layer_rows)

div=pd.read_csv(P["diversity"])
req={"object","image","group","distinct_support_images","frac_from_one_image"}
if not req.issubset(div.columns): raise SystemExit(f"diversity schema mismatch: {div.columns.tolist()}")
div_nh=div[div.group=="N_H"].groupby("object").agg(
    n_images=("image","nunique"),
    median_distinct_support_images=("distinct_support_images","median"),
    median_frac_from_one_image=("frac_from_one_image","median"),
).reset_index()

obj="macaroni2"
gd=geo[geo.object==obj].set_index("metric")
gl=geo_layer[geo_layer.object==obj].set_index("metric")
dv=div_nh[div_nh.object==obj]
if len(dv)!=1: raise SystemExit("macaroni2 diversity row missing")
A=(gd.loc["d1","median_delta"]>0 and gd.loc["d1","p_greater"]<.01 and gd.loc["d1","ci95_lo"]>0)
B=sum(bool(gd.loc[k,"median_delta"]>0 and gd.loc[k,"p_greater"]<.05) for k in ["d1","d2","d5","d10"])>=3
Cfinal=("final" in gl.index and gl.loc["final","median_delta"]>0 and gl.loc["final","p_greater"]<.05)
Cother=sum(bool(l in gl.index and gl.loc[l,"median_delta"]>0 and gl.loc[l,"p_greater"]<.05) for l in ["mid","midlate"])>=1
C=Cfinal and Cother
D=(float(dv.iloc[0].median_distinct_support_images)>=2.0 and float(dv.iloc[0].median_frac_from_one_image)<=.80)
if A and B and C and D: geo_verdict="NORMALITY_GEOMETRY_MISALIGNMENT_SUPPORTED"
elif A and B and C and not D: geo_verdict="SUPPORT_IDENTITY_CONFOUNDED"
else: geo_verdict="NOT_ESTABLISHED"

smoke=pd.read_csv(P["smoke"])
smoke_diag=smoke[smoke.probe=="logreg"].copy()
smoke_diag["ap_gap"]=smoke_diag.patch_ap_probe-smoke_diag.patch_ap_1nn
smoke_diag["image_auc_gap"]=smoke_diag.image_auc_probe-smoke_diag.image_auc_1nn
smoke_diag=smoke_diag[["object","patch_ap_probe","patch_ap_1nn","ap_gap",
                       "image_auc_probe","image_auc_1nn","image_auc_gap"]]

geo.to_csv(OUT/"RC_GEO1_knn.csv",index=False)
geo_layer.to_csv(OUT/"RC_GEO1_layers.csv",index=False)
div_nh.to_csv(OUT/"RC_GEO1_diversity.csv",index=False)
smoke_diag.to_csv(OUT/"RC_GEO1_smoke_support.csv",index=False)

# RC-SUFF1 revised matched pairs
scores=pd.read_csv(P["scores"])
req={"object","fold","probe","image","image_y","score_probe","score_1nn"}
if not req.issubset(scores.columns): raise SystemExit(f"scores schema mismatch: {scores.columns.tolist()}")
if scores.duplicated(["object","probe","image"]).any():
    raise SystemExit("SUFF1 duplicate object/probe/image rows")

pair_rows=[]
suff_rows=[]
for (obj,probe),g0 in scores.groupby(["object","probe"]):
    g0=g0.dropna(subset=["score_probe","score_1nn","image_y"]).copy()
    pooled_sd=float(g0.score_1nn.std(ddof=1))
    if not np.isfinite(pooled_sd) or pooled_sd<=0: continue
    caliper=.25*pooled_sd
    local_pairs=[]
    for fold,g in g0.groupby("fold"):
        bad=g[g.image_y==1].reset_index(drop=True)
        good=g[g.image_y==0].reset_index(drop=True)
        if len(bad)==0 or len(good)==0: continue
        cost=np.abs(bad.score_1nn.to_numpy()[:,None]-good.score_1nn.to_numpy()[None,:])
        rr,cc=linear_sum_assignment(cost)
        for r,c in zip(rr,cc):
            diff=float(cost[r,c])
            if diff>caliper: continue
            b=bad.iloc[r]; q=good.iloc[c]
            local_pairs.append(dict(object=obj,probe=probe,fold=int(fold),
                bad_image=b.image,good_image=q.image,
                bad_1nn=float(b.score_1nn),good_1nn=float(q.score_1nn),
                delta_1nn=float(b.score_1nn-q.score_1nn),abs_delta_1nn=diff,
                bad_probe=float(b.score_probe),good_probe=float(q.score_probe),
                delta_probe=float(b.score_probe-q.score_probe)))
    pp=pd.DataFrame(local_pairs)
    if len(pp)==0:
        suff_rows.append(dict(object=obj,probe=probe,n_pairs=0,caliper=caliper,
            one_nn_smd=np.nan,median_abs_1nn_mismatch=np.nan,median_delta_probe=np.nan,
            mean_delta_probe=np.nan,p_wilcoxon=np.nan,p_pairswap=np.nan,
            ci95_lo=np.nan,ci95_hi=np.nan,pass_gate=False))
        continue
    pair_rows.extend(local_pairs)
    xb=pp.bad_1nn.to_numpy(float); xg=pp.good_1nn.to_numpy(float)
    sp=np.sqrt((np.var(xb,ddof=1)+np.var(xg,ddof=1))/2) if len(pp)>1 else np.nan
    smd=float((np.mean(xb)-np.mean(xg))/sp) if np.isfinite(sp) and sp>0 else np.inf
    delta=pp.delta_probe.to_numpy(float)
    med=float(np.median(delta))
    pw=wtest(delta,"greater")
    lo,hi=bootstrap_ci_median(delta,stable_seed("suffboot",obj,probe))
    rng=np.random.default_rng(stable_seed("suffperm",obj,probe))
    ge=0; step=1000
    for s in range(0,B_PERM,step):
        n=min(step,B_PERM-s)
        signs=rng.choice(np.array([-1.0,1.0]),size=(n,len(delta)))
        null=np.median(signs*delta[None,:],axis=1)
        ge += int(np.sum(null>=med))
    pperm=(1+ge)/(B_PERM+1)
    passed=(len(pp)>=15 and abs(smd)<=.10 and med>0 and pw<.01 and pperm<.01 and lo>0)
    suff_rows.append(dict(object=obj,probe=probe,n_pairs=len(pp),caliper=caliper,
        one_nn_smd=smd,median_abs_1nn_mismatch=float(pp.abs_delta_1nn.median()),
        median_delta_probe=med,mean_delta_probe=float(np.mean(delta)),
        p_wilcoxon=pw,p_pairswap=pperm,ci95_lo=lo,ci95_hi=hi,pass_gate=passed))

pairs=pd.DataFrame(pair_rows)
suff=pd.DataFrame(suff_rows)
pairs.to_csv(OUT/"RC_SUFF1_pairs.csv",index=False)
suff.to_csv(OUT/"RC_SUFF1_summary.csv",index=False)

hard=["screw","macaroni2","pcb2"]
hard_pass={}
for obj in hard:
    q=suff[suff.object==obj]
    hard_pass[obj]=bool(len(q)==2 and q.pass_gate.all())
n_hard=sum(hard_pass.values())
suff_root_supported=n_hard>=2

# RC-COV1 supporting only
g17=pd.read_csv(P["gate17"])
req={"object","shot","split","alpha","img_AUROC"}
if not req.issubset(g17.columns): raise SystemExit(f"gate17 schema mismatch: {g17.columns.tolist()}")
raw=g17[np.isclose(g17.alpha.astype(float),0.0)].copy()
raw=raw[raw.shot.isin([1,4])]
pv=raw.pivot_table(index=["object","split"],columns="shot",values="img_AUROC",aggfunc="first").dropna()
if not {1,4}.issubset(pv.columns): raise SystemExit("RC-COV1 missing shot 1 or 4")
pv["delta_4_minus_1"]=pv[4]-pv[1]
cov_pairs=pv.reset_index()
cov_obj=cov_pairs.groupby("object").agg(
    n_splits=("split","nunique"),
    mean_delta_img_auc=("delta_4_minus_1","mean"),
    median_delta_img_auc=("delta_4_minus_1","median"),
    positive_splits=("delta_4_minus_1",lambda x:int((x>0).sum()))
).reset_index()
med=float(cov_obj.mean_delta_img_auc.median())
npos=int((cov_obj.mean_delta_img_auc>0).sum())
def getgain(o):
    q=cov_obj[cov_obj.object==o]
    return float(q.iloc[0].mean_delta_img_auc) if len(q) else np.nan
dominant_pattern=bool(med>=5 and npos>=12 and getgain("screw")>=5 and getgain("cable")>=5)
cov_pairs.to_csv(OUT/"RC_COV1_pairs.csv",index=False)
cov_obj.to_csv(OUT/"RC_COV1_objects.csv",index=False)

if geo_verdict=="NORMALITY_GEOMETRY_MISALIGNMENT_SUPPORTED" and suff_root_supported:
    stage="NORMALITY_GEOMETRY_READOUT_BOTTLENECK_STRONGLY_SUPPORTED"
else:
    stage="ROOT_CAUSE_NOT_YET_PINNED"

verdict=pd.DataFrame([dict(
    geo1_verdict=geo_verdict,geo1_A=bool(A),geo1_B=bool(B),geo1_C=bool(C),geo1_D=bool(D),
    suff1_hard_pass_count=n_hard,
    suff1_hard_pass_objects=";".join([k for k,v in hard_pass.items() if v]),
    suff1_root_supported=bool(suff_root_supported),
    cov1_median_object_gain=med,cov1_positive_objects=npos,
    cov1_dominant_undercoverage_pattern=dominant_pattern,stage1_verdict=stage
)])
verdict.to_csv(OUT/"RC_STAGE1_VERDICT.csv",index=False)

manifest={k:{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for k,p in P.items()}
try:
    manifest["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
except Exception:
    manifest["git_commit"]="UNKNOWN"
(OUT/"INPUT_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

lines=[
"# RC Root-Cause Stage 1 Results","",
"## Frozen verdict","",verdict.to_string(index=False),"",
"## RC-GEO1 kNN paired D vs N_H","",geo.round(6).to_string(index=False),"",
"## RC-GEO1 layer replication","",geo_layer.round(6).to_string(index=False),"",
"## RC-GEO1 neighbour provenance control","",div_nh.round(6).to_string(index=False),"",
"## RC-SUFF1 scalar-1NN matched pairs","",suff.round(6).to_string(index=False),"",
"Hard-object pass map: "+json.dumps(hard_pass,ensure_ascii=False),"",
"## RC-COV1 supporting 1-shot vs 4-shot, non-nested","",cov_obj.round(6).to_string(index=False),"",
f"Median object mean gain: {med:.4f}; positive objects: {npos}; dominant pattern: {dominant_pattern}","",
"## Interpretation guardrail","",
"Stage-1 support does not establish residual direction specifically. If Stage-1 is strongly supported, RC-DIR remains required."
]
(OUT/"RC_STAGE1_REPORT.md").write_text("\\n".join(lines),encoding="utf-8")
print(verdict.to_string(index=False))
print("\\nHard-object SUFF1:")
print(suff[suff.object.isin(hard)].to_string(index=False))
print("\\nGEO1 macaroni2:")
print(geo[geo.object=="macaroni2"].to_string(index=False))
