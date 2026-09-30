from pathlib import Path
import json,hashlib,subprocess
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"rootcause_rc"/"results_agg3"; OUT.mkdir(parents=True,exist_ok=True)
SRC=ROOT/"results/model_v0/metrics/r6a1_tail_audit/r6a1_tail_occupancy.csv"
PRE=ROOT/"rootcause_rc/RC_AGG3_PREREG_FROZEN.md"
ALPHAS=[.001,.0025,.005,.01,.02,.05]
SEED=20260930; B=20000

d=pd.read_csv(SRC)
d=d[(d.arm=="1nn") & d.alpha.isin(ALPHAS)].copy()
need={"object","image","image_id","alpha","tail_mean","gt_precision","gt_recall"}
if not need.issubset(d.columns): raise SystemExit(f"schema mismatch: {d.columns.tolist()}")
if d.duplicated(["object","image_id","alpha"]).any(): raise SystemExit("duplicate object/image/alpha")

rows=[]
for (obj,a),g in d.groupby(["object","alpha"]):
    y=(g.image.astype(str)=="bad").astype(int).to_numpy()
    s=g.tail_mean.to_numpy(float)
    if len(np.unique(y))<2: continue
    bad=g[g.image.astype(str)=="bad"]; good=g[g.image.astype(str)=="good"]
    rows.append(dict(object=obj,alpha=float(a),n_images=len(g),n_bad=len(bad),n_good=len(good),
        image_auc=float(roc_auc_score(y,s)),image_ap=float(average_precision_score(y,s)),
        bad_median=float(bad.tail_mean.median()),good_median=float(good.tail_mean.median()),
        good_p95=float(good.tail_mean.quantile(.95)),
        bad_gt_precision_median=float(bad.gt_precision.median()),
        bad_gt_recall_median=float(bad.gt_recall.median())))
M=pd.DataFrame(rows); M.to_csv(OUT/"RC_AGG3_alpha_metrics.csv",index=False)

# paired stratified bootstrap for .001 vs .01
contr=[]
for obj in sorted(M.object.unique()):
    q=d[d.object==obj]
    wide=q.pivot(index=["image_id","image"],columns="alpha",values="tail_mean").dropna(subset=[.001,.01]).reset_index()
    yy=(wide.image.astype(str)=="bad").astype(int).to_numpy()
    s1=wide[.001].to_numpy(float); s10=wide[.01].to_numpy(float)
    obs=float(roc_auc_score(yy,s1)-roc_auc_score(yy,s10))
    bad=np.where(yy==1)[0]; good=np.where(yy==0)[0]
    rng=np.random.default_rng(SEED+sum(map(ord,obj)))
    vals=np.empty(B,float)
    for b in range(B):
        ii=np.r_[rng.choice(bad,len(bad),replace=True),rng.choice(good,len(good),replace=True)]
        yb=yy[ii]
        vals[b]=roc_auc_score(yb,s1[ii])-roc_auc_score(yb,s10[ii])
    lo,hi=np.quantile(vals,[.025,.975])
    m1=M[(M.object==obj)&np.isclose(M.alpha,.001)].iloc[0]
    m10=M[(M.object==obj)&np.isclose(M.alpha,.01)].iloc[0]
    large=M[(M.object==obj)&M.alpha.isin([.02,.05])]
    apd=float(m1.image_ap-m10.image_ap)
    ratio=float(m1.bad_gt_precision_median/m10.bad_gt_precision_median) if m10.bad_gt_precision_median>0 else np.inf
    E=bool((large.image_auc<=m10.image_auc+1e-12).any())
    verdict="CONTROL_ONLY"
    if obj=="macaroni2":
        okA=obs>=.05; okB=lo>0; okC=apd>=.05; okD=ratio>=2; okE=E
        verdict="TAIL_WIDTH_DILUTION_CAUSALLY_SUPPORTED" if all([okA,okB,okC,okD,okE]) else "NOT_ESTABLISHED"
    else:
        okA=okB=okC=okD=okE=np.nan
    contr.append(dict(object=obj,delta_auc_001_vs_01=obs,ci95_lo=float(lo),ci95_hi=float(hi),
        delta_ap_001_vs_01=apd,precision_ratio_001_vs_01=ratio,larger_width_not_better=E,
        gate_A=okA,gate_B=okB,gate_C=okC,gate_D=okD,gate_E=okE,verdict=verdict))
C=pd.DataFrame(contr); C.to_csv(OUT/"RC_AGG3_contrasts.csv",index=False)
v=C[C.object=="macaroni2"][["object","verdict"]]
v.to_csv(OUT/"RC_AGG3_VERDICT.csv",index=False)

manifest={
 "input_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),
 "prereg_sha256":hashlib.sha256(PRE.read_bytes()).hexdigest(),
}
try: manifest["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
except: manifest["git_commit"]="UNKNOWN"
(OUT/"INPUT_MANIFEST.json").write_text(json.dumps(manifest,indent=2))

report=["# RC-AGG3 Results","",C.round(6).to_string(index=False),"",
        "Alpha metrics:",M.round(6).to_string(index=False)]
(OUT/"RC_AGG3_REPORT.md").write_text("\\n".join(report))
print(C.round(6).to_string(index=False))
