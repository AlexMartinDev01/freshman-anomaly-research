# -*- coding: utf-8 -*-
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path.cwd()
CFG=json.loads((ROOT/"r6g_readout_decoupling"/"R6G_CONFIG.json").read_text(encoding="utf-8"))
OUT=ROOT/"results"/"model_v0"/"metrics"/"r6g_readout_decoupling"
A1F=ROOT/CFG["a1f_dir"]

m=pd.read_csv(OUT/"object_layer_metrics.csv")
prop=pd.read_csv(A1F/"dense_layer_curve.csv")
v0=pd.read_csv(A1F/"object_verdict.csv")

# frozen propagation curve: r=16 magnitude-controlled residual
p=prop[prop.radius==16][["object","layer_index","resid_mean"]].rename(
    columns={"layer_index":"layer","resid_mean":"propagation_resid"})

gap_rows=[]
relation_rows=[]
probe_state_rows=[]
verdict_rows=[]

for obj in CFG["objects"]:
    b=m[(m.object==obj)&(m.method=="1nn")].sort_values("layer")
    if len(b)!=12: raise RuntimeError(f"{obj}: missing 12-layer 1NN summary")
    a1frow=v0[v0.object==obj]
    a1f_verdict=str(a1frow.iloc[0].verdict) if len(a1frow) else "UNKNOWN"

    for probe in CFG["probes"]:
        q=m[(m.object==obj)&(m.method=="probe")&(m.probe==probe)].sort_values("layer")
        if len(q)!=12: raise RuntimeError(f"{obj}/{probe}: missing 12-layer probe summary")
        z=b[["layer","patch_auc","patch_ap","image_auc","tail_precision_bad","tail_recall_bad"]].merge(
            q[["layer","patch_auc","patch_ap","image_auc","tail_precision_bad","tail_recall_bad"]],
            on="layer",suffixes=("_1nn","_probe"))
        z["object"]=obj; z["probe"]=probe
        z["gap_auc"]=z.patch_auc_probe-z.patch_auc_1nn
        z["gap_ap"]=z.patch_ap_probe-z.patch_ap_1nn
        z["gap_img"]=z.image_auc_probe-z.image_auc_1nn
        z["gap_purity"]=z.tail_precision_bad_probe-z.tail_precision_bad_1nn
        z=z.merge(p[p.object==obj],on=["object","layer"],how="left")
        gap_rows.extend(z.to_dict("records"))

        if z.propagation_resid.isna().any():
            raise RuntimeError(f"{obj}: missing A1F r16 propagation values")
        rho_ap,p_ap=spearmanr(z.propagation_resid,z.gap_ap)
        rho_auc,p_auc=spearmanr(z.propagation_resid,z.gap_auc)
        rho_pur,p_pur=spearmanr(z.propagation_resid,z.gap_purity)
        relation_rows.append(dict(object=obj,probe=probe,a1f_verdict=a1f_verdict,
            rho_prop_gap_ap=rho_ap,p_prop_gap_ap=p_ap,
            rho_prop_gap_auc=rho_auc,p_prop_gap_auc=p_auc,
            rho_prop_gap_purity=rho_pur,p_prop_gap_purity=p_pur))

        e=z[z.layer==2].iloc[0]; f=z[z.layer==11].iloc[0]
        auc_drop=float(f.patch_auc_probe-e.patch_auc_probe)
        ap_drop=float(f.patch_ap_probe-e.patch_ap_probe)
        cue=("CUE_PRESERVED" if auc_drop>=-.03 and ap_drop>=-.10 else
             "CUE_FADED" if auc_drop<=-.05 and ap_drop<=-.15 else "CUE_MIXED")
        coupled=(rho_ap>=.60 and f.gap_ap>e.gap_ap and f.gap_purity>e.gap_purity)
        decoupled=(rho_ap<=.30 and f.gap_ap<=e.gap_ap+.02)
        gap_state="GAP_COUPLED" if coupled else ("GAP_DECOUPLED" if decoupled else "GAP_MIXED")
        probe_state_rows.append(dict(object=obj,probe=probe,
            early_probe_auc=e.patch_auc_probe,final_probe_auc=f.patch_auc_probe,
            cue_auc_change=auc_drop,early_probe_ap=e.patch_ap_probe,final_probe_ap=f.patch_ap_probe,
            cue_ap_change=ap_drop,cue_state=cue,
            early_gap_ap=e.gap_ap,final_gap_ap=f.gap_ap,gap_ap_growth=f.gap_ap-e.gap_ap,
            early_gap_purity=e.gap_purity,final_gap_purity=f.gap_purity,
            gap_purity_growth=f.gap_purity-e.gap_purity,
            rho_prop_gap_ap=rho_ap,gap_state=gap_state))

    st=pd.DataFrame([r for r in probe_state_rows if r["object"]==obj])
    if a1f_verdict=="PROPAGATION_SUPPORTED":
        if (st.cue_state=="CUE_PRESERVED").all() and (st.gap_state=="GAP_COUPLED").all():
            verdict="READOUT_DISTORTION_SUPPORTED"
        elif (st.cue_state=="CUE_FADED").all():
            verdict="CUE_FADING_SUPPORTED"
        elif (st.cue_state=="CUE_PRESERVED").all() and (st.gap_state=="GAP_DECOUPLED").all():
            verdict="PROPAGATION_DECOUPLED_FROM_READOUT"
        else:
            verdict="MIXED_INDETERMINATE"
    else:
        verdict="CONTROL_PROFILE_ONLY"
    verdict_rows.append(dict(object=obj,a1f_verdict=a1f_verdict,r6g_verdict=verdict))

pd.DataFrame(gap_rows).to_csv(OUT/"layerwise_readout_gaps.csv",index=False)
pd.DataFrame(relation_rows).to_csv(OUT/"propagation_gap_relations.csv",index=False)
pd.DataFrame(probe_state_rows).to_csv(OUT/"probe_states.csv",index=False)
pd.DataFrame(verdict_rows).to_csv(OUT/"R6G_FROZEN_OBJECT_VERDICTS.csv",index=False)

print("=== R6-G FROZEN OBJECT VERDICTS ===")
print(pd.DataFrame(verdict_rows).to_string(index=False))
print("\n=== Probe states ===")
print(pd.DataFrame(probe_state_rows)[[
    "object","probe","cue_state","gap_state","cue_auc_change","cue_ap_change",
    "gap_ap_growth","gap_purity_growth","rho_prop_gap_ap"
]].round(4).to_string(index=False))
print("\nR6 IS COMPLETE AFTER THIS GATE. Do not add another propagation-description experiment.")
