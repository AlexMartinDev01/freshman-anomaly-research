# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

ROOT=Path.cwd()
D=ROOT/"results"/"model_v0"/"metrics"/"r6g_readout_decoupling"
g=pd.read_csv(D/"layerwise_readout_gaps.csv")
v=pd.read_csv(D/"R6G_FROZEN_OBJECT_VERDICTS.csv")

lines=["# R6-G Final Root-Cause Gate Report","",
       "## Frozen verdicts","",
       "```text",v.to_string(index=False),"```",""]

for obj in v.object:
    lines += [f"## {obj}",""]
    q=g[(g.object==obj)&(g.probe=="logreg")][[
        "layer","patch_auc_1nn","patch_auc_probe","patch_ap_1nn","patch_ap_probe",
        "gap_ap","image_auc_1nn","image_auc_probe","gap_purity","propagation_resid"]]
    lines += ["```text",q.round(4).to_string(index=False),"```",""]

(D/"R6G_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print("wrote",D/"R6G_REPORT.md")
