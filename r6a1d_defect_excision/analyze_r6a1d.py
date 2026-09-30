# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, pandas as pd, numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--indir",default=r"results\model_v0\metrics\r6a1d_defect_excision")
a=ap.parse_args()
d=Path(a.indir)
s=pd.read_csv(d/"object_radius_alpha_summary.csv")

print("=== R6-A1D DEFECT EXCISION / FAR-FIELD AUDIT ===")
for obj in sorted(s.object.unique()):
    print("\nOBJECT",obj)
    x=s[s.object==obj].copy()
    p=x.pivot(index="radius",columns="alpha",values="pairwise_auc_centered")
    print("Centered far-field pairwise AUC")
    print(p.round(2).to_string())
    y=x[x.alpha==.01][[
        "radius","pairwise_auc_centered","random_mean",
        "random_q05","random_q95","centered_minus_random_mean",
        "mean_removed_fraction"
    ]]
    print("\nalpha=1% centered vs random-mask null")
    print(y.round(3).to_string(index=False))

# Compact decision aid at alpha=1% only; no best-alpha selection.
z=s[s.alpha==.01].copy()
rows=[]
for obj,g in z.groupby("object"):
    g=g.sort_values("radius")
    base=g[g.radius==0].iloc[0] if (g.radius==0).any() else g.iloc[0]
    r8=g[g.radius==8].iloc[0] if (g.radius==8).any() else None
    r16=g[g.radius==16].iloc[0] if (g.radius==16).any() else None
    rows.append({
        "object":obj,
        "auc_r0":base.pairwise_auc_centered,
        "auc_r8":np.nan if r8 is None else r8.pairwise_auc_centered,
        "auc_r16":np.nan if r16 is None else r16.pairwise_auc_centered,
        "r16_minus_random":np.nan if r16 is None else r16.centered_minus_random_mean
    })
v=pd.DataFrame(rows)
v.to_csv(d/"alpha1pct_decision_table.csv",index=False)
print("\n=== alpha=1% decision table ===")
print(v.round(3).to_string(index=False))

print("\nInterpretation boundary:")
print("Persistent AUC > .5 after large-radius excision means far-field label-associated signal exists.")
print("It does NOT by itself tell whether that signal is legitimate anomaly context, unannotated defect,")
print("acquisition covariate, or dataset shortcut.")
