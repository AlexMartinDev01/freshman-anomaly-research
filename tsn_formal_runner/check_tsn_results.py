from pathlib import Path
import sys,pandas as pd,numpy as np
p=Path(sys.argv[1] if len(sys.argv)>1 else r"results\model_v0\metrics\tsn_formal_v2.csv")
if not p.exists(): raise SystemExit(f"Missing {p}")
d=pd.read_csv(p)
print("Rows:",len(d))
print("Unique cells:",len(d.drop_duplicates(["object","shot","split"])))
print("Datasets:",d.dataset.value_counts().to_dict())
print("Shots:",d.shot.value_counts().sort_index().to_dict())
print("Splits:",d["split"].value_counts().sort_index().to_dict())
print("\nBy dataset:")
print(d.groupby("dataset").agg(
    cells=("object","size"),
    baseline=("baseline_agg_img_AUROC","mean"),
    tsn=("tsn_img_AUROC","mean"),
    delta=("delta_tsn_vs_agg","mean"),
    nonnegative=("delta_tsn_vs_agg",lambda s:(s>=0).mean()),
).to_string())
print("\nWorst 15 cells:")
print(d.sort_values("delta_tsn_vs_agg").head(15)[
    ["dataset","object","shot","split","baseline_agg_img_AUROC","tsn_img_AUROC","delta_tsn_vs_agg","elapsed_sec"]
].to_string(index=False))
gate=p.with_name(p.stem+"_gate.csv")
if gate.exists():
    print("\nGate:")
    print(pd.read_csv(gate).to_string(index=False))
else:
    print("\nNo gate file yet.")
