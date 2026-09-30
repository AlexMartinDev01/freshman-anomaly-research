# -*- coding: utf-8 -*-
"""Post-hoc decomposition of the frozen TSN confirmatory result.

Reads only tsn_formal_v2.csv.  Nothing here changes the protocol, the formula
or the gate -- the confirmatory verdict is already fixed by the gate file.
Everything below is mechanism diagnosis of a FAILED confirmatory test.
"""
import os
import numpy as np
import pandas as pd

P = r"results\model_v0\metrics\tsn_formal_v2.csv"
OUT = r"results\model_v0\metrics"
pd.set_option("display.width", 220)

d = pd.read_csv(P)
d["delta"] = d.delta_tsn_vs_agg
print(f"cells {len(d)} | mean delta {d.delta.mean():.2f} | nonneg "
      f"{(d.delta >= 0).mean():.4f}")

print("\n=== per object (mean delta, sorted) ===")
po = (d.groupby(["dataset", "object"])
        .agg(n=("delta", "size"), base=("baseline_agg_img_AUROC", "mean"),
             tsn=("tsn_img_AUROC", "mean"), delta=("delta", "mean"))
        .sort_values("delta", ascending=False).round(2))
print(po.to_string())

print("\n=== hard quartile: composition and what it is made of ===")
for ds, g in d.groupby("dataset"):
    thr = g.baseline_agg_img_AUROC.quantile(.25)
    hard = g[g.baseline_agg_img_AUROC <= thr]
    print(f"\n{ds}: threshold {thr:.2f}, n_hard {len(hard)}, "
          f"mean delta {hard.delta.mean():+.2f} (gate needs >= +2.0)")
    print(hard.groupby("object").delta.agg(["size", "mean"]).round(2).to_string())

print("\n=== does delta shrink as baseline improves? ===")
for ds, g in d.groupby("dataset"):
    r = np.corrcoef(g.baseline_agg_img_AUROC, g.delta)[0, 1]
    print(f"  {ds}: corr(baseline, delta) = {r:+.3f}")

print("\n=== the positive cells in full ===")
pos = d[d.delta >= 0].sort_values("delta", ascending=False)
print(pos[["dataset", "object", "shot", "split", "baseline_agg_img_AUROC",
           "tsn_img_AUROC", "delta"]].round(2).to_string(index=False))
print(f"\n  {len(pos)}/{len(d)} positive; "
      f"of those, macaroni1/2 = {int(pos.object.str.startswith('macaroni').sum())}")

print("\n=== macaroni1/2 vs everything else ===")
m = d.object.str.startswith("macaroni")
print(f"  macaroni1/2 : n={int(m.sum())}, mean baseline "
      f"{d[m].baseline_agg_img_AUROC.mean():.2f}, mean delta {d[m].delta.mean():+.2f}")
print(f"  all others  : n={int((~m).sum())}, mean baseline "
      f"{d[~m].baseline_agg_img_AUROC.mean():.2f}, mean delta {d[~m].delta.mean():+.2f}")

print("\n=== by shot and split ===")
print(d.groupby("shot").delta.agg(["size", "mean", "median"]).round(2).to_string())
print(d.groupby("split").delta.agg(["size", "mean", "median"]).round(2).to_string())

print("\n=== worst 10 cells ===")
print(d.nsmallest(10, "delta")[["dataset", "object", "shot", "split",
                                "baseline_agg_img_AUROC", "tsn_img_AUROC",
                                "delta"]].round(2).to_string(index=False))

d.groupby(["dataset", "object"]).delta.agg(["size", "mean", "median"]).round(3) \
 .to_csv(os.path.join(OUT, "tsn_formal_v2_by_object.csv"))
d.groupby(["dataset", "shot", "split"]).delta.agg(["size", "mean", "median"]).round(3) \
 .to_csv(os.path.join(OUT, "tsn_formal_v2_by_cell.csv"))
print("\nwrote tsn_formal_v2_by_object.csv / _by_cell.csv")
