# -*- coding: utf-8 -*-
"""Checks the frozen protocol requires but the auto-report does not print.

Nothing here re-adjudicates a gate -- it only fills in the parts of the frozen
protocol (B3, and C2's dataset split) that the analysis script omits, so the
report is complete before review.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

D = r"results\model_v0\metrics\r5_rawmap"
sel = pd.read_csv(r"r5_rawmap_causal_interventions\R5_MAIN_SELECTION.csv")
ds = sel[["object", "dataset"]].drop_duplicates()
pd.set_option("display.width", 220)

print("=== B3 (frozen protocol): max must be ~constant after the first implant ===")
ex = pd.read_csv(D + r"\R5B_extent_normal_background.csv")
piv = ex[ex.alpha > 0].pivot_table(index=["object", "image"], columns="alpha",
                                   values="max")
d_max = piv.max(axis=1) - piv.min(axis=1)
print(f"  per-image spread of `max` across the 7 extent levels: "
      f"median {d_max.median():.3e}, max {d_max.max():.3e}")
print("  (h is fixed and strictly above the original max, so max is set by the")
print("   FIRST implant and cannot move afterwards -> B3 holds by construction)")

print("\n=== B mechanism: what actually moves as extent grows ===")
g = ex[ex.alpha > 0].groupby("alpha")[["top1", "max", "q"]].mean()
print(g.round(5).to_string())
print("\n  q = log(max/top1); max is pinned, so q falls exactly when top1 rises.")

print("\n=== C2 (frozen protocol): BOTH datasets must show a positive object effect ===")
co = pd.read_csv(D + r"\R5C_coherence_original.csv")
sh = pd.read_csv(D + r"\R5C_coherence_shuffle.csv")
cx = co.merge(sh, on=["object", "shot", "split", "alpha"], suffixes=("", "_sh"))
cx = cx.merge(ds, on="object", how="left")
cx["lcc_effect"] = cx.auc_lcc - cx.lcc_mean
cx["adj_effect"] = cx.auc_adjacency - cx.adjacency_mean
cx["lcc_over"] = cx.auc_lcc > cx.lcc_q95
cx["adj_over"] = cx.auc_adjacency > cx.adjacency_q95
for a, s in cx.groupby("alpha"):
    print(f"  alpha={a:.3f}")
    for d_, sd in s.groupby("dataset"):
        print(f"    {d_:5s} n_obj={sd.object.nunique():2d}  "
              f"lcc_effect {sd.lcc_effect.mean():+7.2f}  "
              f"adj_effect {sd.adj_effect.mean():+7.2f}  "
              f"lcc>q95 {int(sd.lcc_over.sum())}/{len(sd)}  "
              f"adj>q95 {int(sd.adj_over.sum())}/{len(sd)}")

print("\n=== C caveat: is the spatial statistic INDEPENDENT of, or REDUNDANT with, top1? ===")
cells = pd.read_csv(D + r"\R5_cells.csv")
m = cx[cx.alpha == 0.01].merge(cells, on=["object", "shot", "split"], how="left")
print(f"  corr(auc_lcc, baseline_auc_top1) at alpha=0.01 = "
      f"{np.corrcoef(m.auc_lcc, m.baseline_auc_top1)[0, 1]:+.3f}")
print(f"  corr(auc_adjacency, baseline_auc_top1)         = "
      f"{np.corrcoef(m.auc_adjacency, m.baseline_auc_top1)[0, 1]:+.3f}")
print("  NOTE: the frozen C test shows the spatial statistic is not DETERMINED BY")
print("  the histogram.  It does NOT test whether it ADDS to top1.  Do not")
print("  promote it as incremental evidence on this result alone.")

print("\n=== A: where does the q pathology live? ===")
ar = pd.read_csv(D + r"\R5A_object_summary.csv")
print(ar.groupby("dataset").agg(n=("object", "count"),
                                mean_rho_top1=("rho_top1", "mean"),
                                mean_rho_q=("rho_q", "mean"),
                                n_q_neg=("rho_q", lambda x: int((x < 0).sum()))).round(3).to_string())
print(ar.sort_values("rho_q")[["object", "dataset", "rho_top1", "rho_q"]].round(3).to_string(index=False))
