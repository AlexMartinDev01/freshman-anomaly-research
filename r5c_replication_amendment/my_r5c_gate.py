# -*- coding: utf-8 -*-
"""R5-C locked-replication criterion, evaluated PER SHOT as the amendment requires.

The kit's r5_analyze_main.py summarises R5-C by alpha only -- it pools the two
shot counts together.  The amendment's frozen criterion is stated "for each shot
separately", so pooling would not actually test it.  This script implements the
amendment verbatim and nothing else:

  for each shot in {2, 8}:
    1. adjacency_original > shuffle_q95 in >= 70% of objects, at EVERY
       pre-registered alpha
    2. MVTec and VisA both have positive mean adjacency effect at EVERY alpha
  LCC is reported as secondary and never decides on its own.
  No alpha is selected post-hoc; all four are reported.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--indir", required=True)
ap.add_argument("--selection", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
d, sel = Path(a.indir), pd.read_csv(a.selection)
pd.set_option("display.width", 220)

co = pd.read_csv(d / "R5C_coherence_original.csv")
sh = pd.read_csv(d / "R5C_coherence_shuffle.csv")
dsmap = sel[["object", "dataset"]].drop_duplicates()
cx = co.merge(sh, on=["object", "shot", "split", "alpha"], suffixes=("", "_sh"))
cx = cx.merge(dsmap, on="object", how="left")
cx["adj_effect"] = cx.auc_adjacency - cx.adjacency_mean
cx["lcc_effect"] = cx.auc_lcc - cx.lcc_mean
cx["adj_over"] = cx.auc_adjacency > cx.adjacency_q95
cx["lcc_over"] = cx.auc_lcc > cx.lcc_q95
print(f"=== {a.label or a.indir} ===")
print(f"rows {len(cx)} | objects {cx.object.nunique()} | shots {sorted(cx.shot.unique())} "
      f"| splits {sorted(cx['split'].unique())} | alphas {sorted(cx.alpha.unique())}")

rows = []
for shot in sorted(cx.shot.unique()):
    s = cx[cx.shot == shot]
    for alpha in sorted(s.alpha.unique()):
        g = s[s.alpha == alpha]
        n = g.object.nunique()
        over = int(g.adj_over.sum())
        per_ds = g.groupby("dataset").adj_effect.mean()
        rows.append(dict(shot=shot, alpha=alpha, n_objects=n,
                         adj_over_q95=over, adj_frac=over / n,
                         adj_effect_mean=g.adj_effect.mean(),
                         mvtec_effect=float(per_ds.get("mvtec", np.nan)),
                         visa_effect=float(per_ds.get("visa", np.nan)),
                         lcc_over_q95=int(g.lcc_over.sum()),
                         lcc_effect_mean=g.lcc_effect.mean(),
                         crit1=bool(over / n >= .70),
                         crit2=bool((per_ds > 0).all())))
t = pd.DataFrame(rows)
t["alpha_PASS"] = t.crit1 & t.crit2
print(t.round(3).to_string(index=False))

print("\n--- per-shot verdict (all four alphas must hold) ---")
for shot in sorted(t.shot.unique()):
    s = t[t.shot == shot]
    ok = bool(s.alpha_PASS.all())
    worst = s.adj_frac.min()
    print(f"  shot={shot}: {'PASS' if ok else 'FAIL'}   "
          f"min object fraction over q95 across alphas = {worst:.3f} "
          f"({int(s.adj_over_q95.min())}/{int(s.n_objects.iloc[0])})")
both = all(bool(t[t.shot == s].alpha_PASS.all()) for s in t.shot.unique())
one = any(bool(t[t.shot == s].alpha_PASS.all()) for s in t.shot.unique())
print("\n  AMENDMENT VERDICT:",
      "spatial signal is CROSS-SHOT REPLICATED" if both else
      ("SHOT-DEPENDENT -- do not promote as a general branch" if one else
       "R5-C MAIN DOES NOT REPLICATE -- spatial branch closed"))
t.to_csv(d / "R5C_pershot_gate.csv", index=False)

print("\n--- per-object detail at alpha=0.01 (adjacency) ---")
g = cx[cx.alpha == 0.01].sort_values(["shot", "adj_effect"])
print(g[["shot", "object", "dataset", "auc_adjacency", "adjacency_q95",
         "adj_effect", "adj_over"]].round(2).to_string(index=False))
