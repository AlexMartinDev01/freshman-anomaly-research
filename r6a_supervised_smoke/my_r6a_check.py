# -*- coding: utf-8 -*-
"""Independent read of the R6-A smoke.  Diagnosis only; no gate is changed."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = r"results\model_v0\metrics\r6a_smoke"
pd.set_option("display.width", 220)
s = pd.read_csv(D + r"\summary.csv")
f = pd.read_csv(D + r"\folds.csv")

print("=== folds: one assignment per image, reused by both probes ===")
print(f"  fold rows {len(f)} | images {f.image.nunique()} | objects {f.object.nunique()}")
print(f"  every image has exactly one fold: "
      f"{bool((f.groupby(['object','image']).fold.nunique() == 1).all())}")

HARD = ["screw", "macaroni2", "pcb2"]
EASY = ["bottle", "cable", "chewinggum"]
print("\n=== per-probe table (no best-of selection) ===")
print(s[["object", "probe", "n_folds", "patch_auc_probe", "patch_auc_1nn",
         "patch_auc_gain", "image_auc_probe", "image_auc_1nn",
         "image_auc_gain"]].round(4).to_string(index=False))

print("\n=== frozen smoke criterion, both probes separately ===")
for pr in ["logreg", "linsvm"]:
    g = s[s.probe == pr]
    h = g[g.object.isin(HARD)]
    ok = h[(h.patch_auc_probe >= .90) & (h.patch_auc_gain >= 10)]
    print(f"  {pr:7s} hard>=.90 & gain>=+10 : {len(ok)}/3  {list(ok.object)}")
    print(f"          MVTec mean patch gain {g[g.object.isin(['screw','cable','bottle'])].patch_auc_gain.mean():+.2f}"
          f" | VisA {g[g.object.isin(['macaroni2','pcb2','chewinggum'])].patch_auc_gain.mean():+.2f}"
          f" | easy controls min patch_auc_probe "
          f"{g[g.object.isin(EASY)].patch_auc_probe.min():.3f}")

print("\n=== why the 'gain' bar failed: gain tracks remaining headroom, not signal ===")
g = s[s.probe == "logreg"].copy()
print(f"  probe patch AUC range : {g.patch_auc_probe.min():.3f}..{g.patch_auc_probe.max():.3f}"
      f"  (spread {g.patch_auc_probe.max()-g.patch_auc_probe.min():.3f})")
print(f"  1NN   patch AUC range : {g.patch_auc_1nn.min():.3f}..{g.patch_auc_1nn.max():.3f}"
      f"  (spread {g.patch_auc_1nn.max()-g.patch_auc_1nn.min():.3f})")
print("  => gain = probe - 1nn is dominated by the 1NN term, because the probe term")
print("     barely moves across objects. A near -1 correlation here is an ARITHMETIC")
print("     artefact of a near-constant subtrahend, not evidence about difficulty.")
print(g.assign(headroom=(1 - g.patch_auc_1nn))[
    ["object", "patch_auc_1nn", "headroom", "patch_auc_gain"]].round(3).to_string(index=False))

print("\n=== the dissociation that matters: does AGGREGATION lose the patch signal? ===")
d = s[s.probe == "logreg"].copy()
d["loss_1nn"] = d.patch_auc_1nn - d.image_auc_1nn
d["loss_probe"] = d.patch_auc_probe - d.image_auc_probe
d["loss_diff"] = d.loss_1nn - d.loss_probe
print(d[["object", "patch_auc_1nn", "image_auc_1nn", "loss_1nn",
         "patch_auc_probe", "image_auc_probe", "loss_probe", "loss_diff"]]
      .round(3).sort_values("loss_diff", ascending=False).to_string(index=False))
print("  loss_diff > 0 means the top1% aggregation destroys MORE of the 1NN signal")
print("  than of the supervised direction's signal, on the same held-out images.")
print(f"  objects with loss_diff > 0: {int((d.loss_diff > 0).sum())}/6")

print("\n=== hard objects: is the difficulty at patch level or at image level? ===")
print(d[d.object.isin(HARD)][["object", "patch_auc_1nn", "image_auc_1nn",
                              "patch_auc_probe", "image_auc_probe"]].round(3).to_string(index=False))
