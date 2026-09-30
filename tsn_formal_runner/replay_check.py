# -*- coding: utf-8 -*-
"""Baseline replay check.

The TSN runner recomputes the historical agg_all3 / final_raw image AUROC from
scratch.  If those do not reproduce the frozen values for the same
(object, shot, split), then any TSN number is uninterpretable -- a mismatch is
an experiment-reproduction failure, not a TSN failure (protocol section 9).
"""
import sys
import pandas as pd

HIST_MV = r"results\model_v0\metrics\gate_r2_layer_confirm.csv"
HIST_ALL = r"results\model_v0\metrics\m1confirm_all.csv"   # repr r0 == agg_all3
NEW = [r"results\model_v0\metrics\tsn_smoke.csv",
       r"results\model_v0\metrics\tsn_smoke_visa.csv"]

h = pd.read_csv(HIST_MV)
hall = pd.read_csv(HIST_ALL)
hall = hall[hall.repr.astype(str) == "r0"]
new = pd.concat([pd.read_csv(p) for p in NEW]) if len(sys.argv) < 2 else \
      pd.read_csv(sys.argv[1])

rows = []
for _, r in new.iterrows():
    for cfg, col in (("final_raw", "baseline_final_img_AUROC"),
                     ("agg_all3", "baseline_agg_img_AUROC")):
        old, src = None, None
        if cfg == "agg_all3":
            # 27-category source first -- it is the only one covering VisA
            m2 = hall[(hall.object == r.object) & (hall.shot == r.shot) &
                      (hall["split"] == r["split"])]
            if len(m2):
                old, src = float(m2.img_AUROC.iloc[0]), "m1confirm_all/r0"
        m = h[(h.object == r.object) & (h.shot == r.shot) &
              (h["split"] == r["split"]) & (h.config == cfg)]
        if len(m):
            old2 = float(m.img_AUROC.iloc[0])
            src = (src + "+gate_r2") if src else "gate_r2_layer_confirm"
            if old is None or abs(old - old2) < 1e-6:
                old = old2
        rows.append(dict(object=r.object, shot=int(r.shot), split=int(r["split"]),
                         config=cfg, source=src,
                         historical=None if old is None else round(old, 6),
                         new=round(float(r[col]), 6),
                         abs_diff=None if old is None else abs(old - float(r[col]))))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.to_string(index=False))
cmp = d[d.abs_diff.notna()]
if len(cmp):
    print(f"\ncompared {len(cmp)} (cell, config) pairs")
    print(f"  max |delta| = {cmp.abs_diff.max():.3e}")
    print(f"  exact (<=1e-6): {int((cmp.abs_diff <= 1e-6).sum())}/{len(cmp)}")
    print("  VERDICT:", "REPLAY OK" if cmp.abs_diff.max() <= 1e-6
          else "*** REPLAY MISMATCH -> stop, this is a pipeline failure ***")
else:
    print("\nno historical counterpart found for these cells "
          "(VisA has no gate_r2_layer_confirm entry)")
