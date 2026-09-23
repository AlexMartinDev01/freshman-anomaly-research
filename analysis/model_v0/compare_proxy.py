# -*- coding: utf-8 -*-
"""
Proxy-E verdict: does a label-free pseudo-defect source reproduce what the
real-defect oracle (config E) achieved?

The success criteria are fixed BEFORE the run, so the result cannot be
rationalised afterwards:

  collapsed classes (can, wallplugs)
      frac_defect_below_p99 must FALL versus the baseline
      defect_mean must not collapse (the failure mode of config D)
      img_AUROC and AU-PRO should rise
  healthy classes (vial, sheet_metal)
      must NOT degrade significantly  -- v0's worst failure was breaking these

Reference points on the same held-out half:
  A_half  floor (frozen DINOv2)
  D_half  the failed attempt (no defect information)
  E       the oracle (real defect patches) -- the target to approach

Read the output as: F should look like E, not like D.

Usage: python analysis/model_v0/compare_proxy.py
"""
import os
import sys

import pandas as pd

RESULTS = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
COLLAPSED = ["can", "wallplugs"]
HEALTHY = ["vial", "sheet_metal"]
LABEL = {"A_half": "A  baseline", "D_half": "D  no defect info",
         "E": "E  ORACLE (real defects)", "F1": "F1 pseudo: texture",
         "F2": "F2 pseudo: structural", "F3": "F3 pseudo: feature-space",
         "F4": "F4 pseudo: feature rank-5", "F5": "F5 pseudo: feature rank-20",
         "F3_m10": "F3 + margin 0.10 (stronger hinge)"}
ORDER = ["A_half", "D_half", "E", "F1", "F2", "F3", "F4", "F5", "F3_m10"]
METRICS = ["img_AUROC", "px_AUROC", "AUPRO@0.05", "normal_p99",
           "defect_mean", "frac_defect_below_p99"]


def main():
    df = pd.read_csv(RESULTS)
    have = [c for c in ORDER if c in set(df.config)]
    if not have:
        sys.exit("no comparable configs found")

    for group, objs in (("COLLAPSED classes (must improve)", COLLAPSED),
                        ("HEALTHY classes (must not degrade)", HEALTHY)):
        print("=" * 108)
        print(group)
        print("=" * 108)
        for obj in objs:
            sub = df[(df.object == obj) & (df.config.isin(have))]
            if sub.empty:
                continue
            base = sub[sub.config == "A_half"]
            b = base.iloc[0] if len(base) else None
            print(f"\n--- {obj} ---")
            print(f"{'config':<28}" + "".join(f"{m[:12]:>14}" for m in METRICS)
                  + f"{'Δimg':>8}{'ΔAU-PRO':>9}")
            for c in have:
                r = sub[sub.config == c]
                if r.empty:
                    continue
                r = r.iloc[0]
                d_img = f"{r.img_AUROC - b.img_AUROC:+.1f}" if b is not None else "-"
                d_aup = (f"{r['AUPRO@0.05'] - b['AUPRO@0.05']:+.1f}"
                         if b is not None else "-")
                vals = "".join(
                    f"{r[m]:>14.3f}" if isinstance(r[m], float) else f"{r[m]:>14}"
                    for m in METRICS)
                print(f"{LABEL.get(c, c):<28}{vals}{d_img:>8}{d_aup:>9}")

    # ---- explicit verdict, per the pre-registered criteria ----
    print("\n" + "=" * 108)
    print("VERDICT (criteria fixed before the run)")
    print("=" * 108)
    f_cfgs = [c for c in have if c.startswith("F")]
    base = df[df.config == "A_half"].set_index("object")
    e_row = df[df.config == "E"].set_index("object")
    for c in f_cfgs:
        sub = df[df.config == c].set_index("object")
        ok_collapse, ok_healthy = [], []
        for obj in COLLAPSED:
            if obj not in sub.index or obj not in base.index:
                continue
            r, b = sub.loc[obj], base.loc[obj]
            ok_collapse.append(
                (r.frac_defect_below_p99 < b.frac_defect_below_p99)
                and (r.defect_mean > 0.5 * b.defect_mean))
        for obj in HEALTHY:
            if obj not in sub.index or obj not in base.index:
                continue
            r, b = sub.loc[obj], base.loc[obj]
            ok_healthy.append(r.img_AUROC >= b.img_AUROC - 2.0)
        e_gain = {o: (e_row.loc[o, "img_AUROC"] - base.loc[o, "img_AUROC"])
                  for o in base.index if o in e_row.index}
        c_gain = {o: (sub.loc[o, "img_AUROC"] - base.loc[o, "img_AUROC"])
                  for o in base.index if o in sub.index}
        print(f"\n{LABEL.get(c, c)}")
        print(f"  separation improved on collapsed classes: "
              f"{sum(ok_collapse)}/{len(ok_collapse)}")
        print(f"  healthy classes kept (within -2.0 AUROC): "
              f"{sum(ok_healthy)}/{len(ok_healthy)}")
        print("  AUROC gain vs baseline, per class (E = oracle for reference):")
        for o in base.index:
            if o in c_gain and o in e_gain:
                print(f"      {o:<14} F={c_gain[o]:+6.1f}   E={e_gain[o]:+6.1f}")
        verdict = ("F ~ E  -> proxy carries the defect-preserving signal"
                   if sum(ok_collapse) == len(ok_collapse)
                   else "F ~ D  -> proxy does NOT carry the signal")
        print(f"  => {verdict}")


if __name__ == "__main__":
    main()
