# -*- coding: utf-8 -*-
"""
Gate 3 verdict: does the normal-residual-band subspace have CAUSAL value?

Q1 feeds the same separation loss as F1-F5, but draws its directions from
normal-PCA dims [20, 45) -- the only normal-only candidate that partially beat
the sham null in Gate 2 (wallplugs 0.166 vs 0.114, can 0.091 vs 0.074; it lost
on sheet_metal and vial).

This is the last gate on the normal-only route. All THREE criteria must hold
simultaneously; passing on detection alone is not enough, because config D
already showed that hurting the defect signal can still shuffle AUROC around.

  1. TAIL          normal_p99 falls below the A_half baseline
  2. PRESERVATION  defect_mean does not collapse (>= DEFECT_FLOOR of baseline)
                   AND frac_defect_below_p99 falls
  3. DETECTION     collapsed classes (can, wallplugs) gain on img_AUROC AND
                   AU-PRO, and healthy classes (vial, sheet_metal) stay within
                   HEALTH_TOL of baseline

Thresholds are fixed here, before the numbers are read.

METHODOLOGICAL CAVEAT that must travel with any positive result: the band
[20, 45) was chosen by inspecting real-defect overlap. Success would license
"this normal-derived subspace has intervention value", NOT "normal data alone
can find it". Selecting the band without defects is a separate problem.

Usage: python analysis/model_v0/gate3_verdict.py
"""
import sys

import pandas as pd

RESULTS = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
COLLAPSED = ["can", "wallplugs"]
HEALTHY = ["vial", "sheet_metal"]
DEFECT_FLOOR = 0.80      # defect_mean must stay >= 80% of baseline
HEALTH_TOL = 2.0         # healthy classes may lose at most 2.0 img AUROC


def main():
    df = pd.read_csv(RESULTS)
    if "Q1" not in set(df.config):
        sys.exit("Q1 not found - run run_oracle.py --configs Q1 --eval-half first")
    base = df[df.config == "A_half"].set_index("object")
    q1 = df[df.config == "Q1"].set_index("object")

    print("=" * 100)
    print("GATE 3 -- Q1 (normal residual band dims 20-45) vs A_half baseline")
    print("=" * 100)
    ref = {c: df[df.config == c].set_index("object")
           for c in ("E", "F3") if c in set(df.config)}
    for o in base.index:
        if o not in q1.index:
            continue
        b, r = base.loc[o], q1.loc[o]
        print(f"\n--- {o} ---")
        print(f"  {'metric':<26}{'A_half':>10}{'Q1':>10}{'delta':>10}"
              + "".join(f"{c:>10}" for c in ref))
        for m in ("img_AUROC", "AUPRO@0.05", "normal_p99", "defect_mean",
                  "frac_defect_below_p99"):
            extras = "".join(f"{ref[c].loc[o, m]:>10.3f}" for c in ref)
            print(f"  {m:<26}{b[m]:>10.3f}{r[m]:>10.3f}{r[m]-b[m]:>+10.3f}{extras}")

    print("\n" + "=" * 100)
    print("VERDICT (all three must hold)")
    print("=" * 100)
    c1, c2, c3 = [], [], []
    for o in base.index:
        if o not in q1.index:
            continue
        b, r = base.loc[o], q1.loc[o]
        tail = r.normal_p99 < b.normal_p99
        pres = (r.defect_mean >= DEFECT_FLOOR * b.defect_mean
                and r.frac_defect_below_p99 < b.frac_defect_below_p99)
        if o in COLLAPSED:
            det = r.img_AUROC > b.img_AUROC and r["AUPRO@0.05"] > b["AUPRO@0.05"]
        else:
            det = r.img_AUROC >= b.img_AUROC - HEALTH_TOL
        c1.append(tail); c2.append(pres); c3.append(det)
        print(f"  {o:<12} tail_ok={str(tail):<6} preserve_ok={str(pres):<6} "
              f"detect_ok={str(det):<6}"
              f"   (p99 {b.normal_p99:.3f}->{r.normal_p99:.3f}, "
              f"defect {b.defect_mean:.3f}->{r.defect_mean:.3f}, "
              f"img {b.img_AUROC:.1f}->{r.img_AUROC:.1f})")
    print(f"\n  criterion 1 (tail falls):        {sum(c1)}/{len(c1)}")
    print(f"  criterion 2 (defect preserved):  {sum(c2)}/{len(c2)}")
    print(f"  criterion 3 (detection):         {sum(c3)}/{len(c3)}")
    allok = sum(c1) == len(c1) and sum(c2) == len(c2) and sum(c3) == len(c3)
    print("\n  " + ("=> PASS: the normal-derived band has causal value. "
                    "Note the caveat: the band was defect-informed."
                    if allok else
                    "=> FAIL: stop the normal-only subspace route and move to "
                    "external anomaly prior + target conditioning."))


if __name__ == "__main__":
    main()
