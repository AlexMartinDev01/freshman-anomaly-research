# -*- coding: utf-8 -*-
"""
Gate 5C verdict: is the detection gain carried by the defect cloud's ANCHOR,
its internal structure, or both?

The defect cloud is written as z = mu_d + r, with r the centred rank-40
residual (real coefficients). Gate 5A was confounded because R40 and H0b used
the same basis and the same real coefficients but placed the points
differently. Here the factors move one at a time:

    lambda series   mu(lam) = (1-lam) mu_n + lam mu_d,  z = mu(lam) + r
                    internal shape and coefficients held EXACTLY fixed
    J0 / R40        lambda = 1
    J1              lambda = 0   (anchor swapped to the normal centroid)

    J5              mu_d alone (tiny matched jitter)
    J2 / J3         mu_d + U40 x isotropic / covariance-matched coefficients

PROTOCOL
  Judgement objects are wallplugs and sheet_metal -- the two where E produces a
  real gain. vial is the do-no-harm control. can is RECORDED ONLY: E's own
  effect there is negative (53.7 -> 51.8), so it cannot adjudicate whether a
  mechanism reproduces "E's gain".
  A Hilbert-style caveat applies to J5/J2/J3: they are geometrically
  informative but their hinges are inactive on most objects, so a null result
  there means "the constraint never engaged", NOT "coefficients do not matter".

Constraint-activity fractions were measured in the build step and are repeated
here so the reader can see which rows may be read as intervention evidence.
"""
import sys

import pandas as pd

CSV = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
PRIMARY = ["wallplugs", "sheet_metal"]
GATE = ["vial"]
RECORD = ["can"]
# measured below-hinge fraction (%) from the Gate 5C build
HINGE = {
    "R40": {"can": 74, "wallplugs": 86, "vial": 74, "sheet_metal": 55},
    "Jl75":   {"can": 74, "wallplugs": 85, "vial": 72, "sheet_metal": 58},
    "Jl50":   {"can": 70, "wallplugs": 82, "vial": 64, "sheet_metal": 59},
    "Jl25":   {"can": 70, "wallplugs": 73, "vial": 49, "sheet_metal": 60},
    "J1":     {"can": 52, "wallplugs": 63, "vial": 16, "sheet_metal": 59},
    "J5":     {"can": 0, "wallplugs": 100, "vial": 100, "sheet_metal": 0},
    "J2":     {"can": 0, "wallplugs": 3, "vial": 1, "sheet_metal": 0},
    "J3":     {"can": 19, "wallplugs": 51, "vial": 29, "sheet_metal": 4},
}
LAM_ORDER = ["J1", "Jl25", "Jl50", "Jl75", "R40"]
LAM_NAME = {"J1": "lambda=0.00 (mu_n)", "Jl25": "lambda=0.25",
            "Jl50": "lambda=0.50", "Jl75": "lambda=0.75",
            "R40": "lambda=1.00 (mu_d, = R40)"}


def main():
    df = pd.read_csv(CSV)
    if not any(c in set(df.config) for c in ("J1", "Jl25", "Jl50", "Jl75")):
        sys.exit("no Gate 5C configs - run run_oracle.py --configs "
                 "J1 Jl25 Jl50 Jl75 --eval-half")
    base = df[df.config == "A_half"].set_index("object")
    e_row = df[df.config == "E"].set_index("object")

    print("=" * 108)
    print("GATE 5C -- anchor interpolation. Internal shape and coefficients are "
          "IDENTICAL across rows;")
    print("           only the centroid mu(lambda) = (1-lambda) mu_n + lambda "
          "mu_d moves.")
    print("=" * 108)
    for group, objs in (("PRIMARY (E has a real gain)", PRIMARY),
                        ("DO-NO-HARM control", GATE),
                        ("RECORDED ONLY (E's effect is negative here)", RECORD)):
        print(f"\n### {group}")
        for o in objs:
            if o not in base.index:
                continue
            print(f"\n--- {o} ---   A_half img={base.loc[o,'img_AUROC']:.1f}   "
                  f"E img={e_row.loc[o,'img_AUROC']:.1f}   "
                  f"R40 img={df[(df.object==o)&(df.config=='R40')].iloc[0]['img_AUROC']:.1f}")
            print(f"  {'config':<24}{'hinge':>7}{'img_AUROC':>11}{'AUPRO':>8}"
                  f"{'normal_p99':>12}{'defect_mean':>13}{'defect<p99':>12}")
            for c in LAM_ORDER:
                r = df[(df.object == o) & (df.config == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                h = HINGE.get(c, {}).get(o, -1)
                print(f"  {LAM_NAME[c]:<24}{h:>6}%{r.img_AUROC:>11.1f}"
                      f"{r['AUPRO@0.05']:>8.1f}{r.normal_p99:>12.3f}"
                      f"{r.defect_mean:>13.3f}{r.frac_defect_below_p99:>12.3f}")

    print("\n" + "=" * 108)
    print("READING")
    print("=" * 108)
    for o in PRIMARY:
        r0 = df[(df.object == o) & (df.config == "J1")]
        r1 = df[(df.object == o) & (df.config == "R40")]
        if r0.empty or r1.empty:
            continue
        a, b = r0.iloc[0], r1.iloc[0]
        print(f"  {o:<12} anchor swap lambda=0 -> lambda=1:  "
              f"img {a.img_AUROC:.1f} -> {b.img_AUROC:.1f}   "
              f"AU-PRO {a['AUPRO@0.05']:.1f} -> {b['AUPRO@0.05']:.1f}   "
              f"defect_mean {a.defect_mean:.3f} -> {b.defect_mean:.3f}")
    print("\n  A monotone ramp along lambda that follows the anchor, with internal")
    print("  shape and coefficients held fixed, is the intervention evidence that")
    print("  the cloud's ABSOLUTE PLACEMENT carries the detection benefit.")

    print("\n" + "=" * 108)
    print("GEOMETRY-ONLY (hinge inactive on most objects -- NOT intervention "
          "evidence)")
    print("=" * 108)
    # measured during the hinge check; J5/J2/J3 were deliberately NOT run, so
    # these are geometry numbers, not run outputs
    GEOM = {
        "can":         {"R40": 0.127, "J5": 0.266, "J2": 0.462, "J3": 0.272},
        "wallplugs":   {"R40": 0.134, "J5": 0.122, "J2": 0.266, "J3": 0.198},
        "vial":        {"R40": 0.168, "J5": 0.171, "J2": 0.345, "J3": 0.264},
        "sheet_metal": {"R40": 0.171, "J5": 0.214, "J2": 0.336, "J3": 0.249},
    }
    print(f"  {'object':<12}{'R40':>9}{'J5 mu_d':>10}{'J2 iso':>10}{'J3 cov':>10}"
          f"    mean 1-NN distance to the bank (measured, not run)")
    for o in base.index:
        g = GEOM.get(o, {})
        print(f"  {o:<12}" + "".join(f"{g.get(c, float('nan')):>10.3f}"
                                     for c in ("R40", "J5", "J2", "J3")))
    print("\n  Same anchor mu_d, only the coefficient source changes: isotropic (J2)")
    print("  and even covariance-matched (J3) coefficients push the cloud far from")
    print("  the bank, while the real coefficients keep it close. So the placement")
    print("  is set by higher-order (multi-modal) structure of the coefficient")
    print("  distribution -- not by its mean or covariance.")


if __name__ == "__main__":
    main()
