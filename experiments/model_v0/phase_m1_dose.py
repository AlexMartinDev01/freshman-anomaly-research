# -*- coding: utf-8 -*-
"""
Phase M1-dose -- the 448 / 560 / 672 resolution response (criterion C4).

C4 (docs/PHASE_M1_CONFIRM_PREREGISTRATION.md section 2):

    mean C_feat and mean AUPRO increase monotonically across the three points

Without the middle point, "448 -> 672" would be two samples, and the gain
could be an artefact of one resize/interpolation setting rather than a dose
response.

Subset: the 7 objects fixed in docs/PHASE_M1_PREREGISTRATION.md section 2,
selected by MEASURED baseline C_feat (lowest 3 MVTec + lowest 3 VisA + the
highest MVTec object as a control) -- never by the 672 gain, so conditioning
on it does not make this test circular.  Design matches the smoke exactly:
shots {1, 4} x 3 splits, so the 448 and 672 endpoints are the ones already on
disk and only the 560 middle point is new.

Two disclosures this script prints rather than hides:

  1. C_feat is compared here ACROSS resolutions, which is exactly the scale
     confound the M1 pre-registration section 4 documents.  C4 names C_feat
     literally, so it is computed literally; `sep_auc` (scale-free) is
     reported beside it.  If C_feat is non-monotone while sep_auc is
     monotone, that is a scale artefact and has to be said out loud rather
     than filed under "C4 failed".
  2. `cache_m1_repr.py --edge 560` writes to cache_m1_dino560/ but keeps the
     `dino672_` filename prefix (the prefix is keyed on --repr, not --edge).
     Harmless, but it is why the path and the prefix below disagree.

Usage:
    python experiments/model_v0/phase_m1_dose.py --measure
    python experiments/model_v0/phase_m1_dose.py --analyze
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
import phase_m1_rescue as P  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
HARD = ["transistor", "screw", "zipper", "pcb4", "pcb2", "pcb3"]
CONTROL = ["carpet"]
ALL7 = HARD + CONTROL
SOURCES = ["phase_m1_A_mvtec", "phase_m1_B_visa"]


def measure(shots, splits):
    """Run r560 on the 7-object dose subset, same design as the smoke."""
    root560 = os.path.join(RESULTS, "cache_m1_dino560")
    if not os.path.isdir(root560):
        raise SystemExit(f"{root560} missing -- run:\n"
                         f"  python experiments/model_v0/cache_m1_repr.py "
                         f"--repr dino672 --edge 560")
    # Additive monkeypatch: phase_m1_rescue.py is frozen, so the new point is
    # injected here rather than edited in.  `run()` only uses repr_name for the
    # map directory and the output row; `load_obj()` reads CACHE[repr_name].
    P.CACHE["r560"] = {"mvtec_tr": root560, "mvtec_te": root560,
                       "visa": root560, "layers": ["mid", "midlate", "final"],
                       "pfx": "dino672_"}
    rows, t0 = [], time.time()
    for obj in ALL7:
        rn = "r560"
        root, layers, tr, te = P.load_obj(rn, obj)
        for shot in shots:
            for sp in range(splits):
                P.run(rn, obj, shot, sp, rows, root, layers, tr, te)
        del tr, te
        print(f"  {obj:<12} {rn} done ({time.time() - t0:6.0f}s, "
              f"{len(rows)} rows)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, "phase_m1_dose_r560.csv"),
                                  index=False)
    pd.DataFrame(rows).to_csv(os.path.join(METRICS, "phase_m1_dose_r560.csv"),
                              index=False)
    print(f"\n  {len(rows)} rows -> phase_m1_dose_r560.csv")


def analyze(shots):
    parts = []
    for t in SOURCES:
        p = os.path.join(METRICS, f"{t}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p)
        parts.append(d[d["repr"].isin(["r0", "r1"])])
    base = pd.concat(parts, ignore_index=True)
    base = base[base.shot.isin(shots)]
    p560 = os.path.join(METRICS, "phase_m1_dose_r560.csv")
    if not os.path.exists(p560):
        raise SystemExit("run --measure first")
    d560 = pd.read_csv(p560)
    d560 = d560[d560.shot.isin(shots)][["object", "shot", "split", "repr",
                                        "AUPRO", "px_AUROC", "img_AUROC",
                                        "C_feat", "sep_auc"]]
    df = pd.concat([base[["object", "shot", "split", "repr", "AUPRO",
                          "px_AUROC", "img_AUROC", "C_feat", "sep_auc"]],
                    d560], ignore_index=True)
    df = df[df.object.isin(ALL7)]
    objs = sorted(df.object.unique())
    if len(objs) != 7:
        raise SystemExit(f"expected the 7 dose objects, found {objs}")

    print("=" * 100)
    print("PHASE M1-dose -- 448 / 560 / 672 on the fixed 7-object subset")
    print("=" * 100)
    print(f"  cells: {df.groupby(['object', 'repr']).ngroups} object-repr "
          f"groups, {len(df)} rows, shots {shots}")

    for m in ["C_feat", "sep_auc", "AUPRO", "px_AUROC", "img_AUROC"]:
        print(f"\n  {m} (mean over cells)")
        g = df.groupby(["object", "repr"])[m].mean().unstack()
        g = g[["r0", "r560", "r1"]]
        hard = g.loc[[o for o in HARD if o in g.index]]
        print(g.round(3).to_string())
        print(f"    HARD mean  r0 {hard['r0'].mean():8.3f}  "
              f"r560 {hard['r560'].mean():8.3f}  r1 {hard['r1'].mean():8.3f}")
        allm = g.mean()
        print(f"    ALL7 mean  r0 {allm['r0']:8.3f}  r560 {allm['r560']:8.3f}  "
              f"r1 {allm['r1']:8.3f}")

    print("\n" + "=" * 100)
    print("C4  monotonicity (literal: mean C_feat and mean AUPRO strictly up)")
    print("=" * 100)
    res = {}
    for grp, name in ((HARD, "HARD (6)"), (ALL7, "ALL7 (incl. control)")):
        g = df[df.object.isin(grp)].groupby("repr")[
            ["C_feat", "sep_auc", "AUPRO"]].mean().loc[["r0", "r560", "r1"]]
        print(f"\n  {name}")
        print(g.round(4).to_string())
        for m in ["C_feat", "sep_auc", "AUPRO"]:
            v = g[m].values
            mono = bool(v[0] < v[1] < v[2])
            res[(name, m)] = mono
            print(f"    {m:<8} {v[0]:9.4f} -> {v[1]:9.4f} -> {v[2]:9.4f}   "
                  f"{'MONOTONE UP' if mono else 'NOT monotone'}")

    c4 = res[("HARD (6)", "C_feat")] and res[("HARD (6)", "AUPRO")]
    print(f"\n  C4 (literal, on the preregistered hard subset): "
          f"C_feat monotone = {res[('HARD (6)', 'C_feat')]}, "
          f"AUPRO monotone = {res[('HARD (6)', 'AUPRO')]}")
    print(f"  -> C4 {'PASS' if c4 else 'FAIL'}")
    if not res[("HARD (6)", "C_feat")] and res[("HARD (6)", "sep_auc")]:
        print("  NOTE: C_feat is NOT monotone but the scale-free sep_auc IS. "
              "C_feat's magnitude depends on the distance scale, which moves "
              "with resolution even within one representation family (M1 "
              "pre-registration section 4). Report the literal verdict and "
              "this caveat together -- do not silently substitute sep_auc.")
    print("\n  C1/C2 come from phase_m1_confirm_analysis.py, C3 from "
          "phase_m1_mechanism_analysis.py. ALL of C1-C4 are required.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    a = ap.parse_args()
    shots = [int(s) for s in a.shots.split(",")]
    if a.measure or not a.analyze:
        measure(shots, a.splits)
    if a.analyze or not a.measure:
        analyze(shots)


if __name__ == "__main__":
    main()
