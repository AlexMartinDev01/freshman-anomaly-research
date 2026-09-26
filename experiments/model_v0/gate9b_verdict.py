# -*- coding: utf-8 -*-
"""
Gate 9B-v0 verdict -- applies the PRE-REGISTERED criteria.

Fixed before the run:
    G3 interaction > G1 bank_only            <- THE decisive test
    G3 interaction > raw external (G0)
    majority of held-out categories > normal-only baseline
    >= 2/3 of eligible categories positive with meaningful mean closure
    Oracle Gap Closure = (G - B) / (O - B), only where O > B

G3 > G1 is the one that decides whether the generator learned target
conditioning or merely "perturb some donor defects" -- which is all Gate 9A's
bank_only could do. Beating baseline on AUROC alone does NOT count, because G2
(which consumes NO external anomaly at all) can also move the baseline.

The extra diagnostic G3 - G2 is reported for the same reason: if a model that
never sees an external defect matches the full model, the external content is
contributing nothing regardless of where the AUROC lands.

Usage: python experiments/model_v0/gate9b_verdict.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
import argparse
VARIANTS = ["identity", "bank_only", "target_only", "interaction"]
NAME = {"identity": "G0 identity", "bank_only": "G1 bank_only",
        "target_only": "G2 target_only", "interaction": "G3 interaction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gate9b")
    ap.add_argument("--n-min", type=int, default=10,
                    help="categories that must be better for the G3>G1 test")
    args = ap.parse_args()   # not `a`: the pair loop below rebinds `a`
    raw = pd.read_csv(os.path.join(METRICS, f"{args.tag}_eval.csv"))
    # Each row carries only its OWN variant's column, so pairwise differences
    # need the four variants pivoted onto a shared (heldout, shot, rep) index.
    parts = []
    for v in VARIANTS:
        s = raw[raw.variant == v][["heldout", "shot", "rep", "baseline",
                                   "oracle", f"gen|{v}", f"raw|{v}"]].copy()
        s = s.rename(columns={f"gen|{v}": "G", f"raw|{v}": "RAW"})
        s["variant"] = v
        parts.append(s)
    P = pd.concat(parts)
    d = P.pivot_table(index=["heldout", "shot", "rep"], columns="variant",
                      values="G").reset_index()
    d = d.merge(P.groupby(["heldout", "shot", "rep"])[["baseline", "oracle"]]
                .first().reset_index(), on=["heldout", "shot", "rep"])
    d["G0"] = d["identity"]

    T = pd.DataFrame({NAME[v]: d.groupby("heldout")[v].mean()
                      for v in VARIANTS})
    T["baseline"] = d.groupby("heldout").baseline.mean()
    T["oracle"] = d.groupby("heldout").oracle.mean()
    T = T[["baseline"] + [NAME[v] for v in VARIANTS] + ["oracle"]]
    print("=" * 100)
    print("per held-out category (mean over 14 sources x 3 shots x 3 reps)")
    print("=" * 100)
    print(T.round(1).to_string())

    print("\n" + "=" * 100)
    print("PAIRED DIFFERENCES (per category, then t over the 15 categories)")
    print("=" * 100)
    print(f"  {'comparison':<24}{'mean':>8}{'se':>7}{'t':>7}{'cats better':>14}")
    stats = {}
    for va, vb in [("interaction", "bank_only"), ("interaction", "identity"),
                 ("interaction", "target_only"), ("interaction", None),
                 ("target_only", "bank_only"), ("bank_only", "identity")]:
        diff = d[va] - (d["baseline"] if vb is None else d[vb])
        pc = diff.groupby(d.heldout).mean()
        se = pc.std() / np.sqrt(len(pc))
        t = pc.mean() / se if se > 0 else np.nan
        lbl = f"{va[:4]} - " + ("baseline" if vb is None else vb[:4])
        print(f"  {lbl:<24}{pc.mean():>+8.2f}{se:>7.2f}{t:>7.2f}"
              f"{int((pc > 0).sum()):>10}/{len(pc)}")
        stats[lbl] = (pc, t)

    print("\n" + "=" * 100)
    print("ORACLE GAP CLOSURE  (G - B) / (O - B), eligible = oracle > baseline")
    print("=" * 100)
    elig = T[T.oracle > T.baseline]
    print(f"  eligible: {len(elig)}/{len(T)} categories")
    clos = {}
    for v in VARIANTS:
        c = (elig[NAME[v]] - elig.baseline) / (elig.oracle - elig.baseline) * 100
        clos[v] = c
        print(f"    {NAME[v]:<15} closure {c.mean():+7.1f}%   "
              f"({int((c > 0).sum())}/{len(c)} positive)")

    print("\n" + "=" * 100)
    print("VERDICT (pre-registered)")
    print("=" * 100)
    pc31, t31 = stats["inte - bank"]
    pc30, _ = stats["inte - iden"]
    pc3b, _ = stats["inte - baseline"]
    pc32, _ = stats["inte - targ"]
    c3 = clos["interaction"]
    ok = {
        "G3 > G1 bank_only (mean>0, t>1.5, >=2/3 cats)":
            pc31.mean() > 0 and t31 > 1.5 and int((pc31 > 0).sum()) >= args.n_min,
        "G3 > raw external G0 (mean>0, >=2/3 cats)":
            pc30.mean() > 0 and int((pc30 > 0).sum()) >= args.n_min,
        "G3 > baseline on majority of categories": int((pc3b > 0).sum()) > len(pc3b) / 2,
        "closure > 0 on >=2/3 eligible": int((c3 > 0).sum()) >= len(c3) * 2 / 3,
    }
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  diagnostic G3 - G2 (external content vs no external content): "
          f"{pc32.mean():+.2f}, {int((pc32 > 0).sum())}/{len(pc32)} cats")
    passed = (ok["G3 > G1 bank_only (mean>0, t>1.5, >=2/3 cats)"]
              and ok["G3 > raw external G0 (mean>0, >=2/3 cats)"])
    print("\n  " + ("PASS -> Gate 9B-v1" if passed else
                    "FAIL -> do NOT run AD2; the generator did not learn target "
                    "conditioning"))


if __name__ == "__main__":
    main()
