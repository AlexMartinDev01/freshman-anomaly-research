# -*- coding: utf-8 -*-
"""Frozen R5-A/B gate, re-implemented, runnable on MAIN and on the replication.

Exists for two reasons:
  1. r5_analyze_main.py crashes on the replication because R5-C writes an EMPTY
     csv there (the replication is split=2 and R5-C is hard-gated to split==0),
     and the script guards with .exists() rather than reading defensively.
  2. The replication and MAIN must be scored by ONE implementation, or a
     difference could come from the scorer instead of the data.

The A/B verdict formulas are copied verbatim from r5_analyze_main.py.  Nothing
here changes a gate.  Run it on MAIN first: if it does not reproduce the kit's
MAIN report exactly, the replication numbers are not trustworthy either.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def read_if_any(p):
    if not p.exists() or p.stat().st_size < 3:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def gates(d, sel):
    dsmap = sel[["object", "dataset"]].drop_duplicates()
    sa = read_if_any(d / "R5A_strength_per_image.csv")
    ex = read_if_any(d / "R5B_extent_normal_background.csv")

    ar = sa[sa.dose == -1].copy()
    aobj = ar.groupby("object").agg(
        n_bad=("image", "count"), rho_top1=("top1", "mean"),
        rho_max=("max", "mean"), rho_q=("q", "mean")).reset_index() \
        .merge(dsmap, on="object", how="left")
    n = len(aobj)
    nq = int((aobj.rho_q < 0).sum())
    nt = int((aobj.rho_top1 >= 0).sum())
    pa = binomtest(nq, n, .5, alternative="greater").pvalue
    pt = binomtest(nt, n, .5, alternative="greater").pvalue
    diff = aobj.rho_top1 - aobj.rho_q
    pw = wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0
    ads = aobj.groupby("dataset").agg(
        n=("object", "count"), mean_rho_top1=("rho_top1", "mean"),
        mean_rho_q=("rho_q", "mean"),
        frac_q_negative=("rho_q", lambda x: (x < 0).mean())).reset_index()

    br = ex[ex.alpha == -1].copy()
    bobj = br.groupby("object").agg(
        n_good=("image", "count"), rho_top1=("top1", "mean"),
        rho_max=("max", "mean"), rho_q=("q", "mean"),
        frac_image_q_negative=("q", lambda x: (x < 0).mean()),
        frac_image_top1_nonnegative=("top1", lambda x: (x >= 0).mean())).reset_index() \
        .merge(dsmap, on="object", how="left")
    nb = len(bobj)
    nbq = int((bobj.rho_q < 0).sum())
    pb = binomtest(nbq, nb, .5, alternative="greater").pvalue
    all_image_qneg = float((br.q < 0).mean())
    all_image_topok = float((br.top1 >= 0).mean())
    bds = bobj.groupby("dataset").agg(
        n=("object", "count"), mean_rho_q=("rho_q", "mean"),
        frac_q_negative=("rho_q", lambda x: (x < 0).mean())).reset_index()

    # B3 from the frozen protocol (not printed by the kit's script)
    piv = ex[ex.alpha > 0].pivot_table(index=["object", "image"],
                                       columns="alpha", values="max")
    b3_spread = float((piv.max(axis=1) - piv.min(axis=1)).max())

    r = dict(
        A_n=n, A_nq=nq, A_nt=nt, A_pa=pa, A_pt=pt, A_pw=pw,
        A1=bool(nt / n >= .70),
        A2=bool((nq / n >= .70) and (pa < .05)),
        A3=bool(pw < .05),
        A4=bool((ads.mean_rho_q < 0).all()),
        B_n=nb, B_nq=nbq, B_pb=pb,
        B_img_qneg=all_image_qneg, B_img_topok=all_image_topok, B3_max_spread=b3_spread,
        B1=bool(all_image_topok >= .999999),
        B2=bool((all_image_qneg >= .80) and (nbq == nb) and (pb < .05)),
    )
    r["A_ALL"], r["B_ALL"] = r["A1"] and r["A2"] and r["A3"] and r["A4"], r["B1"] and r["B2"]
    return r, ads, bds, aobj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--selection", required=True)
    a = ap.parse_args()
    d = Path(a.indir)
    r, ads, bds, aobj = gates(d, pd.read_csv(a.selection))
    print(f"--- {a.indir} ---")
    print(f"A: objects {r['A_n']} | rho_top1>=0 {r['A_nt']}/{r['A_n']} p={r['A_pt']:.3g}"
          f" | rho_q<0 {r['A_nq']}/{r['A_n']} p={r['A_pa']:.3g} | paired p={r['A_pw']:.3g}")
    print(f"   A1={r['A1']} A2={r['A2']} A3={r['A3']} A4={r['A4']}  -> A_ALL={r['A_ALL']}")
    print(ads.round(3).to_string(index=False))
    print(f"B: objects {r['B_n']} | rho_q<0 {r['B_nq']}/{r['B_n']} p={r['B_pb']:.3g}"
          f" | images q<0 {r['B_img_qneg']:.4f} | images top1>=0 {r['B_img_topok']:.4f}"
          f" | B3 max spread {r['B3_max_spread']:.3e}")
    print(f"   B1={r['B1']} B2={r['B2']} B3={r['B3_max_spread'] <= 1e-9} -> B_ALL={r['B_ALL']}")
    print(bds.round(3).to_string(index=False))
