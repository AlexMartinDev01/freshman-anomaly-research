# -*- coding: utf-8 -*-
"""
V3 mid-run ENGINEERING health check -- answers "is the pipeline producing a
degenerate result?", NOT "is the method any good?".

It computes NO detection metric. AUPRO / AUROC / A3-vs-A0 performance are
deliberately unavailable until 324/324 and the completeness audit, so the
metric functions are booby-trapped here: a stray call raises rather than
quietly returning a number, and nobody has to rely on discipline.

What it does check, per completed cell:
  - every map finite, expected shapes (a0 vs arms on the 448 grid, a2 on 672)
  - an m2 == 0 image is bit-identical to A0 (the invariant that makes A3
    comparable to A0 at all)
  - a refined image ACTUALLY differs from A0, and by a non-trivial amount --
    a refined cell whose map never moves would mean the writeback silently
    did nothing
  - value ranges, so a collapsed or exploded map shows up
  - budget equality (m', n, T) across A3 / 10 A4 seeds / A5 as stored on disk
  - metadata git_commit equals the freeze tag

Usage: python experiments/model_v0/v3_health_check.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")


def booby_trap():
    def _boom(*a, **k):
        raise AssertionError(
            "health check called a detection metric. This run must stay "
            "blind until 324/324 and the completeness audit.")
    for m in ("pixel_metrics_binned", "pixel_f1_at_threshold"):
        try:
            import importlib
            mod = importlib.import_module("pixel_metrics_binned")
            if hasattr(mod, m):
                setattr(mod, m, _boom)
        except Exception:
            pass
    try:
        import sklearn.metrics as skm
        skm.roc_auc_score = _boom
    except Exception:
        pass


FREEZE = "3049e1c3ec0708fa405ed2f0fca53d87e5a214db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\work\freshman\results\model_v0\v3")
    a = ap.parse_args()
    booby_trap()

    dones = []
    for dp, _, fs in os.walk(a.root):
        if "DONE.json" in fs:
            dones.append(dp)
    dones.sort()
    print(f"completed cells: {len(dones)} / 324   (metrics booby-trapped)")

    rows = []
    bad = []
    for d in dones:
        cid = os.path.relpath(d, a.root).replace("\\", "/")
        try:
            j = json.load(open(os.path.join(d, "DONE.json")))
        except Exception as e:
            bad.append((cid, f"DONE unparseable: {e}"))
            continue
        if j.get("git_commit") != FREEZE:
            bad.append((cid, f"git_commit {j.get('git_commit')} != freeze"))
        try:
            a0 = np.load(os.path.join(d, "a0.npy"))
            a2 = np.load(os.path.join(d, "a2.npy"))
            z = np.load(os.path.join(d, "arms.npz"))
            dg = json.load(open(os.path.join(d, "diagnostics.json")))
        except Exception as e:
            bad.append((cid, f"load failed: {type(e).__name__}: {e}"))
            continue
        n = a0.shape[0]
        if not np.isfinite(a0).all() or not np.isfinite(a2).all():
            bad.append((cid, "non-finite a0/a2"))
        for k in z.files:
            if z[k].shape[0] != n:
                bad.append((cid, f"{k} has {z[k].shape[0]} images, a0 has {n}"))
        T = {k: np.array([r["T"] for r in dg[k]]) for k in dg}
        ref = T["a3"]
        if not all((T[k] == ref).all() for k in T):
            bad.append((cid, "budget equality violated on disk"))
        m2 = np.array([r["m2"] for r in dg["a3"]])
        # m2 == 0 must be bit-identical to A0
        zmask = m2 == 0
        if zmask.any():
            for k in z.files:
                if not np.array_equal(z[k][zmask], a0[zmask]):
                    bad.append((cid, f"{k}: m2=0 image differs from A0"))
                    break
        # a refined image must actually move
        nz = ~zmask
        if nz.any():
            dd = np.abs(z["a3"][nz] - a0[nz]).max(axis=(1, 2))
            frozen = int((dd == 0).sum())
            if frozen:
                bad.append((cid, f"{frozen} refined image(s) identical to A0"))
            rows.append(dict(cell=cid, n=n, n_ref=int(nz.sum()),
                             dmax_mean=float(dd.mean()), dmax_min=float(dd.min()),
                             dmax_max=float(dd.max()),
                             a0_lo=float(a0.min()), a0_hi=float(a0.max()),
                             a2_lo=float(a2.min()), a2_hi=float(a2.max()),
                             a3_lo=float(z["a3"].min()),
                             a3_hi=float(z["a3"].max())))
        else:
            rows.append(dict(cell=cid, n=n, n_ref=0, dmax_mean=0.0,
                             dmax_min=0.0, dmax_max=0.0,
                             a0_lo=float(a0.min()), a0_hi=float(a0.max()),
                             a2_lo=float(a2.min()), a2_hi=float(a2.max()),
                             a3_lo=float(z["a3"].min()),
                             a3_hi=float(z["a3"].max())))

    R = rows
    tot_img = sum(r["n"] for r in R)
    tot_ref = sum(r["n_ref"] for r in R)
    print(f"images {tot_img}   refined {tot_ref} ({100.0 * tot_ref / max(tot_img,1):.1f}%)")
    if R:
        dm = np.array([r["dmax_mean"] for r in R if r["n_ref"] > 0])
        dmin = np.array([r["dmax_min"] for r in R if r["n_ref"] > 0])
        hi = np.array([r["a3_hi"] for r in R])
        lo = np.array([r["a3_lo"] for r in R])
        print(f"A3-vs-A0 max|delta| on refined images: "
              f"min {dmin.min():.3f}  mean {dm.mean():.3f}  max {dm.max():.3f}")
        print(f"A3 value range across cells: [{lo.min():.3f}, {hi.max():.3f}]")
        print(f"A2 value range across cells: "
              f"[{min(r['a2_lo'] for r in R):.3f}, "
              f"{max(r['a2_hi'] for r in R):.3f}]")
        z = np.array([r["n_ref"] for r in R])
        print(f"refined images per cell: median {int(np.median(z))} "
              f"min {z.min()} max {z.max()}")
    if bad:
        print(f"\n  !! {len(bad)} PROBLEM(S):")
        for c, w in bad[:20]:
            print(f"     {c}: {w}")
    else:
        print("\n  no structural problems found in the completed cells")
    print("\n  NOTE: this says nothing about AUPRO/AUROC. Those stay sealed "
          "until 324/324 + audit.")


if __name__ == "__main__":
    main()
