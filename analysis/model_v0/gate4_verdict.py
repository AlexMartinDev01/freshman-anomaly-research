# -*- coding: utf-8 -*-
"""
Gate 4 verdict: can external anomaly residuals be conditioned into the target?

Two validations, both required (Phase 3 rule: geometric proximity is necessary
but never sufficient).

GEOMETRIC -- does the transported direction set align with the target's own
    defect subspace better than an ANISOTROPY-MATCHED null?
    The null is a sham transport: identical whitening + recolouring, but the
    source residuals are built from clean-vs-clean splits instead of
    defect-vs-clean. A random subspace would be a meaningless null here --
    DINOv2 feature anisotropy makes any perturbation look aligned (Phase 3,
    Gate 2), and a sham scored HIGHER than the real defect there.
    The target's oracle defect subspace comes from within-image
    defect-minus-clean deltas, which Gate 1 showed is stable.

INTERVENTION -- does the transported set, fed to the same separation loss as
    E/F*/Q1, satisfy all three criteria at once?
        1. normal_p99 falls
        2. defect_mean does not collapse AND frac_defect_below_p99 falls
        3. collapsed classes gain on img_AUROC AND AU-PRO; healthy classes
           stay within HEALTH_TOL

Protocol: sources are always the OTHER objects (leave-one-object-out) and no
target defect patch enters any predictor. The target defect subspace is used
for diagnosis and comparison only.

Usage: python analysis/model_v0/gate4_verdict.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
from subspace_gates import (defect_deltas, normal_patches, top_subspace,  # noqa: E402
                            overlap, load_cache, split_train)

CSV = r"E:\work\freshman\results\model_v0\metrics\oracle_ablation.csv"
COLLAPSED = ["can", "wallplugs"]
HEALTHY = ["vial", "sheet_metal"]
DEFECT_FLOOR = 0.80
HEALTH_TOL = 2.0
K = 25


def source_normal_residuals(src, seed=0):
    """Sham source residuals: clean-vs-clean split, same construction."""
    tr, te = load_cache(src)
    gt = te["gt_frac"].reshape(len(te["types"]), -1)
    bad = np.where(te["types"] == "bad")[0]
    perm = np.random.default_rng(seed).permutation(len(bad))
    sup = bad[perm[:len(bad) // 2]]
    rng = np.random.default_rng(seed + 13)
    out = []
    for i in sup:
        X = te["feats"][i].astype(np.float32)
        idx = np.where(gt[i] == 0.0)[0]
        if len(idx) < 4:
            continue
        p = rng.permutation(len(idx))
        a, b = idx[p[:len(idx) // 2]], idx[p[len(idx) // 2:]]
        out.append(X[a].mean(axis=0) - X[b].mean(axis=0))
    return np.stack(out) if out else np.zeros((0, 384), np.float32)


def geom():
    print("=" * 96)
    print("GEOMETRIC -- transported directions vs the target's own defect "
          "subspace")
    print("=" * 96)
    print(f"\n  rank k={K}; random floor = k/384 = {K/384:.3f}")
    print(f"{'target':<12}{'source set':<16}{'overlap(trans, D_t)':>20}"
          f"{'overlap(sham, D_t)':>20}{'random':>9}{'beats sham?':>13}")
    from proxy_defects import (_normal_cov, _source_residuals, _sqrtm,
                               OBJECTS)
    for obj in OBJECTS:
        others = [o for o in OBJECTS if o != obj]
        D = defect_deltas(obj)
        k = min(K, (len(D) - 1) // 2)
        Ud = top_subspace(D, k)
        for tag, mode in (("X0 raw", "raw"), ("X4 whitened", "whiten"),
                          ("X5 recoloured", "whiten_rec")):
            R = np.concatenate([_source_residuals(s) for s in others])
            S = np.concatenate([source_normal_residuals(s) for s in others])
            if mode != "raw":
                Rw, Sw = [], []
                for s in others:
                    W = _sqrtm(_normal_cov(s), inv=True)
                    Rw.append(_source_residuals(s) @ W)
                    Sw.append(source_normal_residuals(s) @ W)
                R, S = np.concatenate(Rw), np.concatenate(Sw)
            if mode == "whiten_rec":
                Rt = _sqrtm(_normal_cov(obj))
                R, S = R @ Rt, S @ Rt
            kk = min(k, (len(R) - 1) // 2, (len(S) - 1) // 2)
            ot = overlap(top_subspace(R, kk), Ud)
            os_ = overlap(top_subspace(S, kk), Ud)
            print(f"{obj:<12}{tag:<16}{ot:>20.3f}{os_:>20.3f}{kk/384:>9.3f}"
                  f"{('YES' if ot > os_ else 'no'):>13}")
    print("\n  The sham shares every step except the anomaly content, so a "
          "candidate that")
    print("  does not beat it is transporting feature geometry, not anomaly "
          "structure.")


def intervention(df):
    print("\n" + "=" * 96)
    print("INTERVENTION -- same separation loss as E; all three must hold")
    print("=" * 96)
    base = df[df.config == "A_half"].set_index("object")
    for cfg in ["X0", "X4", "X5", "X6"]:
        sub = df[df.config == cfg].set_index("object")
        if sub.empty:
            continue
        print(f"\n--- {cfg} ---")
        c1, c2, c3 = [], [], []
        for o in base.index:
            if o not in sub.index:
                continue
            b, r = base.loc[o], sub.loc[o]
            tail = r.normal_p99 < b.normal_p99
            pres = (r.defect_mean >= DEFECT_FLOOR * b.defect_mean
                    and r.frac_defect_below_p99 < b.frac_defect_below_p99)
            if o in COLLAPSED:
                det = (r.img_AUROC > b.img_AUROC
                       and r["AUPRO@0.05"] > b["AUPRO@0.05"])
            else:
                det = r.img_AUROC >= b.img_AUROC - HEALTH_TOL
            c1.append(tail); c2.append(pres); c3.append(det)
            print(f"  {o:<12} tail={str(tail):<6} preserve={str(pres):<6} "
                  f"detect={str(det):<6}  (p99 {b.normal_p99:.3f}->"
                  f"{r.normal_p99:.3f}, defect {b.defect_mean:.3f}->"
                  f"{r.defect_mean:.3f}, img {b.img_AUROC:.1f}->"
                  f"{r.img_AUROC:.1f}, AUPRO {b['AUPRO@0.05']:.1f}->"
                  f"{r['AUPRO@0.05']:.1f})")
        ok = (sum(c1) == len(c1) and sum(c2) == len(c2) and sum(c3) == len(c3))
        print(f"  => tail {sum(c1)}/{len(c1)}  preserve {sum(c2)}/{len(c2)}  "
              f"detect {sum(c3)}/{len(c3)}   "
              f"{'PASS' if ok else 'FAIL'}")


def main():
    df = pd.read_csv(CSV)
    if not any(c in set(df.config) for c in ("X0", "X4", "X5", "X6")):
        sys.exit("no Gate 4 configs found - run run_oracle.py --configs "
                 "X0 X4 X5 X6 --eval-half first")
    geom()
    intervention(df)
    print("\n" + "=" * 96)
    print("FORK")
    print("=" * 96)
    print("  X5 passes  -> target-conditioned transfer works; build the model "
          "around it.")
    print("  X5 fails   -> stop hand-built geometry; move to a LEARNED")
    print("                target-conditioned prior, f(external anomaly "
          "repr, target normal repr).")


if __name__ == "__main__":
    main()
