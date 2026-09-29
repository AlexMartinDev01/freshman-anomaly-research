# -*- coding: utf-8 -*-
"""Stage 5D-A -- adaptive-halo budget audit.  FULLY OFFLINE.

Stage 5C showed that no FIXED halo size n <= 33 satisfies both
`agg3 Spearman >= 0.50` and `trigger-mass retention >= 0.50`.  But the frozen
V3 does not use a fixed n.  Addendum 1 allocates per image:

    m' = min(m, K_max),  K_max = floor(B / 25)
    n  = the largest odd n in [5, 33] with  m' * n^2 <= B

so an image with few triggers gets a large halo and an image with many gets a
small one.  A fixed-n Pareto curve is therefore NOT the adaptive Pareto curve,
and the fixed-n STOP does not by itself close the adaptive scheme.

This script answers the open question with no model and no GPU: over all 324
cells and every test image, how much TRIGGER MASS actually lands on each n?

It reads only frozen artefacts -- diagnostics.json, the 672 grids -- and
reproduces the addendum-1 rule, asserting it against the `m2`/`n` that the
frozen runner already recorded, so the audit cannot silently diverge from the
implementation it audits.

NO DINO, no defect labels, no GT, no AUPRO/AUROC, no selector change.

FIDELITY PROXY CAVEAT (read before quoting the number):
    `05d_adaptive_fidelity_proxy.csv` weights the Stage 5C per-category
    agg3 Spearman by the real adaptive n distribution.  That is a SCREENING
    PROXY, not a measured mixed-n Spearman: (a) n is not random -- images that
    get large n are exactly the low-m images, so the weighting assumes fidelity
    depends mainly on n; (b) Stage 5C measured fidelity at k=8/split=0 support
    only, while the allocation here spans all shots and splits.  It is used to
    decide whether a real mixed-n measurement (Stage 5D-B) is worth running --
    never as a result.

Usage: OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5d_adaptive_audit.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "v3_root_cause_diagnosis")
N_GRID = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
CORE = 5
ARMS = ["a3", "a5"] + [f"a4s{j}" for j in range(10)]

# ---- Stage 5D-B development gate.  Fixed BEFORE 5D-B is run, and deliberately
# stricter on retention than the fixed-halo rule: adaptivity's entire point is
# not to throw positions away, so if it cannot keep 80% of trigger mass it has
# no advantage over fixed n.  NOT part of the original model preregistration.
GATE_RHO = 0.50
GATE_RETENTION = 0.80

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def k_max(B):
    return int(B // (CORE * CORE))


def allocate(m, B):
    """The frozen addendum-1 rule, reproduced exactly."""
    mp = min(int(m), k_max(B))
    if mp == 0:
        return 0, 0
    n = 0
    for cand in N_GRID:                 # ascending -> last hit is the largest
        if mp * cand * cand <= B:
            n = cand
    return mp, n


def collect():
    rows, mismatches, arm_mismatches, b_mismatches = [], [], [], []
    budget_derivation = []
    for ds, objs in (("mvtec", MVTEC), ("visa", VISA_CATS)):
        for obj in objs:
            for shot in (1, 2, 4, 8):
                for sp in range(3):
                    d = os.path.join(RESULTS, "v3", ds, obj, f"k{shot}", f"s{sp}")
                    fp = os.path.join(d, "diagnostics.json")
                    if not os.path.exists(fp):
                        continue
                    with open(fp) as f:
                        dg = json.load(f)
                    with np.load(os.path.join(
                            RESULTS, "cache_m1_dino672", "final",
                            f"dino672_{obj}_test.npz"), allow_pickle=True) as z:
                        u = {tuple(int(v) for v in r) for r in z["grids"]}
                    gh6, gw6 = u.pop()
                    B_cell = 0.5 * gh6 * gw6
                    for i, rec in enumerate(dg["a3"]):
                        m, B = int(rec["m"]), float(rec["B"])
                        # The frozen runner derives the 672 grid instead of
                        # reading it (v3_run.py:100):  B = 0.5*ceil(1.5*gh448)
                        # *ceil(1.5*gw448).  Reproduce THAT, or the audit does
                        # not audit the implementation that actually ran.
                        gh4, gw4 = int(rec["shape"][0]), int(rec["shape"][1])
                        B_frozen = 0.5 * int(np.ceil(gh4 * 1.5)) * int(np.ceil(gw4 * 1.5))
                        if abs(B_frozen - B) > 1e-9:
                            b_mismatches.append((ds, obj, shot, sp, i, B, B_frozen))
                        if abs(B - B_cell) > 1e-9:
                            budget_derivation.append(dict(
                                dataset=ds, object=obj, shot=shot, split=sp,
                                gh448=gh4, gw448=gw4, gh672=gh6, gw672=gw6,
                                B_frozen=B, B_true=B_cell, ratio=B / B_cell))
                        mp, n = allocate(m, B)
                        if mp != int(rec["m2"]) or n != int(rec["n"]):
                            mismatches.append((ds, obj, shot, sp, rec["image"],
                                               m, rec["m2"], rec["n"], mp, n))
                        # what the SAME rule would pick if B were read from the
                        # real 672 grid -- sensitivity, not the as-run value
                        mp_t, n_t = allocate(m, B_cell)
                        if mp_t * n_t * n_t > B_cell + 1e-9:
                            raise AssertionError("true-B allocation overspends")
                        overrun = mp * n * n > B_cell + 1e-9
                        # the allocation is position-only, so every arm must agree
                        for arm in ARMS:
                            r2 = dg.get(arm, [])[i] if i < len(dg.get(arm, [])) else None
                            if r2 is None:
                                continue
                            if int(r2["m2"]) != mp or int(r2["n"]) != n:
                                arm_mismatches.append((ds, obj, shot, sp, arm,
                                                       rec["image"], r2["m2"], r2["n"], mp, n))
                        rows.append(dict(
                            dataset=ds, object=obj, shot=shot, split=sp,
                            image_id=i, image=rec["image"],
                            gh672=gh6, gw672=gw6, B=B,
                            m=m, m_prime=mp,
                            n_dynamic=n, n_frozen=int(rec["n"]), n_match=bool(n == int(rec["n"])),
                            n_under_true_B=n_t, n_shifted=bool(n != n_t),
                            B_true=B_cell, budget_overrun_true=bool(overrun),
                            cap_hit=int(rec.get("cap_hit", 0)),
                            tokens_used=mp * n * n,
                            retained_trigger_mass=(mp / m) if m else np.nan))
    return (pd.DataFrame(rows), mismatches, arm_mismatches, b_mismatches,
            pd.DataFrame(budget_derivation))


def distribution(df):
    """n distribution, image-weighted and trigger-mass-weighted.

    Trigger mass is the primary weighting: an image with 30 triggers that gets
    n=5 costs the budget 30 windows, while an image with 1 trigger that gets
    n=33 costs 1.  Counting images would make those look equally important.
    """
    trg = df[df.m > 0]
    out = []
    for ds in ["mvtec", "visa", "ALL"]:
        sub = trg if ds == "ALL" else trg[trg.dataset == ds]
        if not len(sub):
            continue
        tot_img, tot_mass = len(sub), float(sub.m.sum())
        ns = sorted(int(v) for v in sub.n_dynamic.unique())
        mass = np.array([float(sub[sub.n_dynamic == n].m.sum()) for n in ns])
        imgs = np.array([len(sub[sub.n_dynamic == n]) for n in ns])
        for k, n in enumerate(ns):     # cumulative sums done explicitly: the
            s = sub[sub.n_dynamic == n]  # groupby.transform idiom is not worth
            out.append(dict(             # the pandas-version risk here
                dataset=ds, n_dynamic=n, n_images=int(imgs[k]),
                image_share=imgs[k] / tot_img,
                trigger_mass=mass[k], mass_share=mass[k] / tot_mass,
                mean_m_per_image=float(s.m.mean()),
                mean_tokens_used=float(s.tokens_used.mean()),
                share_mass_n_ge=float(mass[k:].sum() / tot_mass),
                share_mass_n_le=float(mass[:k + 1].sum() / tot_mass),
                share_img_n_ge=float(imgs[k:].sum() / tot_img),
                share_img_n_le=float(imgs[:k + 1].sum() / tot_img)))
    return pd.DataFrame(out)


def summarise(df, by):
    """Per-group allocation summary.  Named aggregations only -- no
    groupby.apply, which pandas 3.0 mishandles on some frames and which already
    cost this project one full sweep."""
    t = df[df.m > 0].copy()
    t["w_n"] = t.n_dynamic * t.m
    t["m_le15"] = np.where(t.n_dynamic <= 15, t.m, 0)
    t["m_ge23"] = np.where(t.n_dynamic >= 23, t.m, 0)
    t["m_eq_max"] = np.where(t.n_dynamic == max(N_GRID), t.m, 0)
    out = t.groupby(by).agg(
        n_images=("m", "size"),
        total_trigger_mass=("m", "sum"),
        retained_trigger_mass=("m_prime", "sum"),
        w_n=("w_n", "sum"),
        m_le15=("m_le15", "sum"),
        m_ge23=("m_ge23", "sum"),
        m_eq_max=("m_eq_max", "sum"),
        image_weighted_mean_n=("n_dynamic", "mean"),
        n_distinct_used=("n_dynamic", "nunique"),
    ).reset_index()
    for src, dst in (("retained_trigger_mass", "retention"), ("w_n", "mass_weighted_mean_n"),
                     ("m_le15", "P_mass_n_le_15"), ("m_ge23", "P_mass_n_ge_23"),
                     ("m_eq_max", "P_mass_n_eq_max")):
        out[dst] = out[src] / out.total_trigger_mass
    return out.drop(columns=["w_n", "m_le15", "m_ge23", "m_eq_max"])


def fidelity_proxy(df, dist):
    """Screen the adaptive allocation against the Stage 5C agg3 curve.

        rho_proxy(c) = sum_n  w_c(n) * rho_c(n)

    where w_c(n) is category c's TRIGGER-MASS share at n from the real
    allocation, and rho_c(n) is that category's Stage 5C agg3 Spearman at n.
    Long format so the arithmetic is auditable.  SCREENING PROXY -- see the
    module docstring for why it is not a measured mixed-n Spearman.
    """
    f5c = os.path.join(OUT, "05c_agg3_fidelity.csv")
    if not os.path.exists(f5c):
        raise SystemExit(f"missing {f5c}; run Stage 5C first.")
    rho = pd.read_csv(f5c).rename(
        columns={"n_ctx": "n_dynamic", "agg3_spearman": "rho_c_n"})[
        ["dataset", "object", "n_dynamic", "rho_c_n"]]
    trg = df[df.m > 0]
    rows = []
    for (ds, obj), s in trg.groupby(["dataset", "object"]):
        tot = float(s.m.sum())
        w = s.groupby("n_dynamic").m.sum() / tot
        imgw = s.groupby("n_dynamic").size() / len(s)
        r = rho[(rho.dataset == ds) & (rho.object == obj)].set_index("n_dynamic")
        for n in sorted(w.index):
            if n not in r.index:
                raise SystemExit(f"{ds}/{obj}: n={n} has no Stage 5C fidelity row")
            rows.append(dict(dataset=ds, object=obj, n_dynamic=int(n),
                             mass_share=float(w[n]),
                             image_share=float(imgw.get(n, 0.0)),
                             rho_c_n=float(r.loc[n, "rho_c_n"]),
                             contribution=float(w[n] * r.loc[n, "rho_c_n"])))
    p = pd.DataFrame(rows)
    tot = p.groupby(["dataset", "object"]).contribution.sum().rename("rho_proxy")
    img = (p.image_share * p.rho_c_n).groupby([p.dataset, p.object]).sum().rename(
        "rho_proxy_image_weighted")
    p = p.merge(tot, on=["dataset", "object"]).merge(img, on=["dataset", "object"])
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    df, mismatch, arm_mis, b_mis, bderiv = collect()

    print("=== correctness: reproduced allocation vs the frozen records ===")
    print(f"  images audited                     : {len(df)}")
    print(f"  cells covered                      : "
          f"{df[['dataset','object','shot','split']].drop_duplicates().shape[0]} / 324")
    print(f"  (m2, n) mismatches vs frozen a3    : {len(mismatch)}")
    print(f"  arm-to-arm (m2, n) mismatches      : {len(arm_mis)}")
    print(f"  B != 0.5*ceil(1.5*gh448)*ceil(1.5*gw448) : {len(b_mis)}   "
          f"(must be 0 -- this reproduces the frozen formula)")
    print(f"  B != 0.5*gh672*gw672 (true grid)    : "
          f"{df.budget_overrun_true.notna().sum() and int((~np.isclose(df.B, df.B_true)).sum())}"
          f" / {len(df)} images, in {bderiv[['dataset','object']].drop_duplicates().shape[0]} objects")
    print(f"  images whose as-run allocation exceeds the TRUE budget: "
          f"{int(df.budget_overrun_true.sum())} / {len(df)}")
    print(f"  images whose n would change under the true budget    : "
          f"{int(df.n_shifted.sum())} / {len(df)}")
    for x in mismatch[:5]:
        print("     MISS", x)
    if b_mis:
        raise SystemExit("AUDIT DIVERGED -- it does not reproduce the frozen "
                         "B formula; fix before reading any distribution.")
    if mismatch or arm_mis:
        raise SystemExit("AUDIT DIVERGED from the frozen implementation -- fix "
                         "before reading any distribution below.")

    df.to_csv(os.path.join(OUT, "05d_adaptive_per_image.csv"), index=False)
    dist = distribution(df)
    dist.to_csv(os.path.join(OUT, "05d_n_distribution_by_trigger_mass.csv"), index=False)
    summarise(df, ["dataset", "object"]).to_csv(
        os.path.join(OUT, "05d_adaptive_by_category.csv"), index=False)
    summarise(df, ["dataset"]).to_csv(
        os.path.join(OUT, "05d_adaptive_by_dataset.csv"), index=False)
    bderiv.to_csv(os.path.join(OUT, "05d_budget_derivation_audit.csv"), index=False)
    prox = fidelity_proxy(df, dist)
    prox.to_csv(os.path.join(OUT, "05d_adaptive_fidelity_proxy.csv"), index=False)

    print("\n=== n distribution, TRIGGER-MASS weighted (primary) ===")
    a = dist[dist.dataset == "ALL"][["n_dynamic", "n_images", "image_share",
                                     "trigger_mass", "mass_share",
                                     "mean_m_per_image"]]
    print(a.round(4).to_string(index=False))

    print("\n=== the four headline numbers ===")
    for ds in ["mvtec", "visa", "ALL"]:
        s = df[df.m > 0] if ds == "ALL" else df[(df.dataset == ds) & (df.m > 0)]
        px = prox if ds == "ALL" else prox[prox.dataset == ds]
        med = float(px.groupby(["dataset", "object"]).rho_proxy.first().median())
        print(f"  {ds:6s}  P_mass(n<=15)={float(s.loc[s.n_dynamic<=15,'m'].sum()/s.m.sum()):.3f}"
              f"   P_mass(n>=23)={float(s.loc[s.n_dynamic>=23,'m'].sum()/s.m.sum()):.3f}"
              f"   retention={float(s.m_prime.sum()/s.m.sum()):.3f}"
              f"   mean_n(mass-wtd)={float((s.n_dynamic*s.m).sum()/s.m.sum()):.2f}"
              f"   rho_proxy(median)={med:.3f}")

    print("\n=== Stage 5D-B gate (fixed before 5D-B) ===")
    print(f"  need median_category(rho_adaptive_agg3) >= {GATE_RHO:.2f} "
          f"AND trigger-mass retention >= {GATE_RETENTION:.2f}")
    s = df[df.m > 0]
    ret = float(s.m_prime.sum() / s.m.sum())
    med = float(prox.groupby(["dataset", "object"]).rho_proxy.first().median())
    print(f"  adaptive retention = {ret:.3f}  -> meets {GATE_RETENTION:.2f}? "
          f"{ret >= GATE_RETENTION}")
    print(f"  rho_proxy(median)  = {med:.3f}  -> proxy only, "
          f"NOT a measured mixed-n figure")
    if med < 0.40:
        print("\n  => proxy clearly below 0.40: the mixed-n measurement is not "
              "worth GPU time; the halo family closes.")
    elif med >= GATE_RHO and ret >= GATE_RETENTION:
        print("\n  => proxy already at/above the gate AND retention holds: "
              "Stage 5D-B is warranted.")
    else:
        print("\n  => grey zone: Stage 5D-B is warranted (measured, not proxied).")
    print(f"\n  wrote 05d_* ({len(df)} image rows, {len(dist)} dist rows, "
          f"{len(prox)} proxy rows)")


if __name__ == "__main__":
    main()
