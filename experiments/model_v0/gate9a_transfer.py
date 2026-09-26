# -*- coding: utf-8 -*-
"""
Gate 9A transfer -- train the selector on MVTec AD v1, apply it to MVTec AD 2.

The LOCO result on v1 already contradicts the interaction hypothesis
(bank_only +2.90 vs interaction +1.14 against random), but v1 sits at a ceiling
(baseline 96.2, oracle 97.1) so it may simply be the wrong regime to ask the
question in. AD2 is where external banks actually matter: baselines run 42-90 and
a well-chosen bank is worth +20.

So this trains on ALL of v1 and evaluates on AD2, whose classes NEVER appear in
training. Compared against:
    baseline       no external bank at all
    random         mean utility over the candidate banks (Phase 8's R0 analogue)
    oracle         best candidate bank (upper bound)
    r1_criterion   the bank Phase 8's normal-manifold match would pick
    global_best    the single bank with the best mean utility on v1 -- what a
                   bank-only model collapses to, stated explicitly
    learned        the selector's top pick

Decision rule, fixed in advance (from the user's Gate 9A spec):
    baseline < learned  AND  learned clearly > random   -> retrieval is learnable
    learned ~ random                                    -> selection is not enough;
                                                           go to the Generator

Usage: python experiments/model_v0/gate9a_transfer.py [--epochs N] [--seeds N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate9a_selector import (DEV, VARIANTS, episode_id, fit, load, pick)  # noqa: E402
from tail_calib import RESULTS  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")


def standardise(train, other):
    """Scale by TRAINING-dataset statistics only -- v1 never sees AD2."""
    mu_t, sd_t = train[0].mean(0), train[0].std(0) + 1e-6
    mu_b, sd_b = train[1].mean(0), train[1].std(0) + 1e-6
    f = lambda X, m, s: ((X - m) / s).astype(np.float32)   # noqa: E731
    return (f(train[0], mu_t, sd_t), f(train[1], mu_b, sd_b)), \
           (f(other[0], mu_t, sd_t), f(other[1], mu_b, sd_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    h_t, h_b, u, meta = load("v1")
    V = (h_t, h_b, u, meta)
    ta, _ = standardise(V, V)
    tgt_v = np.array(meta.target)
    # validation classes for early stopping, carved from v1 only
    val_cls = set(sorted(set(tgt_v))[::5])
    vm = np.array([c in val_cls for c in tgt_v])
    tr = (ta[0][~vm], ta[1][~vm], u[~vm], meta[~vm].reset_index(drop=True))
    va = (ta[0][vm], ta[1][vm], u[vm], meta[vm].reset_index(drop=True))

    # global-best bank: purely bank-side, the thing bank_only collapses to
    gb = (pd.DataFrame({"bank": meta.bank, "u": u})
          .groupby("bank").u.mean().sort_values(ascending=False))
    print("v1 mean utility by bank (top 5):")
    print(gb.head(5).round(1).to_string())

    z = np.load(os.path.join(METRICS, "selector_ad2.npz"), allow_pickle=True)
    m2 = pd.DataFrame({
        "target": [x[0] for x in z["meta"]], "bank": [x[1] for x in z["meta"]],
        "shot": [x[2] for x in z["meta"]], "seed": [x[3] for x in z["meta"]],
        "size": [x[4] for x in z["meta"]], "baseline": [x[5] for x in z["meta"]]})
    u2 = z["utility"]
    A = (z["h_t"], z["h_b"], u2, m2)
    _, ad = standardise(V, A)
    R1 = pd.read_csv(os.path.join(METRICS, "gate8_class_match.csv"))
    r1_pick = {o: g.sort_values("match_dist")["class"].iloc[0]
               for o, g in R1.groupby("target")}

    eid = episode_id(m2).to_numpy()
    rows = []
    for v in VARIANTS:
        for sd in range(a.seeds):
            model, vl = fit(tr, va, v, epochs=a.epochs, seed=sd)
            chosen, _ = pick(model, ad[0], ad[1], m2, k=1)
            for g, idx in pd.Series(range(len(u2))).groupby(eid):
                i = idx.to_numpy()
                t = m2.target.iloc[i[0]]
                # rows matching the R1-chosen bank, if present
                r1m = i[(m2.bank.iloc[i] == r1_pick.get(t, None)).to_numpy()]
                rows.append({
                    "variant": v, "seed": sd, "episode": g, "target": t,
                    "shot": m2.shot.iloc[i[0]], "size": m2["size"].iloc[i[0]],
                    "baseline": m2.baseline.iloc[i[0]],
                    "learned": u2[chosen[g][0]],
                    "random": float(np.mean(u2[i])),
                    "oracle": float(np.max(u2[i])),
                    "picked_bank": m2.bank.iloc[chosen[g][0]],
                    "oracle_bank": m2.bank.iloc[i[np.argmax(u2[i])]],
                    "r1": float(u2[r1m[0]]) if len(r1m) else np.nan,
                })
            print(f"  {v} seed={sd} done (val loss {vl:.4f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, "gate9a_transfer_ad2.csv"), index=False)
    piv = df.pivot_table(index="target", columns="variant",
                         values=["learned", "baseline", "random", "oracle", "r1"])
    print("\n" + "=" * 100)
    print("GATE 9A TRANSFER -- trained on MVTec AD v1, evaluated on AD2 "
          "(mean over shots/sizes/seeds)")
    print("=" * 100)
    per = df.groupby(["variant", "target"]).agg(
        base=("baseline", "mean"), learned=("learned", "mean"),
        random=("random", "mean"), oracle=("oracle", "mean"),
        r1=("r1", "mean")).round(1)
    for v in VARIANTS:
        print(f"\n  --- {v} ---")
        s = per.xs(v, level="variant")
        print(s.to_string())
        d = s.learned - s["random"]
        print(f"  learned - random : {d.mean():+.1f}  ({int((d > 0).sum())}/{len(d)} targets better)")
        gap = s.oracle - s["random"]
        print(f"  oracle gap closed: {100 * (d / gap.replace(0, np.nan)).mean():.1f}%")
        print(f"  learned > baseline: {int((s.learned > s.base).sum())}/{len(s)}    "
              f"(random > baseline: {int((s['random'] > s.base).sum())}/{len(s)})")
        print(f"  beats R1 criterion: {int((s.learned > s.r1).sum())}/{len(s)}")


if __name__ == "__main__":
    main()
