# -*- coding: utf-8 -*-
"""
Gate 7A -- clean oracle for the frozen dual-branch detector.

The Gate 6C numbers are NOT usable as an oracle result: its defect support was
built from the SUPERVISION HALF of the defect images while evaluation ran over
ALL defect images, so part of the bank was its own test set. This script fixes
that and three other issues.

  1. STRICT SPLIT. Defect images are split 50/50 into support and evaluation at
     a per-split seed, repeated over N_SPLITS. The support half never appears in
     the evaluation set.
  2. ONE SCORE FOR ALL MECHANISM STATS. normal_p99, defect_mean and
     frac_defect_below_p99 are now all computed from the SAME fused score.
     Gate 6C mixed S_n for the normal side with S_n - beta*S_d for the defect
     side, which is not a comparison.
  3. NO TEST-LABEL BETA. beta is fixed at 1 after standardising each branch
     against the target's own NORMAL support (median/MAD). Nothing about the
     target's anomalies is used to pick it; the raw beta sweep is printed only
     as a diagnostic of how sensitive the result is.
  4. REPEATED SPLITS. Every number is mean +- std over N_SPLITS, so a single
     lucky support set cannot carry the conclusion.

Architecture: frozen DINOv2, no adapter, no training.
    S_n  per-patch 1-NN distance to the NORMAL bank
    S_d  per-patch 1-NN distance to the DEFECT support bank
    score = z(S_n) - beta * z(S_d)   with z = (x - median)/MAD over normal patches

Usage: python experiments/model_v0/gate7a.py [--splits N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from models.tail_adapter import l2norm
from tail_calib import RESULTS, load_cache, split_train, mean_top1p

OBJECTS = ["wallplugs", "sheet_metal", "vial", "can"]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
METRICS = os.path.join(RESULTS, "metrics")


def nn_dist(query, bank, chunk=1024):
    q, b = l2norm(query.float()), l2norm(bank.float())
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ b.T).max(dim=1).values)
    return torch.cat(out)


def one_split(obj, split, device="cuda"):
    tr, te = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                             .astype(np.float32)).to(device)
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    bad = np.where(~(np.array(types) == "good"))[0]
    perm = np.random.default_rng(1000 + split).permutation(len(bad))
    sup_idx = bad[perm[:len(bad) // 2]]
    ev_idx = bad[perm[len(bad) // 2:]]

    # defect support bank: ONLY the support half of the defect images
    dsupp = torch.from_numpy(np.concatenate(
        [te["feats"][i].astype(np.float32)[gt[i] > 0.10] for i in sup_idx]
    )).to(device)

    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(len(types))]
    with torch.no_grad():
        Sn = [nn_dist(z, fbank).cpu().numpy() for z in feats]
        Sd = [nn_dist(z, dsupp).cpu().numpy() for z in feats]

    good = [i for i in range(len(types)) if types[i] == "good"]
    # calibration statistics come from NORMAL patches only -- no target anomaly
    # label is involved anywhere in choosing beta
    nSn = np.concatenate([Sn[i] for i in good])
    nSd = np.concatenate([Sd[i] for i in good])
    mn, madn = np.median(nSn), np.median(np.abs(nSn - np.median(nSn))) + 1e-9
    md, madd = np.median(nSd), np.median(np.abs(nSd - np.median(nSd))) + 1e-9

    # evaluation set: held-out defect images + all good images
    ev = good + list(ev_idx)
    y = np.array([0 if types[i] == "good" else 1 for i in ev])
    rows = []
    for beta in BETAS:
        sc = np.array([mean_top1p((Sn[i] - mn) / madn - beta * (Sd[i] - md) / madd)
                       for i in ev])
        fused_all = np.concatenate([(Sn[i] - mn) / madn - beta * (Sd[i] - md) / madd
                                    for i in ev])
        # every mechanism stat from the SAME fused score
        f_norm = np.concatenate([(Sn[i] - mn) / madn - beta * (Sd[i] - md) / madd
                                 for i in ev if types[i] == "good"])
        f_def = np.concatenate([
            ((Sn[i] - mn) / madn - beta * (Sd[i] - md) / madd)[gt[i] > 0.10]
            for i in ev if types[i] != "good"])
        p99 = float(np.percentile(f_norm, 99))
        rows.append({
            "object": obj, "split": split, "beta": beta,
            "n_eval_bad": int((y == 1).sum()),
            "img_AUROC": roc_auc_score(y, sc) * 100,
            "normal_p99": p99,
            "defect_mean": float(f_def.mean()),
            "frac_defect_below_p99": float((f_def < p99).mean()),
            "defect_p90": float(np.percentile(f_def, 90)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)
    rows = []
    for o in OBJECTS:
        for s in range(args.splits):
            rows.extend(one_split(o, s))
            print(f"  {o} split {s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, "gate7a_clean_oracle.csv"), index=False)

    print("\n" + "=" * 100)
    print("GATE 7A -- clean frozen dual-branch oracle "
          f"(strict defect split, {args.splits} splits)")
    print("=" * 100)
    print("\n  Headline: beta = 1 after median/MAD standardisation against the "
          "target's NORMAL support")
    print(f"\n  {'object':<12}{'img_AUROC':>18}{'normal_p99':>16}"
          f"{'defect_mean':>17}{'defect<p99':>16}")
    for o in OBJECTS:
        s = df[(df.object == o) & (df.beta == 1.0)]
        print(f"  {o:<12}" + "".join(
            f"{s[c].mean():>10.2f}±{s[c].std():<6.2f}" for c in
            ["img_AUROC", "normal_p99", "defect_mean",
             "frac_defect_below_p99"]))
    print("\n  Frozen AnomalyDINO baseline (config A, full bad set):")
    print("    wallplugs 41.7 | sheet_metal 83.6 | vial 91.7 | can 51.5")
    print("\n  beta sensitivity (mean img_AUROC over splits):")
    piv = df.pivot_table(index="beta", columns="object", values="img_AUROC")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
