# -*- coding: utf-8 -*-
"""
Gate 7B -- Defect Evidence Utility & Calibration.

Gate 7A fixed a leak and the frozen dual-branch oracle dropped from a spurious
+48 to a real but modest +4 (wallplugs). Before investing in any generator we
need to know whether that ceiling is intrinsically low or merely badly fused.

7B-1  BRANCH INFORMATION. Score each branch alone. A constant shift cannot
      change AUROC, so the beta=2 improvement in Gate 7A proves S_d carries
      sample-dependent ranking information -- the open question is how much.
      Rescue/damage makes that concrete without any fusion scale:
        rescue = of the pairs the normal branch ranks WRONG, the fraction S_d
                 alone ranks right
        damage = of the pairs the normal branch ranks RIGHT, the fraction S_d
                 alone breaks

7B-2  LABEL-FREE FUSION. No target anomaly label may pick anything. Three
      calibrations, all using only the target's NORMAL support:
        raw        S_n - S_d                      (unstandardised)
        mad        z(S_n) - z(S_d), z = (x-med)/MAD
        cdf        F_n(S_n) - F_d(S_d), F = empirical CDF over normal patches

7B-3  ORACLE SUPPORT SIZE CURVE. How does the gain scale with defect-support
      quantity? If it keeps rising, a generator's job is support COVERAGE; if
      it is flat, generating more samples cannot help.

Protocol: strict support/eval separation, repeated splits, every mechanism
statistic from one fused score.

Usage: python experiments/model_v0/gate7b.py [--splits N]
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
SIZES = [5, 10, 20, 40, 80, 160, 320, -1]
METRICS = os.path.join(RESULTS, "metrics")


def nn_dist(query, bank, chunk=1024):
    q, b = l2norm(query.float()), l2norm(bank.float())
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ b.T).max(dim=1).values)
    return torch.cat(out)


def bottom1p(d):
    d = np.asarray(d).ravel()
    k = max(1, int(len(d) * 0.01))
    return float(np.sort(d)[:k].mean())


def setup(obj, split, device="cuda"):
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
    dsupp = torch.from_numpy(np.concatenate(
        [te["feats"][i].astype(np.float32)[gt[i] > 0.10] for i in sup_idx]
    )).to(device)
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(len(types))]
    with torch.no_grad():
        Sn = [nn_dist(z, fbank).cpu().numpy() for z in feats]
        Sd = [nn_dist(z, dsupp).cpu().numpy() for z in feats]
    good = [i for i in range(len(types)) if types[i] == "good"]
    ev = good + list(ev_idx)
    return Sn, Sd, good, ev, types, dsupp, fbank, feats


def branch_and_pairs(obj, split):
    Sn, Sd, good, ev, types, _, _, _ = setup(obj, split)
    y = np.array([0 if types[i] == "good" else 1 for i in ev])
    sn = np.array([mean_top1p(Sn[i]) for i in ev])
    sd = np.array([-bottom1p(Sd[i]) for i in ev])   # closer to defect support = higher
    a_n = roc_auc_score(y, sn)
    a_d = roc_auc_score(y, sd)
    # pair-level rescue / damage
    g, d = sn[y == 0], sn[y == 1]
    sd_g, sd_d = sd[y == 0], sd[y == 1]
    base_ok = g[:, None] < d[None, :]               # normal scored below defect
    sd_ok = sd_g[:, None] < sd_d[None, :]
    wrong = ~base_ok
    rescue = float(sd_ok[wrong].mean()) if wrong.sum() else np.nan
    damage = float((~sd_ok)[base_ok].mean()) if base_ok.sum() else np.nan
    return {"object": obj, "split": split, "AUROC_Sn": a_n * 100,
            "AUROC_Sd": a_d * 100, "rescue": rescue, "damage": damage,
            "n_pairs": int(base_ok.size)}


def fusion(obj, split):
    Sn, Sd, good, ev, types, _, _, _ = setup(obj, split)
    y = np.array([0 if types[i] == "good" else 1 for i in ev])
    nSn = np.concatenate([Sn[i] for i in good])
    nSd = np.concatenate([Sd[i] for i in good])
    mn, madn = np.median(nSn), np.median(np.abs(nSn - np.median(nSn))) + 1e-9
    md, madd = np.median(nSd), np.median(np.abs(nSd - np.median(nSd))) + 1e-9
    out = []
    for name in ("raw", "mad", "cdf"):
        def comb(i):
            if name == "raw":
                return Sn[i] - Sd[i]
            if name == "mad":
                return (Sn[i] - mn) / madn - (Sd[i] - md) / madd
            # empirical CDF over normal patches: higher = more abnormal
            return (np.searchsorted(np.sort(nSn), Sn[i]) / len(nSn)
                    - np.searchsorted(np.sort(nSd), Sd[i]) / len(nSd))
        sc = np.array([mean_top1p(comb(i)) for i in ev])
        out.append({"object": obj, "split": split, "fusion": name,
                    "img_AUROC": roc_auc_score(y, sc) * 100})
    return out


def size_curve(obj, split, sizes=SIZES):
    Sn, _, good, ev, types, dsupp, _, feats = setup(obj, split)
    y = np.array([0 if types[i] == "good" else 1 for i in ev])
    nSn = np.concatenate([Sn[i] for i in good])
    mn, madn = np.median(nSn), np.median(np.abs(nSn - np.median(nSn))) + 1e-9
    rng = np.random.default_rng(7 + split)
    rows = []
    for n in sizes:
        n_eff = len(dsupp) if n == -1 else min(n, len(dsupp))
        idx = (np.arange(len(dsupp)) if n == -1 else
               rng.choice(len(dsupp), size=n_eff, replace=False))
        sub = dsupp[torch.from_numpy(idx).to(dsupp.device)]
        with torch.no_grad():
            Sd2 = [nn_dist(feats[i], sub).cpu().numpy() for i in ev]
        # calibration on the SAME sub-bank's normal statistics, so the curve
        # isolates support coverage and not a shifting scale
        nSd2 = np.concatenate([Sd2[k] for k, i in enumerate(ev)
                               if types[i] == "good"])
        md = np.median(nSd2)
        madd = np.median(np.abs(nSd2 - md)) + 1e-9
        sc = np.array([mean_top1p((Sn[i] - mn) / madn - (Sd2[k] - md) / madd)
                       for k, i in enumerate(ev)])
        rows.append({"object": obj, "split": split, "support_patches": n_eff,
                     "img_AUROC": roc_auc_score(y, sc) * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)

    print("=" * 96)
    print("7B-1  BRANCH INFORMATION")
    print("=" * 96)
    b = []
    for o in OBJECTS:
        for s in range(args.splits):
            b.append(branch_and_pairs(o, s))
    bdf = pd.DataFrame(b)
    bdf.to_csv(os.path.join(METRICS, "gate7b_branch.csv"), index=False)
    print(f"\n  {'object':<12}{'AUROC(S_n)':>14}{'AUROC(S_d)':>14}"
          f"{'rescue':>10}{'damage':>10}{'n_pairs':>10}")
    for o in OBJECTS:
        s = bdf[bdf.object == o]
        print(f"  {o:<12}{s.AUROC_Sn.mean():>10.1f}±{s.AUROC_Sn.std():<3.1f}"
              f"{s.AUROC_Sd.mean():>10.1f}±{s.AUROC_Sd.std():<3.1f}"
              f"{s.rescue.mean():>10.3f}{s.damage.mean():>10.3f}"
              f"{int(s.n_pairs.mean()):>10}")

    print("\n" + "=" * 96)
    print("7B-2  LABEL-FREE FUSION (nothing tuned on target anomaly labels)")
    print("=" * 96)
    f = []
    for o in OBJECTS:
        for s in range(args.splits):
            f.extend(fusion(o, s))
    fdf = pd.DataFrame(f)
    fdf.to_csv(os.path.join(METRICS, "gate7b_fusion.csv"), index=False)
    piv = fdf.pivot_table(index="fusion", columns="object", values="img_AUROC")
    print(piv.round(1).to_string())
    print("\n  if 'mad'/'cdf' recover what raw beta=2 gave in Gate 7A, the problem")
    print("  was branch SCALE MISMATCH, not a low ceiling.")

    print("\n" + "=" * 96)
    print("7B-3  ORACLE SUPPORT SIZE CURVE (mad-calibrated fusion)")
    print("=" * 96)
    c = []
    for o in OBJECTS:
        for s in range(args.splits):
            c.extend(size_curve(o, s))
    cdf = pd.DataFrame(c)
    cdf.to_csv(os.path.join(METRICS, "gate7b_size_curve.csv"), index=False)
    piv2 = cdf.pivot_table(index="support_patches", columns="object",
                           values="img_AUROC")
    print(piv2.round(1).to_string())
    print("\n  rising with size -> the future problem is support COVERAGE")
    print("  flat -> generating more support samples cannot help")


if __name__ == "__main__":
    main()
