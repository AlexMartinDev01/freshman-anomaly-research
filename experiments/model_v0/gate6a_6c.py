# -*- coding: utf-8 -*-
"""
Gate 6A (normal-generalization audit) and Gate 6C (frozen dual-branch oracle).

6A asks WHY unseen normals get pushed away. The K4 adapter leaves the TRAINING
normal tail flat (cosine p99 x0.98-1.01 through training) yet TEST_pub normal
p99 rises 0.104 -> 0.333. That is the signature of support overfitting: the
adapter may be preserving the few support images rather than normality. Test:
for every test_public good image, correlate its distance to the support (in
frozen DINO space) against how much the adapter inflates its score.

6C asks the more basic question: should a shared adapter be trained at all?
Frozen features, two banks, no representation learning:
    S_n  per-patch 1-NN distance to the NORMAL bank        (AnomalyDINO score)
    S_d  per-patch 1-NN distance to the DEFECT-support bank
    patch score = S_n - beta * S_d
If that alone captures E's gain, then adapting the feature space was never the
right mechanism and the only real problem left is predicting the defect support.

The defect support is the K4 bootstrap set (oracle: built from real defect
coefficients), so 6C is an oracle probe, not a deployable method.

Usage: python experiments/model_v0/gate6a_6c.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from models import ResidualAdapter
from models.tail_adapter import l2norm
from pixel_metrics_binned import pixel_metrics_binned
from tail_calib import RESULTS, load_cache, split_train, mean_top1p
from ad2_pipeline import AD2_ROOT

OBJECTS = ["wallplugs", "sheet_metal", "vial", "can"]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
MAPS = os.path.join(RESULTS, "maps_gate6c")
METRICS = os.path.join(RESULTS, "metrics")


def parse_name(name):
    m = re.match(r"(\d{3})_(.+)\.png", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def nn_dist(query, bank, chunk=1024):
    q, b = l2norm(query.float()), l2norm(bank.float())
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ b.T).max(dim=1).values)
    return torch.cat(out)


def gate6a(obj, device="cuda"):
    """Does score inflation grow with distance from the support?"""
    tr, te = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                             .astype(np.float32)).to(device)
    ck = os.path.join(RESULTS, "checkpoints", f"{obj}_K4.pt")
    if not os.path.exists(ck):
        print(f"  {obj}: no K4 checkpoint, skip 6A")
        return []
    adapter = ResidualAdapter(384).to(device)
    adapter.load_state_dict(torch.load(ck, map_location=device))
    adapter.eval()
    abank = adapter(fbank)

    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    rows = []
    for i, t in enumerate(types):
        if t != "good":
            continue
        z = torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
        with torch.no_grad():
            s_before = mean_top1p(nn_dist(z, fbank).cpu().numpy())
            s_after = mean_top1p(nn_dist(adapter(z), abank).cpu().numpy())
        scene, light = parse_name(str(te["names"][i]))
        rows.append({"object": obj, "image": str(te["names"][i]),
                     "scene": scene, "light": light,
                     "d_support": s_before, "s_after": s_after,
                     "delta": s_after - s_before})
    return rows


def gate6c(obj, device="cuda"):
    """Frozen dual-branch: score = S_n - beta * S_d. No adapter, no training."""
    tr, te = load_cache(obj)
    bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
    fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                             .astype(np.float32)).to(device)
    dpath = os.path.join(RESULTS, "proxy", f"{obj}_K4boot.npy")
    if not os.path.exists(dpath):
        print(f"  {obj}: no K4 defect support, skip 6C")
        return []
    dsupp = torch.from_numpy(np.load(dpath).astype(np.float32)).to(device)

    grid = tuple(int(v) for v in te["gt_frac"].shape[1:])
    types = list(te["types"])
    gt = te["gt_frac"].reshape(len(types), -1)
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
             for i in range(len(types))]
    with torch.no_grad():
        Sn = [nn_dist(z, fbank).cpu().numpy() for z in feats]
        Sd = [nn_dist(z, dsupp).cpu().numpy() for z in feats]

    y = np.array([1 if t == "bad" else 0 for t in types])
    rows = []
    for beta in BETAS:
        sc = np.array([mean_top1p(Sn[i] - beta * Sd[i]) for i in range(len(types))])
        auroc = roc_auc_score(y, sc)
        # mechanism stats
        gmask = y == 0
        nmask = np.repeat(gmask, [len(Sn[i]) for i in range(len(types))])
        alln = np.concatenate([Sn[i] for i in range(len(types)) if gmask[i]])
        bad = [i for i in range(len(types)) if y[i]]
        defd = np.concatenate([(Sn[i] - beta * Sd[i])[gt[i] > 0.10] for i in bad])
        p99 = float(np.percentile(alln, 99))
        # pixel-level metrics
        base = np.array(Sn)
        adj = np.array([Sn[i] - beta * Sd[i] for i in range(len(types))])
        out = os.path.join(MAPS, obj, f"b{beta}")
        jobs = []
        for i, t in enumerate(types):
            d = os.path.join(out, t)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, str(te["names"][i])[:-4] + ".npy")
            np.save(p, adj[i].reshape(grid))
            g = (os.path.join(AD2_ROOT, obj, "test_public", "ground_truth",
                              "bad", str(te["names"][i])[:-4] + "_mask.png")
                 if t == "bad" else None)
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "beta": beta,
                     "img_AUROC": auroc * 100, "px_AUROC": m["px_AUROC"] * 100,
                     "AUPRO@0.05": m["AUPRO"] * 100,
                     "normal_p99": p99,
                     "defect_mean": float(defd.mean()),
                     "frac_defect_below_p99": float((defd < p99).mean()),
                     "defect_p50": float(np.median(defd))})
        print(f"  {obj:<12} beta={beta:<5} img={auroc*100:5.1f} "
              f"px={m['px_AUROC']*100:5.1f} AU-PRO={m['AUPRO']*100:5.1f} "
              f"p99={p99:.3f} defect={float(defd.mean()):.3f}", flush=True)
    return rows


def main():
    os.makedirs(METRICS, exist_ok=True)
    print("=" * 96)
    print("GATE 6A -- does the adapter inflate unseen normals further the farther "
          "they sit from the support?")
    print("=" * 96)
    a_rows = []
    for o in OBJECTS:
        a_rows.extend(gate6a(o))
    if a_rows:
        a = pd.DataFrame(a_rows)
        a.to_csv(os.path.join(METRICS, "gate6a_normal_audit.csv"), index=False)
        print(f"\n{'object':<12}{'n_good':>7}{'spearman(d_support, delta)':>30}"
              f"{'p':>8}{'mean d_support':>16}{'mean delta':>12}")
        for o in OBJECTS:
            s = a[a.object == o]
            if len(s) < 3:
                continue
            r, p = spearmanr(s.d_support, s.delta)
            print(f"{o:<12}{len(s):>7}{r:>30.3f}{p:>8.3f}"
                  f"{s.d_support.mean():>16.3f}{s.delta.mean():>12.3f}")
        print("\n  Positive correlation => the adapter pushes normals away in "
              "proportion to how")
        print("  far they already were from the support: support overfitting, not "
              "a learned invariance.")

    print("\n" + "=" * 96)
    print("GATE 6C -- FROZEN dual-branch oracle: score = S_n - beta * S_d, "
          "no adapter, no training")
    print("=" * 96)
    c_rows = []
    for o in OBJECTS:
        c_rows.extend(gate6c(o))
    if c_rows:
        c = pd.DataFrame(c_rows)
        c.to_csv(os.path.join(METRICS, "gate6c_frozen_dual_branch.csv"),
                 index=False)
        print("\n  A_half reference (frozen AnomalyDINO):")
        print("    wallplugs 46.4 / 2.5 | sheet_metal 86.6 / 51.2 | "
              "vial 90.2 / 74.3 | can 53.7 / 8.7   (img / AU-PRO)")


if __name__ == "__main__":
    main()
