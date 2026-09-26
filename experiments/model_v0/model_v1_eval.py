# -*- coding: utf-8 -*-
"""
Model V1 -- real anomaly detection evaluation (the experiment that matters).

Sixteen gates of mechanism work produced one reliable positive result:

    Frozen DINOv2 final-layer patch features
      + target normal global covariance
      + fractional whitening M_t = (Sigma_t + eps I)^-0.5, gamma = 0.5
      -> anomaly scoring

with no target anomaly labels, no source anomaly data and no training. Defect
retrieval improves 84.7 -> 86.6 and, crucially, the own-vs-shuffled control
holds it up: using ANOTHER object's covariance gives only 84.6, own - shuffled
= +1.94 on 11/15 (gate 12, equivalent protocol, reported +2.6 on 12/15).

But retrieval Recall@1 is not the target metric. This script asks the only
question that still matters: does that representation gain survive contact with
real anomaly detection -- image AUROC, pixel AUROC, AU-PRO@0.05 -- across shot
counts and splits?

    raw        frozen DINOv2, no correction (the AnomalyDINO baseline)
    V1_shot    covariance estimated from the K few-shot normal images ONLY
    V1_full    covariance estimated from all train/good

V1_shot is the honest few-shot setting; V1_full shows what the extra normal data
buys. A 384x384 covariance from one image's ~1000 patches is badly
under-determined, so the gap between them is itself a finding.

Scoring is unchanged from the baseline: 1-NN patch distance to the few-shot
normal bank, image score = mean of the top 1% patch distances.

Usage: python experiments/model_v0/model_v1_eval.py [--shots 1,2,4,8] [--splits 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
# Per-image train features live in cache_v1 (cache_ml's train split is
# flattened and subsampled, so K-shot IMAGE selection is impossible there).
ML = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_v1")
DEV = "cuda"
GAMMA = 0.5
EPS_FRAC = 0.05
NORMAL_CAP = 20000


def l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nn_dist(q, bank, chunk=1024):
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ bank.T).max(dim=1).values)
    return torch.cat(out)


def frac_pow(C, gamma=GAMMA, eps_frac=EPS_FRAC):
    """Floor the eigenvalues at a fraction of the mean POSITIVE eigenvalue.

    eps_frac * w.mean() is 0 when the covariance is identically zero (e.g. a
    1-sample "shot"), and 0 ** -0.25 = inf -> NaN downstream. Flooring against
    the mean positive eigenvalue keeps a degenerate C from poisoning the run.
    """
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    pos = w[w > 0]
    floor = eps_frac * (pos.mean() if len(pos) else 1.0)
    w = np.clip(w, floor, None)
    return ((V * w ** (-gamma / 2.0)) @ V.T).astype(np.float32)


def whitener(X):
    Xd = X.astype(np.float64)
    mu = Xd.mean(0)
    Xc = Xd - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    return mu.astype(np.float32), frac_pow(C)


def gt_path(obj, name):
    """name is '<type>/<file>.png' in the v1 cache."""
    typ, f = str(name).split("/")
    return os.path.join(V1_ROOT, obj, "ground_truth", typ,
                        f[:-4] + "_mask.png")


def run(obj, shot, split, normal_img):
    te = np.load(os.path.join(ML, f"{obj}_test.npz"), allow_pickle=True)
    te = {k: np.asarray(te[k]) for k in te.files}
    types = list(te["types"])
    n_img = len(types)
    feats = [torch.from_numpy(te["feats"][i].astype(np.float32)) for i in range(n_img)]
    y = np.array([1 if t == "bad" else 0 for t in types])

    rng = np.random.default_rng(1000 + split)
    n_tr = len(normal_img)
    idx = rng.choice(n_tr, size=min(shot, n_tr), replace=False)
    bank_raw = torch.from_numpy(normal_img[idx].reshape(-1, 384)).to(DEV)

    configs = {}
    configs["raw"] = (None, None)
    mu_s, W_s = whitener(normal_img[idx].reshape(-1, 384))
    configs["V1_shot"] = (mu_s, W_s)
    mu_f, W_f = whitener(
        normal_img.reshape(-1, 384)[rng.choice(n_tr * normal_img.shape[1],
                                              size=min(NORMAL_CAP,
                                                       n_tr * normal_img.shape[1]),
                                              replace=False)])
    configs["V1_full"] = (mu_f, W_f)

    rows = []
    for name, (mu, W) in configs.items():
        if mu is None:
            bk = l2n(bank_raw)
            tf = [l2n(f.to(DEV)) for f in feats]
        else:
            mut = torch.from_numpy(mu).to(DEV)
            Wt = torch.from_numpy(W).to(DEV)
            bk = l2n((bank_raw - mut) @ Wt.T)
            tf = [l2n((f.to(DEV) - mut) @ Wt.T) for f in feats]
        with torch.no_grad():
            dists = [nn_dist(f, bk).cpu().numpy() for f in tf]
        img = np.array([mean_top1p(d) for d in dists])
        row = {"object": obj, "shot": shot, "split": split, "config": name,
               "img_AUROC": roc_auc_score(y, img) * 100,
               "n_good": int((y == 0).sum()), "n_bad": int((y == 1).sum())}
        d = os.path.join(MAPS, name, obj, f"{shot}shot_s{split}")
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(n_img):
            p = os.path.join(d, str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            np.save(p, dists[i].reshape(tuple(te["gt_frac"].shape[1:])))
            g = gt_path(obj, te["names"][i]) if types[i] == "bad" else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        row["px_AUROC"] = m["px_AUROC"] * 100
        row["AUPRO"] = m["AUPRO"] * 100
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--objects", default="all")
    a = ap.parse_args()
    shots = [int(s) for s in a.shots.split(",")]

    files = sorted(f[:-len("_train.npz")] for f in os.listdir(ML)
                   if f.endswith("_train.npz"))
    objs = files if a.objects == "all" else a.objects.split(",")

    rows = []
    cpath = os.path.join(METRICS, "model_v1_downstream.csv")
    for obj in objs:
        tr = np.load(os.path.join(ML, f"{obj}_train.npz"), allow_pickle=True)
        # keep PER-IMAGE grouping: the K-shot bank must be K images, not K patches
        normal_img = np.asarray(tr["feats"]).astype(np.float32)
        for shot in shots:
            for split in range(a.splits):
                rows.extend(run(obj, shot, split, normal_img))
            pd.DataFrame(rows).to_csv(cpath, index=False)
            print(f"  {obj:<12} {shot}-shot done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(cpath, index=False)
    print("\n" + "=" * 100)
    print("MODEL V1 -- downstream anomaly detection (MVTec AD v1, frozen DINOv2)")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        piv = df.pivot_table(index="shot", columns="config", values=met)
        print(f"\n  {met} (mean over objects x splits):")
        print(piv.round(2).to_string())
        v = df.groupby(["shot", "config"])[met].agg(["mean", "std"]).round(2)
        print("   " + " | ".join(
            f"{s}shot: " + " ".join(
                f"{c} {v.loc[(s, c), 'mean']:.1f}±{v.loc[(s, c), 'std']:.1f}"
                for c in ["raw", "V1_shot", "V1_full"] if (s, c) in v.index)
            for s in shots))

    print("\n" + "=" * 100)
    print("GAIN OVER RAW (per shot)")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        piv = df.pivot_table(index="shot", columns="config", values=met)
        d = piv.sub(piv["raw"], axis=0)
        print(f"\n  {met}:")
        print(d[["V1_shot", "V1_full"]].round(2).to_string())

    wins = (df[df.config == "V1_full"].set_index(["object", "shot", "split"])
            ["img_AUROC"]
            - df[df.config == "raw"].set_index(["object", "shot", "split"])
            ["img_AUROC"])
    print(f"\n  V1_full vs raw image AUROC, per (object, shot, split): "
          f"{int((wins > 0).sum())}/{len(wins)} positive, mean {wins.mean():+.2f}")


if __name__ == "__main__":
    main()
