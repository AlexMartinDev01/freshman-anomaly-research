# -*- coding: utf-8 -*-
"""
Gate 17A -- does the covariance score carry ANY usable downstream increment?

Model V1 failed badly (image AUROC -6.45, AUPRO -31 on 15 objects x 4 shots x 3
splits), and the mechanism is measured: whitening lifts good images 3.1x but bad
images only 1.8x, compressing the good/bad gap by 44%. But a representation that
is bad as a REPLACEMENT score can still hold complementary information, and that
is a different question from whether it works alone.

This is the last diagnostic on the covariance branch, and it is an ORACLE one:
target labels are used to pick the fusion weight, so it is an upper bound rather
than a method. Nothing is retrained, no features are recomputed -- the saved
anomaly maps are reused.

    S_alpha = (1 - alpha) * z(S_raw) + alpha * z(S_cov)

Both scores are z-scored over the object's test set before mixing, so alpha is a
meaningful exchange rate. Alpha is swept, and the BEST alpha is reported per
object, per metric -- the most favourable reading possible.

Pre-registered STOP/GO, frozen before the run:
    STOP   best oracle fusion beats raw by < +0.5 image AUROC, or improves
           fewer than 10/15 objects -> close the covariance branch entirely.
           No tail calibration, no adaptive weighting, no gating, no score-fusion
           network: if even the oracle has no headroom, none of those can work.
    GO     >= +1.0 and >= 10/15 -> then, and only then, the problem is not the
           covariance but using it as the PRIMARY score, and the next question
           becomes normal-tail-aware score calibration (Gate 17B).

Usage: python experiments/model_v0/gate17a_fusion_oracle.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402

V1_ROOT = r"E:\work\freshman\data\mvtec_anomaly_detection"
ML = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_v1")
FUSED = os.path.join(RESULTS, "maps_v1_fused")
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
PIX_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def fname(name):
    return str(name)[:-4].replace("/", "_") + ".npy"


def gt_path(obj, name):
    typ, f = str(name).split("/")
    return os.path.join(V1_ROOT, obj, "ground_truth", typ, f[:-4] + "_mask.png")


def load_set(cfg, obj, shot, split, te):
    d = os.path.join(MAPS, cfg, obj, f"{shot}shot_s{split}")
    out = []
    for i in range(len(te["types"])):
        p = os.path.join(d, fname(te["names"][i]))
        if not os.path.exists(p):
            return None
        out.append(np.load(p))
    return out


def zscores(maps):
    """Pool the patches of this object's test set; z-score each map with it."""
    allv = np.concatenate([m.ravel() for m in maps])
    mu, sd = allv.mean(), allv.std() + 1e-12
    return [(m - mu) / sd for m in maps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,4,8")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--cov", default="V1_full")
    a = ap.parse_args()
    shots = [int(s) for s in a.shots.split(",")]
    os.makedirs(FUSED, exist_ok=True)

    files = sorted(f[:-len("_train.npz")] for f in os.listdir(ML)
                   if f.endswith("_train.npz"))
    rows = []
    for obj in files:
        te = np.load(os.path.join(ML, f"{obj}_test.npz"), allow_pickle=True)
        te = {k: np.asarray(te[k]) for k in te.files}
        y = np.array([1 if t == "bad" else 0 for t in te["types"]])
        # keep the 2-D grid: `gt.shape[1:]` below is used to reshape the fused
        # map before saving, and a flattened gt gives (1024,) -- a 1-D "map" that
        # cv2.resize treats as a COLUMN, scrambling the spatial layout entirely.
        # That produced px_AUROC 81.5 / AUPRO 14.4 against the correct 96.3 / 73.9.
        G = te["gt_frac"].reshape(len(y), *te["gt_frac"].shape[1:])
        gt = G.reshape(len(y), -1)
        for shot in shots:
            for split in range(a.splits):
                R = load_set("raw", obj, shot, split, te)
                C = load_set(a.cov, obj, shot, split, te)
                if R is None or C is None:
                    continue
                ZR, ZC = zscores(R), zscores(C)
                for alpha in ALPHAS:
                    F = [(1 - alpha) * zr + alpha * zc for zr, zc in zip(ZR, ZC)]
                    sc = np.array([mean_top1p(f) for f in F])
                    row = {"object": obj, "shot": shot, "split": split,
                           "alpha": alpha,
                           "img_AUROC": roc_auc_score(y, sc) * 100}
                    if alpha in PIX_ALPHAS and split == 0:
                        d = os.path.join(FUSED, obj, f"{shot}shot_s{split}_a{alpha}")
                        os.makedirs(d, exist_ok=True)
                        jobs = []
                        for i in range(len(y)):
                            p = os.path.join(d, fname(te["names"][i]))
                            np.save(p, F[i].reshape(G.shape[1:]))
                            g = gt_path(obj, te["names"][i]) if y[i] else None
                            jobs.append((p, g, tuple(int(v) for v in
                                                     te["img_hw"][i])))
                        m = pixel_metrics_binned(jobs, pro_limit=0.05)
                        row["px_AUROC"] = m["px_AUROC"] * 100
                        row["AUPRO"] = m["AUPRO"] * 100
                    rows.append(row)
                print(f"  {obj:<12} {shot}shot s{split} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, "gate17a_fusion.csv"), index=False)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(METRICS, "gate17a_fusion.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"GATE 17A -- oracle fusion of raw and {a.cov} scores")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        if met not in R.columns:
            continue
        S = R.dropna(subset=[met])
        piv = S.pivot_table(index=["object", "shot", "split"], columns="alpha",
                            values=met)
        raw = piv[0.0]
        best = piv.max(axis=1)
        d = best - raw
        print(f"\n  {met}: best-alpha oracle fusion vs raw")
        print(f"    mean raw {raw.mean():.2f}   mean best {best.mean():.2f}   "
              f"gain {d.mean():+.2f}   better {int((d > 0).sum())}/{len(d)}")
        per_obj = pd.DataFrame({"raw": raw, "best": best, "gain": d}).groupby(
            level=0).mean()
        print(f"    per-object gain: " + " ".join(
            f"{o}={v:+.1f}" for o, v in per_obj.gain.items()))
        amode = piv.idxmax(axis=1).value_counts().head(3)
        print(f"    most common best alpha: " +
              ", ".join(f"{k} x{v}" for k, v in amode.items()))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED")
    print("=" * 100)
    S = R.dropna(subset=["img_AUROC"])
    piv = S.pivot_table(index=["object", "shot", "split"], columns="alpha",
                        values="img_AUROC")
    raw, best = piv[0.0], piv.max(axis=1)
    po = pd.DataFrame({"raw": raw, "best": best}).groupby(level=0).mean()
    g = (po.best - po.raw)
    b, bb = g.mean(), int((g > 0).sum())
    print(f"  image AUROC: mean gain {b:+.2f}, better {bb}/{len(g)} objects")
    if b >= 1.0 and bb >= 10:
        print("  GO   -> covariance holds complementary information; next is "
              "normal-tail-aware score calibration (Gate 17B)")
    else:
        print("  STOP -> close the covariance branch. No tail calibration, no "
              "adaptive weighting, no gating, no fusion network: the oracle has "
              "no headroom, so none of those can create any.")


if __name__ == "__main__":
    main()
