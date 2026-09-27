# -*- coding: utf-8 -*-
"""
Phase R step 2 -- downstream-only layer smoke test. Run once.

Gate 16 selected the final layer using Recall@1, and Gate 17A then established
that Recall@1 is a misleading proxy: it rose +1.9 while image AUROC fell 6.45 and
AUPRO fell 31. So the layer choice has NOT actually been tested on the metric
that matters, and recent work reports mid-layer aggregation beating the final
layer on real downstream AD. This retests it properly.

    final_raw     block 11 patch distances
    mid_raw       block 5
    midlate_raw   block 8
    mid_mean      z(mid) + z(midlate), averaged
    agg_all3      z(mid) + z(midlate) + z(final), averaged

No covariance, no training, no block sweep. Metrics are real downstream only:
image AUROC, pixel AUROC, AU-PRO@0.05.

Bank control: cache_ml's train split is flattened, so the bank is a matched
PATCH budget (shot x 1024 patches) rather than shot IMAGES. That is also the
right control here -- Gate 7B-4 showed patch count is itself a strong lever, so
holding it fixed is what isolates the layer effect. The same budget is used for
every layer.

Layer maps are z-scored over the object's test set before averaging, so the
average is not dominated by whichever layer happens to have the larger scale.

Objects: one texture (carpet), one object (bottle), one hard one (transistor,
whose AUPRO@0.05 is 49.0 -- the lowest in the baseline audit).

Usage: python experiments/model_v0/gate_r1_layer_smoke.py
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
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402

V1 = r"E:\work\freshman\data\mvtec_anomaly_detection"
ML = os.path.join(RESULTS, "cache_ml")
METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_layer_smoke")
DEV = "cuda"
LAYERS = ["mid", "midlate", "final"]
OBJECTS = ["carpet", "bottle", "transistor"]
SHOTS = [1, 4]
PATCHES_PER_IMAGE = 1024


def l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nn_dist(q, bank, chunk=1024):
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ bank.T).max(dim=1).values)
    return torch.cat(out)


def gt_path(obj, name):
    typ, f = str(name).split("/")
    return os.path.join(V1, obj, "ground_truth", typ, f[:-4] + "_mask.png")


def run(obj, shot, split, want=None):
    rng = np.random.default_rng(1000 + split)
    per_layer = {}
    for layer in LAYERS:
        te = np.load(os.path.join(ML, layer, f"{obj}_test.npz"),
                     allow_pickle=True)
        te = {k: np.asarray(te[k]) for k in te.files}
        tr = np.load(os.path.join(ML, layer, f"{obj}_train.npz"),
                     allow_pickle=True)
        pool = np.asarray(tr["feats"]).reshape(-1, 384).astype(np.float32)
        n = shot * PATCHES_PER_IMAGE
        bank = torch.from_numpy(
            pool[rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
        ).to(DEV)
        bk = l2n(bank)
        maps = []
        for i in range(len(te["types"])):
            z = torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(z), bk).cpu().numpy())
        if layer == LAYERS[0]:
            keep = te
        per_layer[layer] = maps

    def z(maps):
        allv = np.concatenate([m.ravel() for m in maps])
        mu, sd = allv.mean(), allv.std() + 1e-12
        return [(m - mu) / sd for m in maps]

    Z = {k: z(v) for k, v in per_layer.items()}
    configs = {
        "final_raw": Z["final"],
        "mid_raw": Z["mid"],
        "midlate_raw": Z["midlate"],
        "mid_mean": [(a + b) / 2 for a, b in zip(Z["mid"], Z["midlate"])],
        "agg_all3": [(a + b + c) / 3 for a, b, c in
                     zip(Z["mid"], Z["midlate"], Z["final"])],
    }

    te = keep
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    rows = []
    for name, M in configs.items():
        if want and name not in want:
            continue
        sc = np.array([mean_top1p(m) for m in M])
        d = os.path.join(MAPS, obj, f"{shot}shot_s{split}", name)
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(len(y)):
            p = os.path.join(d, str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            np.save(p, M[i].reshape(gh, gw))          # ALWAYS a 2-D grid
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "config": name,
                     "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(OBJECTS))
    ap.add_argument("--shots", default=",".join(str(s) for s in SHOTS))
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--tag", default="gate_r1_layer_smoke")
    ap.add_argument("--configs", default="",
                    help="comma-separated subset; the smoke test's winner only")
    a = ap.parse_args()
    objs = a.objects.split(",")
    shots = [int(x) for x in a.shots.split(",")]
    rows = []
    for obj in objs:
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(obj, shot, sp,
                                 want=set(a.configs.split(",")) if a.configs else None))
            print(f"  {obj:<12} {shot}-shot done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print("\n" + "=" * 100)
    print(f"DOWNSTREAM LAYER TEST ({len(objs)} objects)")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        print(f"\n  {met}:")
        p = df.pivot_table(index="object", columns=["shot"], values=met,
                           aggfunc="mean")
        q = df.pivot_table(index="config", columns=["object", "shot"],
                           values=met).round(1)
        print(q.to_string())
    print("\n  vs final_raw:")
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        piv = df.pivot_table(index=["object", "shot"], columns="config",
                             values=met)
        d = piv.sub(piv["final_raw"], axis=0)
        print(f"\n  {met}: mean gain over final_raw")
        print(d[["mid_raw", "midlate_raw", "mid_mean", "agg_all3"]]
              .mean().round(2).to_string())


if __name__ == "__main__":
    main()
