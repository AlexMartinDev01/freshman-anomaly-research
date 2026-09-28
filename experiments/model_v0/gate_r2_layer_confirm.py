# -*- coding: utf-8 -*-
"""
Phase R step 3 -- confirm the multi-layer result on a TRUE k-shot bank.

The layer smoke test (`gate_r1_layer_smoke.py`) found agg_all3 beating
final-only by +1.03 px_AUROC / +4.96 AUPRO on 27/30 cells, but it had two
confounds:

  1. `cache_ml`'s train split is flattened, so the bank was a fixed PATCH
     budget (shot x 1024) drawn from all train images at once -- not a k-shot
     sample. At 8-shot the budget was clamped to the 6000-patch pool.
  2. Only split 0, and only 1/4-shot.

This run fixes both: the bank is `shot` IMAGES drawn from cache_ml_img, and it
covers 1/2/4/8-shot x 3 splits on all 15 objects.

Everything else is held bit-identical to the run being confirmed: test features
come from the same `cache_ml` npz files, the maps go through the same
`pixel_metrics_binned` with pro_limit=0.05, and the same z-score-then-average
aggregation is used.

Pre-registered verdict, frozen BEFORE this run:

    CONFIRMED  if  px_AUROC mean gain >= +0.5  AND  AUPRO mean gain >= +2.0
                   AND a majority of (object, shot, split) cells improve on
                   AUPRO, AND img_AUROC mean gain >= -0.3 (no systematic
                   image-level degradation).
    NOT CONFIRMED otherwise -> the +4.96 was an artifact of the patch-budget
                   bank or of split 0, and agg_all3 does not get frozen.

Two configs only. No mid_raw, no midlate_raw, no mid_mean, no block sweep --
those were already answered by the smoke test and the point here is
confirmation, not exploration.

Usage: python experiments/model_v0/gate_r2_layer_confirm.py
"""
import argparse
import os
import sys
import time

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
ML = os.path.join(RESULTS, "cache_ml")          # test features (unchanged)
MLI = os.path.join(RESULTS, "cache_ml_img")     # per-image train features
METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_layer_confirm")
DEV = "cuda"
LAYERS = ["mid", "midlate", "final"]
CONFIGS = ["final_raw", "agg_all3"]
Q_CHUNK = 2048


def l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nn_dist(q, bank, chunk=Q_CHUNK):
    """1 - max cosine similarity, chunked over query patches."""
    out = []
    for i in range(0, len(q), chunk):
        out.append(1.0 - (q[i:i + chunk] @ bank.T).max(dim=1).values)
    return torch.cat(out)


def gt_path(obj, name):
    typ, f = str(name).split("/")
    return os.path.join(V1, obj, "ground_truth", typ, f[:-4] + "_mask.png")


def load_train(layer, obj):
    with np.load(os.path.join(MLI, layer, f"{obj}_train.npz"),
                 allow_pickle=True) as z:
        feats = np.asarray(z["feats"])
        offs = np.asarray(z["offsets"])
    return feats, offs


def obj_seed(obj):
    """Stable integer seed from the object NAME.

    Not `hash()` (randomized per process) and not the enumerate() index: the
    draw must not depend on how many objects happen to precede this one, or
    running the same object in a different shard/call would silently produce a
    different bank and a different number.
    """
    import hashlib
    return int.from_bytes(hashlib.sha256(obj.encode()).digest()[:4], "big")


def draw_images(offs, shot, split, obj):
    """Which `shot` train images this (object, shot, split) cell uses.

    A FRESH generator derived from the cell's coordinates -- never a shared
    stream consumed inside a loop. Last time a shared stream meant each layer
    drew a different patch subset, which silently made the three layers'
    banks non-matched.

    The same draw is used for all three layers here, so mid/midlate/final banks
    come from the SAME images. That is the control the smoke test lacked.
    """
    n_img = len(offs) - 1
    rng = np.random.default_rng(np.random.SeedSequence([1000 + split, shot,
                                                        obj_seed(obj)]))
    return np.sort(rng.choice(n_img, size=min(shot, n_img), replace=False))


def zscores(maps):
    """Global affine per layer, pooled over the object's test set.

    Note this cannot inflate any metric: AUROC and AUPRO are rank-based and a
    single (mu, sd) applied to every map of a layer is the same monotone
    transform for all of them. It only sets the relative weight of the layers
    in the average, which is exactly what it is for.
    """
    allv = np.concatenate([m.ravel() for m in maps])
    mu, sd = allv.mean(), allv.std() + 1e-12
    return [(m - mu) / sd for m in maps]


def load_test(layer, obj):
    """Materialize once per object: np.load on an npz is lazy, so touching
    te["feats"] per image would decompress the whole array every time."""
    with np.load(os.path.join(ML, layer, f"{obj}_test.npz"),
                 allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def run(obj, shot, split, TESTS):
    """TESTS[layer] is the materialized test dict for this object."""
    per_layer = {}
    offs = None
    rng_used = None
    for layer in LAYERS:
        feats, offs = load_train(layer, obj)
        te = TESTS[layer]
        idx = draw_images(offs, shot, split, obj)
        rng_used = idx
        bank = torch.from_numpy(
            np.concatenate([feats[offs[i]:offs[i + 1]] for i in idx])
            .astype(np.float32)).to(DEV)
        bk = l2n(bank)
        maps = []
        for i in range(len(te["types"])):
            q = torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bk).cpu().numpy())
        per_layer[layer] = maps

    Z = {k: zscores(v) for k, v in per_layer.items()}
    M = {
        "final_raw": Z["final"],
        "agg_all3": [(a + b + c) / 3 for a, b, c in
                     zip(Z["mid"], Z["midlate"], Z["final"])],
    }

    te = TESTS["final"]
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    rows = []
    for name in CONFIGS:
        maps = M[name]
        sc = np.array([mean_top1p(m) for m in maps])
        d = os.path.join(MAPS, obj, f"{shot}shot_s{split}", name)
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(len(y)):
            p = os.path.join(d,
                             str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            g2 = maps[i].reshape(gh, gw)          # ALWAYS a 2-D grid
            assert g2.shape == (gh, gw), f"map shape {g2.shape}"
            np.save(p, g2)
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "config": name,
                     "n_bank_img": len(rng_used), "n_bank_patch": int(
                         sum(int(offs[i + 1] - offs[i]) for i in rng_used)),
                     "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100})
    return rows


def report(df, shots):
    """Print the table and evaluate the PRE-REGISTERED verdict.

    Kept as one function so the sharded run and the merged run are judged by
    the exact same code -- a re-typed copy of the thresholds is exactly how a
    pre-registered criterion drifts after the fact.
    """
    print("\n" + "=" * 100)
    print(f"PHASE R step 3 -- true k-shot bank, "
          f"{df['object'].nunique()} objects x {len(shots)} shots x "
          f"{df['split'].nunique()} splits")
    print("=" * 100)
    piv = df.pivot_table(index=["object", "shot", "split"], columns="config",
                         values=["img_AUROC", "px_AUROC", "AUPRO"])
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        f, g = piv[(met, "final_raw")], piv[(met, "agg_all3")]
        d = g - f
        print(f"\n  {met}:  final_raw {f.mean():.2f}  ->  agg_all3 {g.mean():.2f}"
              f"   gain {d.mean():+.2f}   better {int((d > 0).sum())}/{len(d)} cells")
        per_obj = pd.DataFrame({"g": d}).groupby(level=0).mean()["g"]
        print("    per-object: " + " ".join(f"{o}={v:+.1f}"
                                            for o, v in per_obj.items()))

    print("\n  gain by shot:")
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        d = piv[(met, "agg_all3")] - piv[(met, "final_raw")]
        print(f"    {met:<10} " + "  ".join(
            f"{s}shot={d.xs(s, level='shot').mean():+.2f}" for s in shots))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT")
    print("=" * 100)
    d_px = (piv[("px_AUROC", "agg_all3")] - piv[("px_AUROC", "final_raw")])
    d_pr = (piv[("AUPRO", "agg_all3")] - piv[("AUPRO", "final_raw")])
    d_im = (piv[("img_AUROC", "agg_all3")] - piv[("img_AUROC", "final_raw")])
    ok = (d_px.mean() >= 0.5 and d_pr.mean() >= 2.0
          and (d_pr > 0).sum() > len(d_pr) / 2 and d_im.mean() >= -0.3)
    print(f"  px_AUROC  gain {d_px.mean():+.2f}   (need >= +0.5)   "
          f"{'OK' if d_px.mean() >= 0.5 else 'FAIL'}")
    print(f"  AUPRO     gain {d_pr.mean():+.2f}   (need >= +2.0)   "
          f"{'OK' if d_pr.mean() >= 2.0 else 'FAIL'}")
    print(f"  AUPRO     better {(d_pr > 0).sum()}/{len(d_pr)} cells "
          f"(need majority)   "
          f"{'OK' if (d_pr > 0).sum() > len(d_pr) / 2 else 'FAIL'}")
    print(f"  img_AUROC gain {d_im.mean():+.2f}   (need >= -0.3)  "
          f"{'OK' if d_im.mean() >= -0.3 else 'FAIL'}")
    print(f"\n  -> {'CONFIRMED: freeze agg_all3 as the new baseline' if ok else 'NOT CONFIRMED: do not freeze'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--objects", default="")
    ap.add_argument("--tag", default="gate_r2_layer_confirm")
    a = ap.parse_args()
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1)
        if os.path.isdir(os.path.join(V1, d, "train")))
    shots = [int(s) for s in a.shots.split(",")]
    os.makedirs(MAPS, exist_ok=True)

    rows = []
    t0 = time.time()
    for obj in objs:
        TESTS = {layer: load_test(layer, obj) for layer in LAYERS}
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(obj, shot, sp, TESTS))
        print(f"  {obj:<12} done  ({time.time() - t0:5.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    report(df, shots)


if __name__ == "__main__":
    main()
