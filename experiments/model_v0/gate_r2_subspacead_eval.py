# -*- coding: utf-8 -*-
"""
Phase R step 5 -- re-score the official SubspaceAD dumps with OUR evaluator,
paired against our kNN baseline on SubspaceAD's OWN k-shot images.

Two things make this a fair fight rather than two numbers side by side:

1. Same evaluator. SubspaceAD's AU-PRO pools negatives across the category's
   test set and its maps live at 448x448; ours feeds the raw patch grid through
   `dists2map` at the ORIGINAL image size. Re-scoring their dumped grids with
   `pixel_metrics_binned(pro_limit=0.05)` removes both differences. Their own
   CSV values are printed alongside as a cross-check on the dump path.

2. Same shots. SubspaceAD draws its k images with `random.shuffle` on one
   process-wide stream; we cannot make our own RNG produce the same draw. So
   instead of predicting the draw, this reads the chosen filenames back out of
   SubspaceAD's run.log and builds our kNN bank from exactly those images.
   The comparison is then paired cell by cell.

Usage: python experiments/model_v0/gate_r2_subspacead_eval.py
"""
import argparse
import os
import re
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
from gate_r2_layer_confirm import (DEV, LAYERS, ML, MLI, V1, gt_path,  # noqa: E402
                                   l2n, load_test, load_train, nn_dist,
                                   zscores)

SA = r"E:\work\freshman\results\subspacead_b"
METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_subspacead_b")


def parse_kshot(logpath):
    """run.log -> {category: [train/good filenames in draw order]}.

    A category header RESETS its list: re-running the same config appends to the
    same run.log, and a stale first block would otherwise be concatenated with
    the new one and trip the caller's `len(...) != shot` check.
    """
    cur, out = None, {}
    with open(logpath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"Processing Category:\s*(\S+)", line)
            if m:
                cur = m.group(1).rstrip("-").strip()
                out[cur] = []
            m = re.search(r"K-Shot image\s+\d+/\d+:\s*(\S+)", line)
            if m and cur is not None:
                out[cur].append(m.group(1).strip())
    return out


def load_train_named(layer, obj):
    """(feats, offsets, names) once per object, for every layer."""
    feats, offs = load_train(layer, obj)
    with np.load(os.path.join(MLI, layer, f"{obj}_train.npz"),
                 allow_pickle=True) as z:
        names = [str(x) for x in np.asarray(z["names"])]
    return feats, offs, names


def bank_from_names(filenames, train):
    """Bank of the named train images, in the SAME order as the cache."""
    feats, offs, names = train
    idx = [names.index(f) for f in filenames]
    return np.concatenate([feats[offs[i]:offs[i + 1]] for i in idx]).astype(
        np.float32)


def knn_agg_all3(filenames, TESTS, TRAINS):
    per = {}
    for layer in LAYERS:
        bank = bank_from_names(filenames, TRAINS[layer])
        bk = l2n(torch.from_numpy(bank).to(DEV))
        maps = []
        for i in range(len(TESTS[layer]["types"])):
            q = torch.from_numpy(
                TESTS[layer]["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bk).cpu().numpy())
        per[layer] = maps
    Z = {k: zscores(v) for k, v in per.items()}
    return [(a + b + c) / 3 for a, b, c in
            zip(Z["mid"], Z["midlate"], Z["final"])]


def evaluate(maps, te, obj, save_dir, tag):
    """Score maps with OUR evaluator. Returns the metric row."""
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    os.makedirs(save_dir, exist_ok=True)
    sc = np.array([mean_top1p(m) for m in maps])
    jobs = []
    for i in range(len(y)):
        p = os.path.join(save_dir,
                         str(te["names"][i])[:-4].replace("/", "_") + ".npy")
        g2 = np.asarray(maps[i]).reshape(gh, gw)
        assert g2.shape == (gh, gw), f"{p}: map shape {g2.shape} != {(gh, gw)}"
        np.save(p, g2)
        g = gt_path(obj, te["names"][i]) if y[i] else None
        jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
    m = pixel_metrics_binned(jobs, pro_limit=0.05)
    return {"img_AUROC": roc_auc_score(y, sc) * 100,
            "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100}


def load_sa_maps(dump_dir, obj):
    """The dumped raw grids, ordered to match our test cache's image order.

    The dump directory is shared across categories, so the filename carries the
    object as well as the defect type -- keying on the defect type alone let
    categories overwrite each other (defect names repeat across objects).
    """
    with np.load(os.path.join(ML, "final", f"{obj}_test.npz"),
                 allow_pickle=True) as z:
        names = [str(x) for x in np.asarray(z["names"])]
    out = []
    for nm in names:
        typ, f = nm.split("/")
        p = os.path.join(dump_dir, f"{obj}_{typ}_{f[:-4]}.npy")
        if not os.path.exists(p):
            return None
        out.append(np.load(p))
    return out


def report(df):
    """Print the paired comparison. One implementation, shared by every shard."""
    print("\n" + "=" * 100)
    print("SubspaceAD (official, ViT-S/14 @448) vs our kNN agg_all3 -- "
          "SAME evaluator, SAME shot images")
    print("=" * 100)
    for aug in sorted(df["aug"].unique()):
        S = df[df["aug"] == aug]
        print(f"\n  aug_count={aug}   ({len(S)} cells, "
              f"{S['object'].nunique()} objects)")
        for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
            d = S[f"sa_{met}"] - S[f"knn_{met}"]
            print(f"    {met:<10} knn {S[f'knn_{met}'].mean():6.2f}   "
                  f"subspacead {S[f'sa_{met}'].mean():6.2f}   "
                  f"delta {d.mean():+6.2f}   "
                  f"SA better {int((d > 0).sum())}/{len(d)}")
    print("\n  paired delta by shot (subspacead - knn):")
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        d = (df[f"sa_{met}"] - df[f"knn_{met}"])
        print(f"    {met:<10} " + "  ".join(
            f"{s}shot={d[df['shot'] == s].mean():+.2f}"
            for s in sorted(df["shot"].unique())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gate_r2_subspacead_b")
    ap.add_argument("--objects", default="",
                    help="comma-separated subset; shards write <tag>.csv")
    a = ap.parse_args()
    runs = []
    for d in sorted(os.listdir(os.path.join(SA, "dumps"))):
        m = re.match(r"k(\d+)_seed(\d+)_aug(\d+)$", d)
        if m:
            runs.append((d, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not runs:
        print("no dumps found"); return
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1)
        if os.path.isdir(os.path.join(V1, d, "train")))

    rows = []
    knn_cache = {}   # (obj, shot, seed) -> (shot_files, metrics)
    # object-outer: the object's features are then loaded once and reused for
    # every (shot, seed, aug) cell instead of reloaded for each.
    for obj in objs:
        TESTS = {l: load_test(l, obj) for l in LAYERS}
        TRAINS = {l: load_train_named(l, obj) for l in LAYERS}
        te = TESTS["final"]
        for tag, shot, seed, aug in runs:
            logp = os.path.join(SA, f"{tag}.log")
            kshot = parse_kshot(logp) if os.path.exists(logp) else {}
            dump_dir = os.path.join(SA, "dumps", tag)
            if len(kshot.get(obj, [])) != shot:
                print(f"  ! {tag} {obj}: log yielded {len(kshot.get(obj, []))} "
                      f"shots, expected {shot} -- skipping")
                continue
            sa = load_sa_maps(dump_dir, obj)
            if sa is None:
                print(f"  ! {tag} {obj}: incomplete dump -- skipping")
                continue
            r = {"object": obj, "shot": shot, "seed": seed, "aug": aug,
                 "n_shot_img": len(kshot[obj]),
                 "shot_files": ",".join(kshot[obj])}
            r.update({f"sa_{k}": v for k, v in evaluate(
                sa, te, obj,
                os.path.join(MAPS, tag, obj, "subspacead"), tag).items()})
            # The paired kNN bank depends only on (object, shot, seed): the
            # k-shot draw is seeded before any augmentation and the augments
            # use a separate torch RNG, so aug=0 and aug=30 must select the
            # SAME images. Recomputing it per aug would double the run for
            # nothing; the assert is what makes that reuse safe rather than
            # assumed.
            ck = (obj, shot, seed)
            if ck in knn_cache:
                prev_files, knn_metrics = knn_cache[ck]
                assert prev_files == kshot[obj], (
                    f"{tag} {obj}: shot draw differs between aug settings "
                    f"({prev_files} vs {kshot[obj]}) -- kNN reuse is invalid")
            else:
                kn = knn_agg_all3(kshot[obj], TESTS, TRAINS)
                knn_metrics = evaluate(
                    kn, te, obj,
                    os.path.join(MAPS, tag, obj, "knn_agg_all3"), tag)
                knn_cache[ck] = (kshot[obj], knn_metrics)
            r.update({f"knn_{k}": v for k, v in knn_metrics.items()})
            rows.append(r)
        print(f"  {obj:<12} done  ({len(rows)} cells)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    report(df)


if __name__ == "__main__":
    main()
