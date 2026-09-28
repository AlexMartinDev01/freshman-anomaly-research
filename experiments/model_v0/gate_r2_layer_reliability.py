# -*- coding: utf-8 -*-
"""
Phase R step 6c -- does a normal-only layer statistic predict per-layer AUPRO?

The augmentation route is closed: a fixed augmentation recipe fixed screw
(+23.11 AUPRO) but cost transistor (-5.16), and the normal-tail difference dT
FAILED its pre-registered ordering test as an augmentation predictor. The
diagnosis is that dT only sees the normal side and is structurally blind to an
augmented bank absorbing rotated defects.

What survives is a different, weaker claim about the ABSOLUTE normal tail:

    T_l = Q0.99 of a held-out TRAIN normal image's distances to the bank at
          layer l   (no anomalies, no labels, no test data)

Motivation: screw has mid/final = 1.449 (every other object 0.96-1.18) and is
the one object where agg_all3 collapses. Mechanism: a layer that already scores
NORMAL images high will inject false positives when averaged in unweighted.

But n=4 objects and one within-object inversion (screw's midlate has a HIGHER T
than mid yet a BETTER AUPRO) mean this is a candidate, not a signal. This run
produces the data needed to actually test it: per-layer AUPRO for all 15
objects, giving a 45-point (object, layer) set.

Pre-registered criteria, frozen BEFORE the run:

  P1  Spearman rho(T_l, AUPRO_l) over all (object, layer, shot, split) points
      <= -0.5 with p < 0.01     -- higher normal tail goes with worse AUPRO
  P2  within each (object, shot, split), the layer with the highest T_l is also
      the layer with the lowest AUPRO_l in >= 60% of cells

Both must hold. If either fails, layer-reliability gating is NOT built and this
line is closed rather than re-tuned -- the previous branch died from swapping in
one more statistic each time.

Usage: python experiments/model_v0/gate_r2_layer_reliability.py --objects a,b
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, MAPS, V1, draw_images,  # noqa: E402
                                   gt_path, l2n, load_test, load_train,
                                   nn_dist, obj_seed)
from gate_r2_aug_bank import feats_of, load_train_imgs  # noqa: E402

METRICS = os.path.join(RESULTS, "metrics")
MAPS_LR = os.path.join(RESULTS, "maps_layer_reliability")


def held_out_index(obj, shot, split, n_files, bank_idx):
    cand = [i for i in range(n_files) if i not in set(bank_idx.tolist())]
    rng = np.random.default_rng(
        np.random.SeedSequence([obj_seed(obj), shot, split, 777]))
    return int(cand[rng.integers(len(cand))])


def run(model, obj, shot, split, TESTS):
    files, folder = load_train_imgs(obj)
    _, offs = load_train("final", obj)
    idx = draw_images(offs, shot, split, obj)
    ho = held_out_index(obj, shot, split, len(files), idx)
    pil = Image.open(os.path.join(folder, files[ho])).convert("RGB")
    hf, _ = feats_of(model, pil)

    te = TESTS["final"]
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    rows = []
    for li, layer in enumerate(LAYERS):
        feats, offs_l = load_train(layer, obj)
        bank = l2n(torch.from_numpy(np.concatenate(
            [feats[offs_l[i]:offs_l[i + 1]] for i in idx]
        ).astype(np.float32)).to(DEV))
        # normal-only reliability: held-out TRAIN image scored on this layer
        qh = torch.from_numpy(hf[li].astype(np.float32)).to(DEV)
        with torch.no_grad():
            dh = nn_dist(l2n(qh), bank).cpu().numpy()
        kk = max(1, int(len(dh) * 0.01))
        T = float(np.sort(dh)[-kk:].mean())

        maps = []
        for i in range(len(te["types"])):
            q = torch.from_numpy(
                TESTS[layer]["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bank).cpu().numpy())
        sc = np.array([mean_top1p(m) for m in maps])
        d = os.path.join(MAPS_LR, obj, f"{shot}shot_s{split}", layer)
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(len(y)):
            p = os.path.join(d,
                             str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            g2 = maps[i].reshape(gh, gw)
            assert g2.shape == (gh, gw)
            np.save(p, g2)
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "layer": layer,
                     "T": T, "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="gate_r2_layer_reliability")
    a = ap.parse_args()
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1) if os.path.isdir(os.path.join(V1, d, "train")))
    shots = [int(s) for s in a.shots.split(",")]

    from src.backbones import get_model
    model = get_model("dinov2_vits14", DEV, smaller_edge_size=448)
    rows = []
    for obj in objs:
        TESTS = {l: load_test(l, obj) for l in LAYERS}
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(model, obj, shot, sp, TESTS))
        print(f"  {obj:<12} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print("\n" + "=" * 100)
    print(f"LAYER RELIABILITY: does T_l predict per-layer AUPRO?  ({len(df)} points)")
    print("=" * 100)
    rho, p = spearmanr(df["T"], df["AUPRO"])
    print(f"\n  Spearman rho(T, AUPRO) = {rho:+.3f}  (p = {p:.2e}, n = {len(df)})")
    print(f"  P1 needs rho <= -0.5 and p < 0.01  -> "
          f"{'OK' if (rho <= -0.5 and p < 0.01) else 'FAIL'}")

    ok = tot = 0
    for _, g in df.groupby(["object", "shot", "split"]):
        if g["T"].idxmax() == g["AUPRO"].idxmin():
            ok += 1
        tot += 1
    print(f"\n  P2 max-T layer == min-AUPRO layer in {ok}/{tot} cells "
          f"({100 * ok / tot:.0f}%), need >= 60%  -> "
          f"{'OK' if ok / tot >= 0.6 else 'FAIL'}")

    print("\n  per-object T by layer (mean over cells):")
    g = df.groupby(["object", "layer"])[["T", "AUPRO"]].mean().unstack()
    print(g.round(3).to_string())
    print(f"\n  -> {'GO: build reliability gating' if (rho <= -0.5 and p < 0.01 and ok / tot >= 0.6) else 'NO-GO: close this line'}")


if __name__ == "__main__":
    main()
