# -*- coding: utf-8 -*-
"""
Phase R step 6b -- can a NORMAL-ONLY statistic predict where augmentation helps?

The 4-object smoke gave a NO-GO for a fixed augmentation recipe: screw +23.11
AUPRO, hazelnut +6.11, tile +0.43, but transistor -5.16. So the missing piece is
not a stronger augmentation, it is a rule for deciding *per object* whether to
apply it. Before building any gating, this tests whether such a rule can exist
at all, using only target normal data.

The proposed statistic follows the leave-one-out idea, but the held-out sample
is a NORMAL image, so nothing here touches anomalies, labels, or the test set:

    hold out ONE training normal image (never in the bank)
    score its patches against the bank -> normal tail T = Q0.99 of distances
    (and the image-level analogue, mean of the top 1% -- our actual image score)

    T_base : bank built from the k drawn images
    T_aug  : bank built from those images + 30 rotations each

    delta_T = T_aug - T_base

Hypothesis (falsifiable, frozen before the run):

    H1  delta_T < 0  <=>  augmentation reduces this object's normal
                         false-positive tail <=> augmentation helps the object.
    So delta_T should ORDER the four objects the same way the measured
    AUPRO/img benefit does: screw, hazelnut (help) < tile < transistor (hurt).

Falsified if the ordering disagrees -- in which case a normal-only gating rule
of this form does not exist and the augmentation question has no cheap answer.

Runs shots {1,4} x 3 splits (6 cells/object) -- enough for a sign test, and it
reuses exactly the same bank draws as the smoke so the comparison is paired.

Usage: python experiments/model_v0/gate_r2_normal_tail_predictor.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, V1, draw_images, l2n,  # noqa: E402
                                   load_test, load_train, nn_dist, obj_seed)
from gate_r2_aug_bank import (AUG_COUNT, build_aug_bank, feats_of,  # noqa: E402
                              load_train_imgs)

METRICS = os.path.join(RESULTS, "metrics")
DEVICE = DEV


def holdout_image(obj, shot, split, n_files, in_bank):
    """A training normal image NOT among the bank images, chosen deterministically."""
    cand = [i for i in range(n_files) if i not in set(in_bank.tolist())]
    rng = np.random.default_rng(
        np.random.SeedSequence([obj_seed(obj), shot, split, 777]))
    return int(cand[rng.integers(len(cand))])


def tail_stats(model, obj, shot, split):
    files, folder = load_train_imgs(obj)
    _, offs = load_train("final", obj)
    idx = draw_images(offs, shot, split, obj)
    ho = holdout_image(obj, shot, split, len(files), idx)

    # the unaugmented bank, built exactly as the confirmation run builds it
    base_banks = {}
    for l in LAYERS:
        feats, offs_l = load_train(l, obj)
        base_banks[l] = np.concatenate(
            [feats[offs_l[i]:offs_l[i + 1]] for i in idx]).astype(np.float32)
    aug_banks = build_aug_bank(model, obj, files, folder, idx)

    pil = Image.open(os.path.join(folder, files[ho])).convert("RGB")
    fs, grid = feats_of(model, pil)

    out = {"object": obj, "shot": shot, "split": split,
           "holdout": files[ho], "in_bank": bool(ho in idx.tolist())}
    for l, f in zip(LAYERS, fs):
        for tag, bank in (("base", base_banks[l]), ("aug", aug_banks[l])):
            bk = l2n(torch.from_numpy(bank).to(DEVICE))
            q = torch.from_numpy(f.astype(np.float32)).to(DEVICE)
            with torch.no_grad():
                d = nn_dist(l2n(q), bk).cpu().numpy()
            k = max(1, int(len(d) * 0.01))
            out[f"T_{tag}_{l}"] = float(np.sort(d)[-k:].mean())
    for l in LAYERS:
        out[f"dT_{l}"] = out[f"T_aug_{l}"] - out[f"T_base_{l}"]
    out["dT_mean"] = float(np.mean([out[f"dT_{l}"] for l in LAYERS]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="screw,hazelnut,tile,transistor")
    ap.add_argument("--shots", default="1,4")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="gate_r2_normal_tail_predictor")
    a = ap.parse_args()
    objs = a.objects.split(",")
    shots = [int(s) for s in a.shots.split(",")]
    model = get_model("dinov2_vits14", DEVICE, smaller_edge_size=448)
    rows = []
    for obj in objs:
        for shot in shots:
            for sp in range(a.splits):
                rows.append(tail_stats(model, obj, shot, sp))
        print(f"  {obj:<12} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print("\n" + "=" * 100)
    print("NORMAL-ONLY TAIL: does dT predict where augmentation helps?")
    print("=" * 100)
    g = df.groupby("object")[["dT_mid", "dT_midlate", "dT_final", "dT_mean"]].mean()
    print(g.round(3).to_string())

    # measured benefit, taken from the smoke CSV rather than re-typed by hand
    sm = pd.read_csv(os.path.join(METRICS, "gate_r2_aug_bank_smoke.csv"))
    ref = pd.read_csv(os.path.join(METRICS, "gate_r2_layer_confirm.csv"))
    both = pd.concat([ref[["object", "shot", "split", "config", "AUPRO"]],
                      sm[["object", "shot", "split", "config", "AUPRO"]]])
    gp = both.pivot_table(index="object", columns="config", values="AUPRO")
    benefit = (gp["agg_all3_aug"] - gp["agg_all3"]).to_dict()

    common = [o for o in g.index if o in benefit]
    print("\n  object      dT_mean(aug-base)   measured AUPRO benefit")
    for o in common:
        print(f"  {o:<12} {g.loc[o, 'dT_mean']:+18.3f}   {benefit[o]:+8.2f}")
    if len(common) < 2:
        print("\n  (need >= 2 objects with both statistics to test the ordering)")
        return
    by_pred = sorted(common, key=lambda o: g.loc[o, "dT_mean"])
    by_true = sorted(common, key=lambda o: benefit[o])
    print(f"\n  predicted order (small dT = aug helps): {by_pred}")
    print(f"  measured  order (low benefit = hurt):   {by_true}")
    print(f"  -> ordering {'AGREES' if by_pred == by_true else 'DISAGREES'}")
    r = np.corrcoef([g.loc[o, "dT_mean"] for o in by_true],
                    [benefit[o] for o in by_true])[0, 1]
    print(f"  pearson r(dT_mean, measured AUPRO benefit) over {len(by_true)} objects = {r:+.3f}")


if __name__ == "__main__":
    main()
