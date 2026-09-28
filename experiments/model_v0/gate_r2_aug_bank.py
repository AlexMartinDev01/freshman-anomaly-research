# -*- coding: utf-8 -*-
"""
Phase R step 6 -- does normal augmentation rescue the bank? (4-object smoke)

The SubspaceAD natural experiment is the strongest single clue from step 4/5:
same backbone, same k-shot images, and simply rotating each exemplar 30 times
moved screw from AUPRO 40.9 -> 65.7 (1-shot) and 48.1 -> 75.0 (4-shot). screw is
also the one object where agg_all3 collapses. The parsimonious reading is:

    screw's failure is not the scoring family, it is that k normal images
    do not cover the object's normal appearance manifold.

This tests that directly on OUR detector. Nothing else changes: same layers,
same scoring, same evaluator, same shots. Only the reference bank is rebuilt as

    k images -> 30 random rotations each -> their patches -> kNN bank

Augmentation matches SubspaceAD's exactly (RandomRotation on [0, 345) degrees,
fill 0, original always kept) so the comparison is against the same natural
experiment rather than a new one. Angles are drawn from a per-(object, image,
aug) SeedSequence and applied with functional.rotate, not torchvision's global
RNG, so the bank is reproducible regardless of call order.

Pre-registered criteria, frozen BEFORE the smoke run:

  S1  screw recovers:  agg_all3+aug vs agg_all3  gives  img_AUROC >= +5
                      AND AUPRO >= +10
  S2  no collateral:   no other object's AUPRO drops more than 5 below agg_all3
  S3  guard holds:     over the smoke objects, mean img_AUROC of agg_all3+aug
                      is within -0.3 of final_raw

Smoke on 4 objects first (screw = the failure; tile = multi-layer's best
localization win; transistor = hard low-contrast; hazelnut = an agg_all3
regression). Full 15 objects only if the smoke shows a clear direction.

Usage:
    python experiments/model_v0/gate_r2_aug_bank.py --objects screw,tile,transistor,hazelnut
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from pixel_metrics_binned import pixel_metrics_binned  # noqa: E402
from tail_calib import RESULTS, mean_top1p  # noqa: E402
from gate_r2_layer_confirm import (DEV, LAYERS, MAPS, MLI, V1, draw_images,  # noqa: E402
                                   gt_path, l2n, load_test, nn_dist, obj_seed,
                                   zscores)

BLOCK = {"mid": 5, "midlate": 8, "final": 11}
METRICS = os.path.join(RESULTS, "metrics")
MAPS_AUG = os.path.join(RESULTS, "maps_layer_aug")
AUG_COUNT = 30
MAX_DEG = 345.0


def load_train_imgs(obj):
    d = os.path.join(V1, obj, "train", "good")
    return sorted(f for f in os.listdir(d) if f.endswith(".png")), d


def feats_of(model, pil_img):
    """One forward -> all three layers, matching the cached block choice."""
    t, grid = model.prepare_image(np.array(pil_img))
    with torch.no_grad():
        toks = model.model.get_intermediate_layers(
            t.unsqueeze(0).to(model.device),
            n=[BLOCK[l] for l in LAYERS])
    return [tk.squeeze(0).cpu().numpy().astype(np.float32) for tk in toks], grid


def aug_angle(obj, img_idx, a):
    """Explicit angle per (object, image, aug) -- never a shared global stream."""
    rng = np.random.default_rng(
        np.random.SeedSequence([obj_seed(obj), img_idx, a]))
    return float(rng.uniform(0.0, MAX_DEG))


def build_aug_bank(model, obj, files, folder, idx):
    """Bank of the drawn images plus AUG_COUNT rotations each, per layer."""
    banks = {l: [] for l in LAYERS}
    for i in idx:
        p = os.path.join(folder, files[i])
        pil = Image.open(p).convert("RGB")
        for a in range(AUG_COUNT + 1):          # a == 0 keeps the original
            img = pil if a == 0 else TF.rotate(pil, aug_angle(obj, int(i), a),
                                               fill=0)
            fs, grid = feats_of(model, img)
            for l, f in zip(LAYERS, fs):
                assert f.shape[0] == int(np.prod(grid)), "grid/feature mismatch"
                banks[l].append(f)
    return {l: np.concatenate(v).astype(np.float32) for l, v in banks.items()}


def run(model, obj, shot, split, TESTS):
    files, folder = load_train_imgs(obj)
    feats0, offs = None, None
    # the SAME draw as the confirmation run -- same function, same seeds
    from gate_r2_layer_confirm import load_train
    _, offs = load_train("final", obj)
    idx = draw_images(offs, shot, split, obj)
    t0 = time.time()
    banks = build_aug_bank(model, obj, files, folder, idx)
    build_s = time.time() - t0

    per = {}
    for l in LAYERS:
        bk = l2n(torch.from_numpy(banks[l]).to(DEV))
        maps = []
        for i in range(len(TESTS[l]["types"])):
            q = torch.from_numpy(
                TESTS[l]["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bk).cpu().numpy())
        per[l] = maps
    Z = {k: zscores(v) for k, v in per.items()}
    M = {
        "final_raw_aug": Z["final"],
        "agg_all3_aug": [(a + b + c) / 3 for a, b, c in
                         zip(Z["mid"], Z["midlate"], Z["final"])],
    }

    te = TESTS["final"]
    y = np.array([1 if t == "bad" else 0 for t in te["types"]])
    gh, gw = te["gt_frac"].shape[1], te["gt_frac"].shape[2]
    rows = []
    for name, maps in M.items():
        sc = np.array([mean_top1p(m) for m in maps])
        d = os.path.join(MAPS_AUG, obj, f"{shot}shot_s{split}", name)
        os.makedirs(d, exist_ok=True)
        jobs = []
        for i in range(len(y)):
            p = os.path.join(d,
                             str(te["names"][i])[:-4].replace("/", "_") + ".npy")
            g2 = maps[i].reshape(gh, gw)
            assert g2.shape == (gh, gw), f"map shape {g2.shape}"
            np.save(p, g2)
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "config": name,
                     "n_bank_img": len(idx) * (AUG_COUNT + 1),
                     "n_bank_patch": int(sum(banks[l].shape[0] for l in LAYERS)
                                         / len(LAYERS)),
                     "build_s": round(build_s, 1),
                     "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="screw,tile,transistor,hazelnut")
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--tag", default="gate_r2_aug_bank_smoke")
    a = ap.parse_args()
    objs = a.objects.split(",")
    shots = [int(s) for s in a.shots.split(",")]
    os.makedirs(MAPS_AUG, exist_ok=True)

    model = get_model("dinov2_vits14", DEV, smaller_edge_size=448)
    rows = []
    t0 = time.time()
    for obj in objs:
        TESTS = {l: load_test(l, obj) for l in LAYERS}
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(model, obj, shot, sp, TESTS))
        print(f"  {obj:<12} done ({time.time() - t0:5.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    aug = pd.DataFrame(rows)
    aug.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    ref = pd.read_csv(os.path.join(METRICS, "gate_r2_layer_confirm.csv"))
    ref = ref[ref.object.isin(objs)][["object", "shot", "split", "config",
                                      "img_AUROC", "px_AUROC", "AUPRO"]]
    df = pd.concat([ref, aug.drop(columns=["build_s"])], ignore_index=True)
    df.to_csv(os.path.join(METRICS, f"{a.tag}_merged.csv"), index=False)

    piv = df.pivot_table(index=["object", "shot", "split"], columns="config",
                         values=["img_AUROC", "px_AUROC", "AUPRO"])
    print("\n" + "=" * 100)
    print("NORMAL AUGMENTATION BANK -- 4-object smoke (30 rotations/exemplar)")
    print("=" * 100)
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        print(f"\n  {met} (mean over shots x splits)")
        g = df.groupby(["object", "config"])[met].mean().unstack()
        base = g["agg_all3"]
        for c in ["final_raw", "final_raw_aug", "agg_all3", "agg_all3_aug"]:
            print(f"    {c:<16} " + " ".join(
                f"{o}={g.loc[o, c]:6.2f}" for o in objs))
        print("    " + "-" * 60)
        print(f"    {'final_raw_aug-final_raw':<28} " + " ".join(
            f"{o}={g.loc[o, 'final_raw_aug'] - g.loc[o, 'final_raw']:+6.2f}"
            for o in objs))
        print(f"    {'agg_aug - agg':<28} " + " ".join(
            f"{o}={g.loc[o, 'agg_all3_aug'] - base[o]:+6.2f}" for o in objs))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED SMOKE CRITERIA")
    print("=" * 100)
    sc = df[df.object == "screw"].groupby("config")[["img_AUROC", "AUPRO"]].mean()
    d_img = sc.loc["agg_all3_aug", "img_AUROC"] - sc.loc["agg_all3", "img_AUROC"]
    d_pr = sc.loc["agg_all3_aug", "AUPRO"] - sc.loc["agg_all3", "AUPRO"]
    s1 = d_img >= 5.0 and d_pr >= 10.0
    print(f"  S1 screw recovers  img {d_img:+.2f} (need >= +5)   "
          f"AUPRO {d_pr:+.2f} (need >= +10)   {'OK' if s1 else 'FAIL'}")
    ga = df.groupby(["object", "config"])["AUPRO"].mean().unstack()
    drops = {o: ga.loc[o, "agg_all3_aug"] - ga.loc[o, "agg_all3"] for o in objs}
    others = [(v, o) for o, v in drops.items() if o != "screw"]
    if not others:
        print("  S2 no collateral   (only screw in this run -- not evaluable)")
        s2 = False
    else:
        worst = min(others)
        s2 = worst[0] >= -5.0
        print(f"  S2 no collateral   worst non-screw AUPRO delta {worst[0]:+.2f} "
              f"({worst[1]})  (need >= -5)   {'OK' if s2 else 'FAIL'}")
    mi = df.groupby(["object", "config"])["img_AUROC"].mean().unstack()
    d3 = (mi["agg_all3_aug"] - mi["final_raw"]).mean()
    s3 = d3 >= -0.3
    print(f"  S3 guard holds     mean(agg_all3_aug - final_raw) {d3:+.2f} "
          f"(need >= -0.3)   {'OK' if s3 else 'FAIL'}")
    print(f"\n  -> {'GO: run full 15 objects' if (s1 and s2 and s3) else 'NO-GO: do not scale up'}")


if __name__ == "__main__":
    main()
