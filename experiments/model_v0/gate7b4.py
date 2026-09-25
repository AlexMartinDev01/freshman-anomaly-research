# -*- coding: utf-8 -*-
"""
Gate 7B-4 -- decouple "more defect INSTANCES" from "more patch EXEMPLARS".

Gate 7B-3R showed detection rises with the number of distinct defect images in
the support, but that axis moved two things at once: more instances AND a larger
patch pool. This separates them.

WHY THE OBVIOUS DESIGN DOES NOT WORK. The natural design -- hold the patch
budget at 100 and vary the image count 1,2,4,8,16 -- cannot be run on MVTec AD 2.
Per-image defect patch counts (gt_frac > 0.10) are:

    object        median   max   total over the class
    wallplugs          2    28    528
    vial              22   208   5258
    sheet_metal       36   197   4257
    can                1     3     54

No single image comes close to supplying 100 patches, so at n=1 the "fixed"
budget silently collapses (wallplugs gets 3 patches, not 100) and the curve
measures patch count after all. Section 1 below prints that run as a recorded
diagnostic, plainly labelled infeasible; nothing is concluded from it.

THE FEASIBLE DESIGN. Shrink the budget until every image can supply its share.
With a budget of B patches spread over n images, each image owes ceil(B/n)
patches, so at most B images can take part. Two runs:
    B = 4,  n in {1, 2, 4}
    B = 8,  n in {1, 2, 4, 8}
At n = B every image contributes exactly one patch (maximum diversity for that
budget); at n = 1 a single image supplies all B (maximum density). The budget is
exactly B in every cell, enforced by construction and asserted at runtime.

    rising with n at fixed B  -> cross-INSTANCE diversity does work, so a
                                 generator must produce MODES, not just volume
    flat with n at fixed B    -> only the patch count ever mattered, so
                                 generating many exemplars suffices

Each cell averages REPEATS independent draws, because an AUROC built on 4
patches is noisy.

SECTION 3 is the other axis: fix the instance count at 8 images and vary how
many patches are drawn from them (local support density at constant instance
count).

Objects: wallplugs, vial, sheet_metal. can is recorded only -- 7B-3R showed it
is not support-limited, and with 54 defect patches in the entire class against a
median of 1 per image it cannot support any budgeted design.

Raw fusion S_n - S_d, strict support/eval separation, 3 splits.

Usage: python experiments/model_v0/gate7b4.py [--splits N] [--repeats N]
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
from gate7b3r import make_dist  # noqa: E402
from tail_calib import RESULTS, load_cache, split_train, mean_top1p  # noqa: E402

OBJECTS = ["wallplugs", "vial", "sheet_metal", "can"]
IMG_COUNTS = [1, 2, 4, 8, 16, -1]
PATCH_BUDGET = 100              # section 1, recorded only -- see module docstring
BUDGETS = [4, 8]                # section 2, feasible
FIXED_IMAGES = 8                # section 3
PATCH_COUNTS = [2, 4, 8, 16, 32, 64]
METRICS = os.path.join(RESULTS, "metrics")


def make_prepare():
    """prepare(obj) -> ctx, with the normal bank normalised once."""
    def prepare(obj, device="cuda"):
        tr, te = load_cache(obj)
        bank_idx, _ = split_train(tr["feats"].shape[0], 0.8, 0)
        fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                                 .astype(np.float32)).to(device)
        types = list(te["types"])
        gt = te["gt_frac"].reshape(len(types), -1)
        bad = np.where(~(np.array(types) == "good"))[0]
        good = [i for i in range(len(types)) if types[i] == "good"]
        feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to(device)
                 for i in range(len(types))]
        dist = make_dist(fbank, device)
        with torch.no_grad():
            Sn = [dist(z) for z in feats]
        return te, gt, feats, Sn, bad, good
    return prepare


prepare = make_prepare()


def run(obj, split, ctx, repeats, device="cuda"):
    te, gt, feats, Sn, bad, good = ctx
    perm = np.random.default_rng(1000 + split).permutation(len(bad))
    sup_idx = bad[perm[:len(bad) // 2]]
    ev_idx = bad[perm[len(bad) // 2:]]
    ev = good + list(ev_idx)
    y = np.array([0 if te["types"][i] == "good" else 1 for i in ev])

    # defect patches available in each support image
    per_img = {int(i): te["feats"][i].astype(np.float32)[gt[i] > 0.10]
               for i in sup_idx}
    per_img = {i: p for i, p in per_img.items() if len(p)}

    def fused(dsupp):
        dist = make_dist(dsupp.to(device), device)
        with torch.no_grad():
            Sd = [dist(feats[i]) for i in ev]
        sc = np.array([mean_top1p(Sn[i] - Sd[k]) for k, i in enumerate(ev)])
        return roc_auc_score(y, sc) * 100

    rows = []
    rng = np.random.default_rng(7 + split)

    # ---- 1: budget 100, varying images. RECORDED ONLY, NOT INFEASIBLE-SAFE ----
    for n_img in IMG_COUNTS:
        imgs = list(per_img) if n_img == -1 else rng.choice(
            list(per_img), size=min(n_img, len(per_img)), replace=False)
        pool = np.concatenate([per_img[int(i)] for i in imgs])
        k = min(PATCH_BUDGET, len(pool))
        sel = rng.choice(len(pool), size=k, replace=False)
        rows.append({"object": obj, "split": split, "exp": "1_budget100",
                     "n_images": n_img, "n_images_actual": len(imgs),
                     "n_patches": k, "budget_met": k == PATCH_BUDGET,
                     "img_AUROC": fused(torch.from_numpy(pool[sel]))})

    # ---- 2: FEASIBLE fixed budget, varying images ----
    ids = np.array(sorted(per_img))
    for B in BUDGETS:
        for n_img in [n for n in (1, 2, 4, 8) if n <= B]:
            need = int(np.ceil(B / n_img))
            cand = [i for i in ids if len(per_img[int(i)]) >= need]
            if len(cand) < n_img:
                rows.append({"object": obj, "split": split, "exp": "2_feasible",
                             "B": B, "n_images": n_img, "n_patches": -1,
                             "budget_met": False, "img_AUROC": np.nan})
                continue
            for rep in range(repeats):
                r = np.random.default_rng(hash((7 + split, B, n_img, rep)) % 2**31)
                imgs = r.choice(cand, size=n_img, replace=False)
                parts = [per_img[int(i)][r.choice(len(per_img[int(i)]), size=need,
                                                  replace=False)] for i in imgs]
                pool = np.concatenate(parts)
                assert len(pool) == B, f"budget drifted: {len(pool)} != {B}"
                rows.append({"object": obj, "split": split, "exp": "2_feasible",
                             "B": B, "n_images": n_img, "n_patches": B,
                             "budget_met": True, "rep": rep,
                             "img_AUROC": fused(torch.from_numpy(pool))})

    # ---- 2b: same as 2 but the eligible images are held fixed across n ----
    # In section 2 an image may serve at n=1 only if it holds >= B patches, but
    # at n=B any image with >= 1 patch qualifies, so the eligible set widens with
    # n and the "instance diversity" gain could be a selection effect. Here every
    # cell draws from the SAME set: images carrying at least B patches.
    for B in BUDGETS:
        cand = [i for i in ids if len(per_img[int(i)]) >= B]
        if len(cand) < B:
            continue
        for n_img in [n for n in (1, 2, 4, 8) if n <= B]:
            need = int(np.ceil(B / n_img))
            for rep in range(repeats):
                r = np.random.default_rng(hash((23 + split, B, n_img, rep))
                                          % 2**31)
                imgs = r.choice(cand, size=n_img, replace=False)
                parts = [per_img[int(i)][r.choice(len(per_img[int(i)]), size=need,
                                                  replace=False)] for i in imgs]
                pool = np.concatenate(parts)
                assert len(pool) == B, f"budget drifted: {len(pool)} != {B}"
                rows.append({"object": obj, "split": split, "exp": "2b_rich_only",
                             "B": B, "n_images": n_img, "n_patches": B,
                             "budget_met": True, "rep": rep,
                             "img_AUROC": fused(torch.from_numpy(pool))})

    # ---- 3: fixed 8 images, varying patch count ----
    imgs = rng.choice(ids, size=min(FIXED_IMAGES, len(ids)), replace=False)
    pool = np.concatenate([per_img[int(i)] for i in imgs])
    for P in PATCH_COUNTS:
        if P > len(pool):
            continue
        for rep in range(repeats):
            r = np.random.default_rng(hash((11 + split, P, rep)) % 2**31)
            sel = r.choice(len(pool), size=P, replace=False)
            rows.append({"object": obj, "split": split, "exp": "3_density",
                         "n_images": len(imgs), "n_patches": P,
                         "budget_met": True, "rep": rep,
                         "img_AUROC": fused(torch.from_numpy(pool[sel]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)
    rows = []
    cpath = os.path.join(METRICS, "gate7b4_decouple.csv")
    for o in OBJECTS:
        ctx = prepare(o)
        for sp in range(args.splits):
            rows.extend(run(o, sp, ctx, args.repeats))
            pd.DataFrame(rows).to_csv(cpath, index=False)
            print(f"  {o} split {sp} done", flush=True)
    df = pd.DataFrame(rows)

    def tab(d, idx, cols="object", vals="img_AUROC"):
        return d.pivot_table(index=idx, columns=cols, values=vals).round(1)

    print("\n" + "=" * 100)
    print("(1) BUDGET 100, VARYING IMAGES -- RECORDED, NOT INTERPRETABLE")
    print("    a single defect image supplies far fewer than 100 patches, so the")
    print("    budget is not held and this curve is really a patch-count curve")
    print("=" * 100)
    a = df[df.exp == "1_budget100"]
    print(tab(a, "n_images"))
    print("\n  patches actually achieved:")
    print(tab(a, "n_images", vals="n_patches").fillna(-1).astype(int).to_string())

    print("\n" + "=" * 100)
    print("(2) FEASIBLE FIXED BUDGET -- budget is exactly B in every cell")
    print("=" * 100)
    f = df[(df.exp == "2_feasible") & df.budget_met]
    for B in BUDGETS:
        s = f[f.B == B]
        if s.empty:
            continue
        print(f"\n  B = {B} patches "
              f"(at n = {B} each image contributes exactly 1 patch)")
        print(tab(s, "n_images").to_string())
    infeas = df[(df.exp == "2_feasible") & ~df.budget_met]
    if len(infeas):
        print("\n  infeasible cells (not enough images with the required "
              "patches):")
        print(infeas.pivot_table(index=["B", "n_images"], columns="object",
                                 values="img_AUROC", dropna=False).to_string())

    print("\n" + "=" * 100)
    print("(2b) SAME AS (2) BUT ELIGIBLE IMAGES HELD FIXED (all hold >= B "
          "patches)")
    print("     if the rise in (2) survives, it is instance diversity; if it")
    print("     vanishes, (2) was measuring which images were eligible")
    print("=" * 100)
    fb = df[(df.exp == "2b_rich_only") & df.budget_met]
    for B in BUDGETS:
        s = fb[fb.B == B]
        if s.empty:
            continue
        print(f"\n  B = {B} patches")
        print(tab(s, "n_images").to_string())

    print("\n" + "=" * 100)
    print(f"(3) FIXED {FIXED_IMAGES} IMAGES, VARYING PATCH COUNT "
          "(support density)")
    print("=" * 100)
    d = df[df.exp == "3_density"]
    print(tab(d, "n_patches"))
    print("\n  n_patches pooled from the 8 fixed images:")
    print(d.pivot_table(index="n_patches", columns="object", values="rep",
                        aggfunc="count").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
