# -*- coding: utf-8 -*-
"""
Feature-level mechanism diagnosis (the closing experiment).

The detection experiments say *that* few-shot anomaly detection collapses on
some AD2 categories. This script says *why*, by measuring three kinds of
feature-space perturbation in the SAME frozen DINOv2 space the detector uses,
with the SAME 1-shot memory bank (train/good[0], seed 0):

    D_light   legitimate lighting change   (ref scene, other lighting)
    D_scene   legitimate scene change      (other scene, ref lighting)
    D_defect  real defect                  (GT>10% patches of anomalous images)

Everything is a per-patch 1-NN cosine distance to the 1-shot reference bank,
i.e. exactly the quantity AnomalyDINO thresholds. The central question:

    is the feature perturbation caused by LEGITIMATE normal variation already
    larger than the perturbation caused by a REAL defect?

Headline statistics
  * mean/median/p95 of each group
  * nuisance ratio   D_scene / D_defect  (>= 1 means normal variation wins)
  * AUROC(defect patches vs scene-shifted normal patches) -- <= 0.5 means the
    feature space cannot separate a defect from a legitimate scene change
  * within-image AUROC (defect vs clean in the SAME image) -- the matched
    contrast, which is what pixel-level metrics measure
  * image-level AUROC reproduction (mean-top-1% of each group's images)

Outputs
  results/mvtec_ad2/metrics/feature_diagnostics.csv        per object summary
  results/mvtec_ad2/metrics/feature_patchdist_<obj>.csv    per-patch distances
  results/mvtec_ad2/figures/feature_box_<obj>.png          group distributions
  results/mvtec_ad2/figures/feature_pca_<obj>.png          PCA of image embeds

Usage: python experiments/mvtec_ad2/feature_diagnostics.py [obj ...]
"""
import os
import re
import sys

import cv2
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from src.backbones import get_model
from ad2_pipeline import (AD2_ROOT, RESULTS_ROOT, build_memory_bank,
                          extract_patch_features, list_png, mean_top1p)
from patch_contrast import gt_to_patch_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# can/wallplugs = collapse, vial = coverage-limited, sheet_metal = saturated,
# fabric/rice = healthy controls (baseline works there).
OBJECTS = ["can", "wallplugs", "vial", "sheet_metal", "fabric", "rice"]
SHOT, SEED = 1, 0

METRICS_DIR = os.path.join(RESULTS_ROOT, "metrics")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")

GROUP_LABEL = {
    "same": "normal: same scene+light",
    "light": "normal: D_light (lighting only)",
    "scene": "normal: D_scene (scene only)",
    "scene_light": "normal: D_scene+light",
    "defect": "DEFECT patches",
    "clean": "clean patches in bad imgs",
}
GROUPS = list(GROUP_LABEL)


def parse_name(name):
    m = re.match(r"(\d{3})_(.+)\.png", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def patch_stats(obj, typ, image, scene, light, group, d):
    """Per-(image, group) summary of the patch distances -- the saved artifact."""
    return {
        "object": obj, "type": typ, "image": image, "scene": scene,
        "light": light, "group": group, "n_patches": len(d),
        "mean": float(np.mean(d)), "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
        "max": float(np.max(d)), "img_score": mean_top1p(d),
    }


def patch_dists(model, index, path):
    """Per-patch 1-NN cosine distance to the memory bank."""
    feats, grid, _ = extract_patch_features(model, path)
    feats = feats.astype("float32")
    faiss.normalize_L2(feats)
    d, _ = index.search(feats, k=1)
    return d.squeeze() / 2.0, grid


def run_object(model, obj):
    ref_scene, ref_light = "000", "regular"   # train/good[0] == 000_regular.png
    index, ref_files = build_memory_bank(model, obj, SHOT, SEED)
    assert os.path.basename(ref_files[0]) == f"{ref_scene}_{ref_light}.png", ref_files
    print(f"\n=== {obj} === bank={ref_files}  ref=({ref_scene},{ref_light})")

    patch_rows = []
    dist_by_group = {g: [] for g in GROUPS}
    img_scores = []          # (group, mean_top1p) for image-level reproduction

    # ---------- good images: normal variation ----------
    good_dir = os.path.join(AD2_ROOT, obj, "test_public", "good")
    for f in list_png(good_dir):
        scene, light = parse_name(f)
        d, _ = patch_dists(model, index, os.path.join(good_dir, f))
        if scene == ref_scene and light == ref_light:
            g = "same"
        elif scene == ref_scene:
            g = "light"
        elif light == ref_light:
            g = "scene"
        else:
            g = "scene_light"
        dist_by_group[g].append(d)
        img_scores.append((g, mean_top1p(d), f))
        patch_rows.append(patch_stats(obj, "good", f, scene, light, g, d))

    # ---------- bad images: defect vs clean ----------
    bad_dir = os.path.join(AD2_ROOT, obj, "test_public", "bad")
    gt_dir = os.path.join(AD2_ROOT, obj, "test_public", "ground_truth", "bad")
    within_auroc, matched_defect = [], []
    for f in list_png(bad_dir):
        scene, light = parse_name(f)
        path = os.path.join(bad_dir, f)
        d, grid = patch_dists(model, index, path)
        img = cv2.imread(path)
        gt = os.path.join(gt_dir, f[:-4] + "_mask.png")
        if not os.path.exists(gt):
            continue
        frac = gt_to_patch_grid(gt, img.shape[:2], grid)
        dmask, cmask = frac > 0.10, frac == 0.0
        if dmask.sum() == 0 or cmask.sum() == 0:
            continue
        dmap = d.reshape(grid)          # d is flat; masks are (grid_h, grid_w)
        d_def, d_clean = dmap[dmask], dmap[cmask]
        dist_by_group["defect"].append(d_def)
        dist_by_group["clean"].append(d_clean)
        img_scores.append(("bad", mean_top1p(d), f))
        patch_rows.append(patch_stats(obj, "bad", f, scene, light, "defect", d_def))
        patch_rows.append(patch_stats(obj, "bad", f, scene, light, "clean", d_clean))
        within_auroc.append(roc_auc_score(
            np.r_[np.ones(len(d_def)), np.zeros(len(d_clean))],
            np.r_[d_def, d_clean]))
        # matched: defect in an image that shares scene AND lighting with the ref
        if scene == ref_scene and light == ref_light:
            matched_defect.append(d_def)

    flat = {g: np.concatenate(v) if v else np.array([])
            for g, v in dist_by_group.items()}

    # ---------- floor reference ----------
    # Scene ids DO correspond between train/ and test_public/ (verified visually:
    # same physical scene), but they are different captures, so this group is
    # the irreducible capture-noise floor rather than ~0.
    same_d = flat["same"]
    print(f"  capture-noise floor (same scene+light): median d = "
          f"{np.median(same_d):.4f}  mean = {np.mean(same_d):.4f}")

    # ---------- headline statistics ----------
    med = {g: float(np.median(v)) for g, v in flat.items() if len(v)}
    mean = {g: float(np.mean(v)) for g, v in flat.items() if len(v)}
    p95 = {g: float(np.percentile(v, 95)) for g, v in flat.items() if len(v)}

    def auc(pos, neg):
        if len(flat[pos]) == 0 or len(flat[neg]) == 0:
            return np.nan
        return roc_auc_score(np.r_[np.ones(len(flat[pos])),
                                   np.zeros(len(flat[neg]))],
                             np.r_[flat[pos], flat[neg]])

    d_def_all = flat["defect"]
    normals_all = np.concatenate([flat[g] for g in
                                  ("same", "light", "scene", "scene_light")])
    matched_all = (np.concatenate(matched_defect) if matched_defect
                   else np.array([]))

    # ---------- tail comparison: defect vs the TOP of the normal distribution ----
    # The image score is the mean of the top-1% patch distances, so what decides
    # detectability is the tail, not the mean. frac_defect_below_normal_pXX is
    # the fraction of REAL defect patches that a threshold at the normal pXX
    # would not fire on.
    thr = {q: float(np.percentile(normals_all, q)) for q in (95, 99, 99.9)}
    frac_below = {q: float((d_def_all < thr[q]).mean()) for q in thr}

    res = {
        "normal_p95": thr[95], "normal_p99": thr[99], "normal_p999": thr[99.9],
        "defect_p50": float(np.median(d_def_all)),
        "defect_p95": float(np.percentile(d_def_all, 95)),
        "frac_defect_below_normal_p95": frac_below[95],
        "frac_defect_below_normal_p99": frac_below[99],
        "object": obj, "shot": SHOT, "seed": SEED,
        "n_patches_defect": len(d_def_all),
        # capture-noise floor: same scene AND same lighting, different capture
        "mean_same": mean.get("same", np.nan),
        "median_same": med.get("same", np.nan),
        "median_defect": med.get("defect", np.nan),
        "median_light": med.get("light", np.nan),
        "median_scene": med.get("scene", np.nan),
        "median_scene_light": med.get("scene_light", np.nan),
        "median_clean": med.get("clean", np.nan),
        "mean_defect": mean.get("defect", np.nan),
        "mean_light": mean.get("light", np.nan),
        "mean_scene": mean.get("scene", np.nan),
        "p95_defect": p95.get("defect", np.nan),
        "p95_light": p95.get("light", np.nan),
        "p95_scene": p95.get("scene", np.nan),
        # nuisance ratio: >1 => legitimate variation perturbs more than a defect
        "ratio_light_defect": mean.get("light", np.nan) / mean.get("defect", np.nan),
        "ratio_scene_defect": mean.get("scene", np.nan) / mean.get("defect", np.nan),
        # separability of a real defect from legitimate normal variation
        "AUROC_defect_vs_light": auc("defect", "light"),
        "AUROC_defect_vs_scene": auc("defect", "scene"),
        "AUROC_defect_vs_scene_light": auc("defect", "scene_light"),
        "AUROC_defect_vs_clean": auc("defect", "clean"),
        "AUROC_defect_vs_allnormal": roc_auc_score(
            np.r_[np.ones(len(d_def_all)), np.zeros(len(normals_all))],
            np.concatenate([d_def_all, normals_all])),
        # matched contrast: same scene AND same lighting as the reference
        "n_matched_defect_patches": len(matched_all),
        "AUROC_matched_defect_vs_same": roc_auc_score(
            np.r_[np.ones(len(matched_all)), np.zeros(len(same_d))],
            np.concatenate([matched_all, same_d]))
        if len(matched_all) and len(same_d) else np.nan,
        "within_image_AUROC": float(np.mean(within_auroc)) if within_auroc else np.nan,
    }

    # ---------- image-level reproduction ----------
    sc = pd.DataFrame(img_scores, columns=["group", "score", "name"])
    y_bad = (sc.group == "bad").astype(int).to_numpy()
    for g in ("light", "scene", "scene_light"):
        m = sc.group.isin([g, "bad"])
        if m.sum() and y_bad[m.to_numpy()].sum() and (~y_bad[m.to_numpy()]).sum():
            res[f"img_AUROC_bad_vs_{g}"] = roc_auc_score(
                y_bad[m.to_numpy()], sc.score[m].to_numpy())
    allgood = sc.group != "bad"
    res["img_AUROC_bad_vs_allgood"] = roc_auc_score(
        (~allgood).astype(int), sc.score.to_numpy())

    pd.DataFrame(patch_rows).to_csv(
        os.path.join(METRICS_DIR, f"feature_patchdist_{obj}.csv"), index=False)
    return res, flat, sc


def plot_groups(obj, flat, sc, out_png):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    data = [flat[g] for g in GROUPS if len(flat[g])]
    labels = [GROUP_LABEL[g].replace(": ", ":\n") for g in GROUPS if len(flat[g])]
    bp = axs[0].boxplot(data, tick_labels=labels, showfliers=False,
                        patch_artist=True)
    cols = {"defect": "#d62728", "clean": "#ff9896"}
    for patch, g in zip(bp["boxes"], [g for g in GROUPS if len(flat[g])]):
        patch.set_facecolor(cols.get(g, "#aec7e8"))
    axs[0].set_ylabel("1-NN cosine distance to 1-shot reference")
    axs[0].set_title(f"{obj}: per-patch distance by group")
    axs[0].tick_params(axis="x", labelsize=7)
    plt.setp(axs[0].get_xticklabels(), rotation=20, ha="right")

    # NB: good images carry their normal-variation group name, not "good"
    good_mask = (sc.group != "bad").to_numpy()
    axs[1].hist(sc.score[good_mask], bins=30, alpha=0.55, label="good",
                color="#1f77b4", density=True)
    axs[1].hist(sc.score[~good_mask], bins=30, alpha=0.55, label="bad",
                color="#d62728", density=True)
    axs[1].set_xlabel("image score (mean top-1% patch distance)")
    axs[1].set_ylabel("density")
    axs[1].set_title(f"{obj}: good vs bad image scores")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def plot_pca(obj, model, out_png):
    """PCA of mean-pooled image embeddings, coloured by scene / light / good-bad."""
    base = os.path.join(AD2_ROOT, obj, "test_public")
    embs, typ, scene, light = [], [], [], []
    for t in ("good", "bad"):
        for f in list_png(os.path.join(base, t)):
            feats, _, _ = extract_patch_features(model, os.path.join(base, t, f))
            embs.append(feats.mean(axis=0))
            typ.append(t)
            s, l = parse_name(f)
            scene.append(s)
            light.append(l)
    X = np.stack(embs)
    Y = PCA(n_components=2).fit_transform(X)
    typ, scene, light = np.array(typ), np.array(scene), np.array(light)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    sc0 = axs[0].scatter(Y[:, 0], Y[:, 1], c=np.array([int(s) for s in scene]),
                         cmap="tab20", s=16)
    axs[0].set_title(f"{obj}: by SCENE")
    plt.colorbar(sc0, ax=axs[0])
    lights = sorted(set(light))
    sc1 = axs[1].scatter(Y[:, 0], Y[:, 1],
                         c=[lights.index(l) for l in light], cmap="Set1", s=16)
    axs[1].set_title("by LIGHTING")
    axs[1].legend(handles=[plt.Line2D([], [], marker='o', ls='', label=l,
                                      color=plt.cm.Set1(i / max(len(lights) - 1, 1)))
                           for i, l in enumerate(lights)], fontsize=7)
    axs[2].scatter(Y[typ == "good", 0], Y[typ == "good", 1], c="steelblue",
                   s=16, label="good")
    axs[2].scatter(Y[typ == "bad", 0], Y[typ == "bad", 1], c="red", s=16,
                   label="bad")
    axs[2].set_title("by GOOD/BAD")
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def main():
    objects = sys.argv[1:] or OBJECTS
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    rows = []
    for obj in objects:
        res, flat, sc = run_object(model, obj)
        rows.append(res)
        plot_groups(obj, flat, sc, os.path.join(FIG_DIR, f"feature_box_{obj}.png"))
        plot_pca(obj, model, os.path.join(FIG_DIR, f"feature_pca_{obj}.png"))
        print(f"  within-image AUROC={res['within_image_AUROC']:.3f}  "
              f"AUROC(defect vs scene)={res['AUROC_defect_vs_scene']:.3f}  "
              f"ratio_scene_defect={res['ratio_scene_defect']:.2f}")
        print(f"  normal tail p95={res['normal_p95']:.3f} "
              f"p99={res['normal_p99']:.3f}  vs defect p50={res['defect_p50']:.3f} "
              f"p95={res['defect_p95']:.3f}  ->  "
              f"{100*res['frac_defect_below_normal_p99']:.0f}% of defect patches "
              f"sit BELOW the normal p99")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "feature_diagnostics.csv"), index=False)

    show = df[["object", "mean_same", "mean_defect", "mean_light", "mean_scene",
               "ratio_light_defect", "ratio_scene_defect",
               "normal_p99", "defect_p50", "defect_p95",
               "frac_defect_below_normal_p99",
               "AUROC_defect_vs_light", "AUROC_defect_vs_scene",
               "AUROC_defect_vs_clean", "within_image_AUROC",
               "img_AUROC_bad_vs_allgood"]].round(3)
    print("\n=== Feature-level diagnosis (1-shot, seed 0, DINOv2-S/14 @448) ===")
    print(show.to_string(index=False))
    print("\nsaved:", os.path.join(METRICS_DIR, "feature_diagnostics.csv"))


if __name__ == "__main__":
    main()
