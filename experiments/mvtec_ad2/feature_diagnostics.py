# -*- coding: utf-8 -*-
"""
Feature-level diagnosis: does DINOv2 cluster by scene/lighting rather than
by defect status?

For one object (default can), 1-shot seed 0:
  * extract image-level embeddings (mean of patch tokens) with frozen DINOv2-S/14
  * PCA 2D + UMAP scatter colored by scene / lighting / good-bad
  * quantitative: silhouette-style ratio = mean intra-scene distance /
    mean inter-scene distance on good images (lower = stronger scene clustering)

Outputs:
  results/mvtec_ad2/figures/feature_pca_<obj>.png
  results/mvtec_ad2/figures/feature_umap_<obj>.png   (if umap installed)
  results/mvtec_ad2/metrics/feature_clustering_<obj>.json
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from src.backbones import get_model
from ad2_pipeline import AD2_ROOT, RESULTS_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OBJ = sys.argv[1] if len(sys.argv) > 1 else "can"
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def parse_name(name):
    m = re.match(r"(\d{3})_(.+)\.png", name)
    return int(m.group(1)), m.group(2)


def img_embedding(model, path):
    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor, grid = model.prepare_image(image)
    feats = model.extract_features(tensor)
    return feats.mean(axis=0)  # mean-pooled patch tokens


def main():
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)
    base = os.path.join(AD2_ROOT, OBJ, "test_public")
    rows = []
    for typ in ("good", "bad"):
        folder = os.path.join(base, typ)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".png"):
                continue
            scene, light = parse_name(f)
            emb = img_embedding(model, os.path.join(folder, f))
            rows.append((emb, typ, scene, light))
    X = np.stack([r[0] for r in rows])
    typ = np.array([r[1] for r in rows])
    scene = np.array([r[2] for r in rows])
    light = np.array([r[3] for r in rows])
    print(f"{OBJ}: {len(X)} images embedded, dim={X.shape[1]}")

    # ---- quantitative scene clustering on GOOD images ----
    g = X[typ == "good"]
    gscene = scene[typ == "good"]
    Xn = g / np.linalg.norm(g, axis=1, keepdims=True)
    D = 1 - Xn @ Xn.T
    intra = [D[i, gscene == s].mean() for i, s in enumerate(gscene)
             if (gscene == s).sum() > 1 and (gscene == s)[i]]
    intra = np.mean([d for d in intra if d > 0])
    inter = D[np.triu_indices(len(g), 1)]
    inter_other = [D[i][gscene != gscene[i]].mean() for i in range(len(g))]
    inter_other = np.mean(inter_other)
    ratio = intra / inter_other
    print(f"intra-scene dist={intra:.4f}  inter-scene dist={inter_other:.4f}  "
          f"ratio={ratio:.3f} (<<1 = strong scene clustering)")

    # good-vs-bad direction: cosine distance of bad images to good centroid
    centroid = Xn.mean(axis=0) if False else (g.mean(axis=0))
    centroid_n = centroid / np.linalg.norm(centroid)
    alln = X / np.linalg.norm(X, axis=1, keepdims=True)
    d_to_centroid = 1 - alln @ centroid_n
    from sklearn.metrics import roc_auc_score
    emb_auroc = roc_auc_score((typ == "bad").astype(int), d_to_centroid)

    out = {
        "object": OBJ, "intra_scene": float(intra), "inter_scene": float(inter_other),
        "scene_ratio": float(ratio), "embedding_goodbad_AUROC": float(emb_auroc),
    }
    with open(os.path.join(RESULTS_ROOT, "metrics",
                           f"feature_clustering_{OBJ}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figures: PCA + t-SNE ----
    pca = PCA(n_components=2).fit_transform(X)
    for reducer, name in [(pca, "pca")]:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
        sc0 = axs[0].scatter(reducer[:, 0], reducer[:, 1], c=scene, cmap="tab20", s=18)
        axs[0].set_title(f"{OBJ}: colored by SCENE")
        plt.colorbar(sc0, ax=axs[0])
        lights = sorted(set(light))
        lc = np.array([lights.index(l) for l in light])
        sc1 = axs[1].scatter(reducer[:, 0], reducer[:, 1], c=lc, cmap="Set1", s=18)
        axs[1].set_title("colored by LIGHTING")
        axs[1].legend(handles=[plt.Line2D([], [], marker='o', ls='', color=plt.cm.Set1(i / max(len(lights) - 1, 1)), label=l) for i, l in enumerate(lights)], fontsize=7)
        axs[2].scatter(reducer[typ == "good", 0], reducer[typ == "good", 1],
                       c="steelblue", s=18, label="good")
        axs[2].scatter(reducer[typ == "bad", 0], reducer[typ == "bad", 1],
                       c="red", s=18, label="bad")
        axs[2].set_title("colored by GOOD/BAD")
        axs[2].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"feature_{name}_{OBJ}.png"), dpi=110)
        plt.close()
    print("figures saved:", FIG_DIR)


if __name__ == "__main__":
    main()
