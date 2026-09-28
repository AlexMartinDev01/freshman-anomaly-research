# -*- coding: utf-8 -*-
"""
Phase R step 4 -- kNN memory bank vs PCA normal subspace, paired.

The question is narrow and is the one SubspaceAD makes worth asking:

    given the SAME frozen features, the SAME k-shot bank images, the SAME
    splits, and the SAME evaluator, does scoring patches by PCA reconstruction
    residual beat scoring them by 1-NN distance to the bank?

SubspaceAD's own pipeline answers a *different*, broader question: it resizes
to exactly 448x448, reads pre-LayerNorm HF hidden_states, fuses layers by
averaging raw features, fits PCA on the k images PLUS 30 random-rotation
augmentations each, and computes AU-PRO with globally pooled negatives. Every
one of those is a difference that would be confounded with "PCA vs kNN" if we
just ran their CLI and compared to our number.

So this script removes them by construction: it reads the same `cache_ml_img`
bank and the same `cache_ml` test features as `gate_r2_layer_confirm.py`, draws
the bank with the SAME `draw_images` call, and scores with the same
`pixel_metrics_binned` at pro_limit=0.05. Only the score function changes.

Configs:
    knn_agg_all3   z(mid)+z(midlate)+z(final) of 1-NN distance maps  (= baseline)
    pca_perlayer   the same three layers, each scored by its own PCA residual,
                   z-scored and averaged the same way
    pca_fused      raw features averaged across the three layers first, then a
                   single PCA residual -- SubspaceAD's --agg_method mean recipe
                   transplanted onto our representation

Score (SubspaceAD `reconstruction`, drop_k=0), for PCA params (mu, C) with C
orthonormal:
    s(x) = || x - mu - C C^T (x - mu) ||^2  =  || (I - C C^T)(x - mu) ||^2
k is chosen as the smallest k with cumulative explained variance >= ev.

Note this script must not be read as a verdict on SubspaceAD -- the faithful
run of their code is separate. It answers only whether the scoring mechanism
itself is where the gap is.

Usage: python experiments/model_v0/gate_r2_knn_vs_pca.py
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
from gate_r2_layer_confirm import (LAYERS, V1, draw_images, gt_path,  # noqa: E402
                                   l2n, load_test, load_train, nn_dist,
                                   zscores)

METRICS = os.path.join(RESULTS, "metrics")
MAPS = os.path.join(RESULTS, "maps_knn_vs_pca")
DEV = "cuda"
CONFIGS = ["knn_agg_all3", "pca_perlayer", "pca_fused"]


def pca_fit(bank, ev=0.99):
    """PCA parameters in SubspaceAD's convention: center, no scaling."""
    mu = bank.mean(axis=0)
    B = (bank - mu).astype(np.float64)
    C = (B.T @ B) / (len(B) - 1)
    evals, evecs = np.linalg.eigh(C)          # ascending
    evals = evals[::-1].copy()
    evecs = evecs[:, ::-1].copy()
    cum = np.cumsum(evals) / evals.sum()
    k = int(np.searchsorted(cum, ev) + 1)     # same rule as PCAModel._select_k
    return mu, evecs[:, :k].astype(np.float32), k


def pca_residual(q, mu, P):
    """s(x) = ||(I - P P^T)(x - mu)||^2, per patch. SubspaceAD scoring.py:97."""
    X0 = q - mu
    R = X0 - (X0 @ P) @ P.T
    return np.einsum("ij,ij->i", R, R)


def scores_for(bank_np, queries, ev):
    """Return (per-query score maps) for one layer given one bank."""
    mu, P, k = pca_fit(bank_np, ev)
    out = [pca_residual(q, mu, P) for q in queries]
    return out, k


def run(obj, obj_idx, shot, split, TESTS):
    per_layer = {}
    for layer in LAYERS:
        feats, offs = load_train(layer, obj)
        te = TESTS[layer]
        idx = draw_images(offs, shot, split, obj_idx)
        bank_np = np.concatenate([feats[offs[i]:offs[i + 1]] for i in idx]
                                 ).astype(np.float32)
        # ---- kNN (baseline) ----
        bk = l2n(torch.from_numpy(bank_np).to(DEV))
        knn = []
        for i in range(len(te["types"])):
            q = torch.from_numpy(te["feats"][i].astype(np.float32)).to(DEV)
            with torch.no_grad():
                knn.append(nn_dist(l2n(q), bk).cpu().numpy())
        # ---- PCA residual (raw features, SubspaceAD convention) ----
        pca, k = scores_for(bank_np, [te["feats"][i].astype(np.float32)
                                      for i in range(len(te["types"]))], 0.99)
        per_layer[layer] = {"knn": knn, "pca": pca, "k": k,
                            "bank": bank_np}

    # fused: average the RAW features across layers, then one PCA (their recipe).
    # Caveat: averaging raw features weights each layer by its feature norm, and
    # our three layers span a wider depth range than SubspaceAD's near-contiguous
    # blocks 23-29, so this variant is reported mainly to show the aggregation
    # choice is not what drives the result.
    norms = {l: float(np.linalg.norm(per_layer[l]["bank"], axis=1).mean())
             for l in LAYERS}
    fused_bank = np.mean([per_layer[l]["bank"] for l in LAYERS], axis=0)
    FQ = []
    for i in range(len(TESTS[LAYERS[0]]["types"])):
        FQ.append(np.mean([TESTS[l]["feats"][i].astype(np.float32)
                           for l in LAYERS], axis=0))
    mu_f, P_f, k_f = pca_fit(fused_bank, 0.99)
    fused_maps = [pca_residual(q, mu_f, P_f) for q in FQ]

    Z = {l: zscores(per_layer[l]["knn"]) for l in LAYERS}
    ZP = {l: zscores(per_layer[l]["pca"]) for l in LAYERS}
    M = {
        "knn_agg_all3": [(a + b + c) / 3 for a, b, c in
                         zip(Z["mid"], Z["midlate"], Z["final"])],
        "pca_perlayer": [(a + b + c) / 3 for a, b, c in
                         zip(ZP["mid"], ZP["midlate"], ZP["final"])],
        "pca_fused": zscores(fused_maps),
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
            g2 = maps[i].reshape(gh, gw)
            assert g2.shape == (gh, gw), f"map shape {g2.shape}"
            np.save(p, g2)
            g = gt_path(obj, te["names"][i]) if y[i] else None
            jobs.append((p, g, tuple(int(v) for v in te["img_hw"][i])))
        m = pixel_metrics_binned(jobs, pro_limit=0.05)
        rows.append({"object": obj, "shot": shot, "split": split, "config": name,
                     "img_AUROC": roc_auc_score(y, sc) * 100,
                     "px_AUROC": m["px_AUROC"] * 100, "AUPRO": m["AUPRO"] * 100,
                     "pca_k_mid": per_layer["mid"]["k"],
                     "pca_k_final": per_layer["final"]["k"],
                     "pca_k_fused": k_f,
                     "norm_mid": norms["mid"], "norm_midlate": norms["midlate"],
                     "norm_final": norms["final"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--objects", default="")
    ap.add_argument("--tag", default="gate_r2_knn_vs_pca")
    a = ap.parse_args()
    objs = a.objects.split(",") if a.objects else sorted(
        d for d in os.listdir(V1)
        if os.path.isdir(os.path.join(V1, d, "train")))
    shots = [int(s) for s in a.shots.split(",")]
    os.makedirs(MAPS, exist_ok=True)

    rows = []
    t0 = time.time()
    for obj_idx, obj in enumerate(objs):
        TESTS = {layer: load_test(layer, obj) for layer in LAYERS}
        for shot in shots:
            for sp in range(a.splits):
                rows.extend(run(obj, obj_idx, shot, sp, TESTS))
        print(f"  {obj:<12} done  ({time.time() - t0:5.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(METRICS, f"{a.tag}.csv"),
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"{a.tag}.csv"), index=False)
    print("\n" + "=" * 100)
    print(f"kNN vs PCA SUBSPACE (paired) -- {len(objs)} objects x "
          f"{len(shots)} shots x {a.splits} splits")
    print("=" * 100)
    piv = df.pivot_table(index=["object", "shot", "split"], columns="config",
                         values=["img_AUROC", "px_AUROC", "AUPRO"])
    for met in ["img_AUROC", "px_AUROC", "AUPRO"]:
        print(f"\n  {met}")
        print(f"    knn_agg_all3   {piv[(met, 'knn_agg_all3')].mean():6.2f}")
        for c in ["pca_perlayer", "pca_fused"]:
            d = piv[(met, c)] - piv[(met, "knn_agg_all3")]
            print(f"    {c:<14} {piv[(met, c)].mean():6.2f}   "
                  f"delta vs kNN {d.mean():+6.2f}   "
                  f"better {int((d > 0).sum())}/{len(d)} cells")
        dpl = piv[(met, "pca_perlayer")] - piv[(met, "knn_agg_all3")]
        byshot = "  ".join(f"{s}s {dpl.xs(s, level='shot').mean():+.2f}"
                           for s in shots)
        print(f"    pca_perlayer by shot: {byshot}")

    print("\n  PCA dimension chosen (median over cells):")
    print(f"    k(mid)   {df['pca_k_mid'].median():.0f}   "
          f"k(final) {df['pca_k_final'].median():.0f}   "
          f"k(fused) {df['pca_k_fused'].median():.0f}")
    print(f"  mean feature norm: mid {df['norm_mid'].median():.1f}  "
          f"midlate {df['norm_midlate'].median():.1f}  "
          f"final {df['norm_final'].median():.1f}"
          f"   <- pca_fused averages these raw, so the largest dominates")

    print("\n  per-object AUPRO delta (pca_perlayer - knn):")
    d = (piv[("AUPRO", "pca_perlayer")] - piv[("AUPRO", "knn_agg_all3")]
         ).groupby(level=0).mean()
    print("    " + " ".join(f"{o}={v:+.1f}" for o, v in d.items()))


if __name__ == "__main__":
    main()
