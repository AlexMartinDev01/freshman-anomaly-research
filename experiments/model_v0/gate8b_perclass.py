# -*- coding: utf-8 -*-
"""
Gate 8B -- does the retrieval criterion actually work?

Gate 8 found that the pooled/random external banks do not help, but that ONE
(target, external class) cell did: wallplugs against walnuts defects, +16 to +19
over baseline, consistent across all three splits, with the bank alone scoring
~67 AUROC on wallplugs anomalies. One cell out of four objects is a hypothesis,
not a method -- and the user's fork ("retrieval works -> build a Retriever;
retrieval does not -> build a Conditional Generator") cannot be decided from it.

The criterion R1 uses is: rank external classes by how well their NORMAL
manifold covers the target's NORMAL manifold, then take defects from the best.
That is a testable claim, so test it directly instead of through top-1 only:

  For every (target, external class) pair, build a bank of exactly N defect
  patches from that single class and score the target. That yields a full
  matrix, and the criterion is validated only if match distance PREDICTS the
  AUROC column -- not merely if the argmin happens to be good once.

Three outcomes and what each licenses:
  match distance predicts AUROC   -> retrieval is real; build the Retriever
  one class is an unexplained outlier -> the wallplugs/walnuts cell was luck
  several classes are equally good -> the signal is not class identity

Usage: python experiments/model_v0/gate8b_perclass.py [--splits N] [--draws N]
       python experiments/model_v0/gate8b_perclass.py --source normal
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate7b3r import make_dist  # noqa: E402
from tail_calib import RESULTS, load_cache, split_train, mean_top1p  # noqa: E402

OBJECTS = ["wallplugs", "vial", "sheet_metal", "can"]
# No "-1 / whole pool" size: for sheet_metal a 20000-patch bank against 4096
# patch queries costs 168 ms per image, i.e. ~42 min for that one cell, and it
# is the control's least informative point. The matched sizes are the comparison
# that matters.
SIZES = [200, 500, 1000]
METRICS = os.path.join(RESULTS, "metrics")
EXTERNAL = os.path.join(RESULTS, "external")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--source", choices=["defect", "normal"], default="defect",
                    help="'normal' is the control: same sweep, but the bank is "
                         "built from that class's NORMAL patches. If a normal "
                         "bank reproduces a defect bank's gain, the gain was "
                         "class appearance, not defect knowledge.")
    args = ap.parse_args()
    suffix = "" if args.source == "defect" else "_normalbank"
    os.makedirs(METRICS, exist_ok=True)

    match = pd.read_csv(os.path.join(METRICS, "gate8_class_match.csv"))
    rows = []
    for obj in OBJECTS:
        tr, te = load_cache(obj)
        n_train = tr["feats"].shape[0]
        bank_idx, _ = split_train(n_train, 1.0, 0)
        fbank = torch.from_numpy(tr["feats"][bank_idx].reshape(-1, 384)
                                 .astype(np.float32)).to("cuda")
        types = list(te["types"])
        n_img = len(types)
        gt = te["gt_frac"].reshape(n_img, -1)
        bad = np.where(~(np.array(types) == "good"))[0]
        good = [i for i in range(n_img) if types[i] == "good"]
        feats = [torch.from_numpy(te["feats"][i].astype(np.float32)).to("cuda")
                 for i in range(n_img)]
        ext = f"_{args.source}.npy"
        pools = {f[:-len(ext)]: np.load(os.path.join(EXTERNAL, f))
                 for f in sorted(os.listdir(EXTERNAL))
                 if f.endswith(ext) and f[:-len(ext)] != obj}
        pools = {k: v for k, v in pools.items() if len(v)}
        # Cap every bank at the corresponding DEFECT pool size, for both
        # sources, so the normal-bank control is matched bank-for-bank rather
        # than merely sharing a nominal N (can's defect pool is 54 patches).
        cap = {f[:-len("_defect.npy")]:
               len(np.load(os.path.join(EXTERNAL, f)))
               for f in os.listdir(EXTERNAL) if f.endswith("_defect.npy")}

        # Sn depends only on the object, not on the split, so compute it once.
        # For sheet_metal this is a 4096-patch query against a 450k-patch bank
        # (~3.7 s/image) and recomputing it per split dominated the run.
        dist_n = make_dist(fbank)
        with torch.no_grad():
            Sn_all = [dist_n(z) for z in feats]

        for sp in range(args.splits):
            perm = np.random.default_rng(1000 + sp).permutation(len(bad))
            ev_idx = bad[perm[len(bad) // 2:]]
            ev = good + list(ev_idx)
            y = np.array([1 if types[i] == "bad" else 0 for i in ev])
            Sn = [Sn_all[i] for i in ev]

            for cname, pool in sorted(pools.items()):
                for N in SIZES:
                    k = min(N, len(pool), cap.get(cname, len(pool)))
                    for draw in range(args.draws):
                        r = np.random.default_rng(
                            hash((sp, cname, N, draw)) % 2**31)
                        sel = r.choice(len(pool), size=k, replace=False)
                        dist_d = make_dist(torch.from_numpy(
                            pool[sel].astype(np.float32)))
                        with torch.no_grad():
                            Sd = [dist_d(feats[i]) for i in ev]
                        sc = np.array([mean_top1p(Sn[j] - Sd[j])
                                       for j in range(len(ev))])
                        rows.append({"object": obj, "split": sp, "class": cname,
                                     "N": N, "n_used": k, "draw": draw,
                                     "img_AUROC": roc_auc_score(y, sc) * 100})
            print(f"  {obj} split {sp} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(METRICS, f"gate8b_perclass{suffix}.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS, f"gate8b_perclass{suffix}.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"GATE 8B -- single-external-class {args.source.upper()} banks, "
          "matched size")
    print("=" * 100)
    for N in SIZES:
        s = df[df.N == N]
        if s.empty:
            continue
        print(f"\n  N = {N} patches")
        piv = s.pivot_table(index="class", columns="object",
                            values="img_AUROC").round(1)
        print(piv.to_string())

    print("\n" + "=" * 100)
    print("DOES THE RETRIEVAL CRITERION PREDICT THE OUTCOME?")
    print("  Spearman over all (target, class) pairs at each size.")
    print("  Negative rho (closer match -> higher AUROC) is what R1 assumes.")
    print("=" * 100)
    print(f"\n  {'N':>6}{'rho':>10}{'p':>10}{'n_pairs':>10}")
    for N in SIZES:
        s = df[df.N == N].groupby(["object", "class"]).img_AUROC.mean()
        s = s.reset_index().merge(
            match.rename(columns={"target": "object"}), on=["object", "class"])
        if len(s) < 4:
            continue
        rho, p = spearmanr(s.match_dist, s.img_AUROC)
        print(f"  {N:>6}{rho:>10.3f}{p:>10.3f}{len(s):>10}")

    print("\n  per-class mean over all objects (is one class doing everything?):")
    print(df.pivot_table(index="class", columns="N", values="img_AUROC")
          .round(1).to_string())


if __name__ == "__main__":
    main()
