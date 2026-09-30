# -*- coding: utf-8 -*-
"""R6-A1C -- Nuisance Tail Source Audit.

One question: what ARE the far-field, GT-clean, high-tail patches?

Two passes, for memory: pass 1 uses the final layer only to find the top-1% tail
and define the patch groups (a few hundred patches per image); pass 2 fetches
the other layers for those patches only.  Test queries are never held resident.

Patch groups (per object, per bad test image)
    D        GT defect patches (gt_frac > 0.10)
    N_H      GT-clean, far from any defect (Chebyshev > 4), inside the top-1% tail
    N_R_far  GT-clean, far from any defect, NOT in the tail -- matched spatial
             population, so any difference is a SCORE difference, not a
             SPATIAL one
    N_R_all  GT-clean anywhere, not in the tail (secondary contrast)

Four audits
  1 support coverage   d_k for k = 1,2,5,10.  d1 high but d5/d10 normal
                       -> single-neighbour instability; d1..d10 all high
                       -> genuinely outside support coverage
  2 neighbour diversity among the 10 nearest support patches: how many distinct
                       support images, and how spread out they are
  3 layer consistency  the same patch's d1 at mid / midlate / final
  4 normal-only detectability  pooled support-LOO distance distribution gives a
                       99th-percentile threshold per layer; what fraction of each
                       group exceeds it, using ONLY normal data

Defect labels define the diagnostic groups and nothing else -- no score,
threshold or calibration uses them.  Diagnostic only; no method.
"""
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "model_v0"))
sys.path.insert(0, str(ROOT / "experiments" / "baseline"))
sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDINO"))

from phase_m1_rescue import load_obj                          # noqa: E402
from gate_r2_layer_confirm import draw_images, l2n            # noqa: E402

DEV = "cuda"
GT_THR = 0.10
FAR = 4
ALPHA = 0.01
KS = [1, 2, 5, 10]
N_NEIGH = 10
LAYERS = ["mid", "midlate", "final"]
CHUNK = 512
N_RANDOM = 5
SEED = 20260930
SELECTION = ROOT / "r6a_supervised_smoke" / "R6A_SMOKE_SELECTION.csv"
OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1c_nuisance_source"


def bank_of(c, ids):
    o = c["offsets"]
    return l2n(torch.from_numpy(np.concatenate(
        [c["feats"][int(o[i]):int(o[i + 1])] for i in ids]).astype(np.float32)).to(DEV))


def knn(q, bank, ks, want_idx=0):
    """k-th nearest distances (k in ks) and, if want_idx, the indices of the
    want_idx nearest bank rows.  Chunked over queries."""
    kmax = max(max(ks), want_idx)
    dk = {k: np.empty(len(q), np.float64) for k in ks}
    ix = np.empty((len(q), want_idx), np.int64) if want_idx else None
    for s in range(0, len(q), CHUNK):
        b = q[s:s + CHUNK]
        with torch.inference_mode():
            d = (1.0 - b @ bank.T).cpu().numpy().astype(np.float64)
        part = np.partition(d, kmax - 1, axis=1)[:, :kmax]
        for k in ks:
            dk[k][s:s + len(b)] = part[:, k - 1]
        if want_idx:
            ix[s:s + len(b)] = np.argpartition(d, want_idx - 1, axis=1)[:, :want_idx]
        del d
    return dk, ix


def cheb(a, b, C):
    if not len(a) or not len(b):
        return np.full(len(a), np.inf)
    ar, ac = np.divmod(a, C)
    br, bc = np.divmod(b, C)
    return np.maximum(np.abs(ar[:, None] - br[None, :]),
                      np.abs(ac[:, None] - bc[None, :])).min(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(a.selection)

    cov, div, lay, det = [], [], [], []
    for _, row in sel.iterrows():
        obj, shot, split = str(row.object), int(row.shot), int(row["split"])
        t0 = time.perf_counter()
        _, _, tr, te = load_obj("r0", obj)
        tec = te["final"]
        ids = [int(v) for v in draw_images(tr["final"]["offsets"], shot, split, obj)]
        o = np.asarray(tec["offsets"])
        types = np.asarray([str(v) for v in tec["types"]])
        R, C = (int(v) for v in tec["grids"][0])
        if any(tuple(int(v) for v in g) != (R, C) for g in tec["grids"]):
            raise SystemExit(f"{obj}: token grid varies")
        tnames = np.asarray([str(v) for v in tec["names"]])
        gt = [np.asarray(tec["gt_frac"][int(o[i]):int(o[i + 1])], float)
              for i in range(len(types))]

        # ---- pass 1: final layer -> groups ---------------------------------
        bank = bank_of(tr["final"], ids)
        n_tok = int(np.asarray(tr["final"]["offsets"])[1]
                    - np.asarray(tr["final"]["offsets"])[0])
        groups = {}          # image index -> dict(group -> index array)
        d1_final = {}
        ndiv = {}
        for i in range(len(types)):
            if types[i] != "bad":
                continue
            g = gt[i]
            G = np.where(g > GT_THR)[0]
            clean = np.where(g == 0)[0]
            if not len(G) or not len(clean):
                continue
            far = clean[cheb(clean, G, C) > FAR]
            if not len(far):
                continue
            q = l2n(torch.from_numpy(
                tec["feats"][int(o[i]):int(o[i + 1])].astype(np.float32)).to(DEV))
            dk, ix = knn(q, bank, KS, want_idx=N_NEIGH)
            k = max(1, int(round(ALPHA * len(q))))
            tail = np.argpartition(dk[1], len(dk[1]) - k)[-k:]
            NH = np.intersect1d(tail, far)
            if not len(NH):
                del q; continue
            rng = np.random.default_rng(SEED + int(zlib.crc32(obj.encode())) % 100000 + i)
            grp = {"D": G, "N_H": NH,
                   "N_R_all": rng.choice(clean, min(len(NH), len(clean)), replace=False)}
            pool = np.setdiff1d(far, tail)
            if len(pool) >= len(NH):
                grp["N_R_far"] = rng.choice(pool, len(NH), replace=False)
            groups[i] = grp
            d1_final[i] = dk[1]
            for gname, idx in grp.items():
                if not len(idx):
                    continue
                cov.append(dict(object=obj, image=tnames[i], group=gname, n=len(idx),
                                **{f"d{k}": float(np.mean(dk[k][idx])) for k in KS}))
                slots = ix[idx] // n_tok
                pos = ix[idx] % n_tok
                pr, pc = pos // C, pos % C
                div.append(dict(object=obj, image=tnames[i], group=gname, n=len(idx),
                                distinct_support_images=float(np.mean(
                                    [len(np.unique(s)) for s in slots])),
                                neighbour_spatial_std=float(np.mean(
                                    np.std(pr, axis=1) + np.std(pc, axis=1))),
                                frac_from_one_image=float(np.mean(
                                    [np.bincount(s, minlength=len(ids)).max() / N_NEIGH
                                     for s in slots]))))
            del q
        del bank
        if not groups:
            print(f"  {obj}: no usable bad images, skip", flush=True)
            continue

        # ---- support-LOO null per layer (normal data only) ------------------
        loo = {}
        for l in LAYERS:
            b = bank_of(tr[l], ids)
            oo = np.asarray(tr[l]["offsets"])
            nl = int(oo[1] - oo[0])
            vals = []
            for si, sid in enumerate(ids):
                keep = np.ones(len(ids) * nl, bool)
                keep[si * nl:(si + 1) * nl] = False
                q = l2n(torch.from_numpy(
                    tr[l]["feats"][int(oo[sid]):int(oo[sid + 1])].astype(np.float32)).to(DEV))
                vals.append(knn(q, b[keep], [1])[0][1])
                del q
            loo[l] = np.concatenate(vals)
            thr = float(np.percentile(loo[l], 99.0))
            # ---- pass 2: this layer, only the grouped patches ---------------
            for i, grp in groups.items():
                flat = np.concatenate([grp[k_] for k_ in grp])
                owner = np.concatenate([[k_] * len(grp[k_]) for k_ in grp])
                q = l2n(torch.from_numpy(
                    tr[l]["feats"][int(oo[i]):int(oo[i + 1])][flat]
                    .astype(np.float32)).to(DEV))
                d1 = knn(q, b, [1])[0][1]
                del q
                for k_ in grp:
                    m = owner == k_
                    if not m.any():
                        continue
                    lay.append(dict(object=obj, layer=l, group=k_, n=int(m.sum()),
                                    d1_mean=float(d1[m].mean())))
                    det.append(dict(object=obj, layer=l, group=k_, n=int(m.sum()),
                                    p99_threshold=thr,
                                    frac_over_p99=float(np.mean(d1[m] > thr))))
            del b
        print(f"  {obj} done ({time.perf_counter()-t0:.1f}s)", flush=True)
        for rows, fn in ((cov, "r6a1c_support_coverage.csv"),
                         (div, "r6a1c_neighbour_diversity.csv"),
                         (lay, "r6a1c_layer_consistency.csv"),
                         (det, "r6a1c_normal_only_detectability.csv")):
            pd.DataFrame(rows).to_csv(out / fn, index=False)
    print("DONE", out)


if __name__ == "__main__":
    main()
