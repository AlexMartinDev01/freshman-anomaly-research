# -*- coding: utf-8 -*-
"""R6-A1C audit 4, bank-size matched.

The first version compared test distances against a bank of 4 support images
with the support-LOO null built on a bank of 3.  The null is therefore harder by
construction, its 99th percentile sits too high, and even real defect patches
mostly fail to exceed it (D: 0.000-0.020).  That is a statistic artefact, not a
detectability statement.

Here BOTH sides use a 3-image bank:
  null  support image i against the other 3            (leave-one-support-out)
  test  test patch against the support minus one image, averaged over the 4
        possible drops, so the bank size and the image mix match the null.

Everything else is unchanged: the threshold is the 99th percentile of the pooled
normal-only null, and no defect label enters any score or threshold.
"""
import argparse
import sys
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
CHUNK = 512
N_RANDOM = 5
SEED = 20260930
SELECTION = ROOT / "r6a_supervised_smoke" / "R6A_SMOKE_SELECTION.csv"
OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1c_nuisance_source"


def nnd(q, bank):
    out = np.empty(len(q), np.float64)
    for s in range(0, len(q), CHUNK):
        with torch.inference_mode():
            out[s:s + CHUNK] = (1.0 - q[s:s + CHUNK] @ bank.T).max(1).values \
                .cpu().numpy().astype(np.float64)
    return out


def cheb(a, b, C):
    if not len(a) or not len(b):
        return np.full(len(a), np.inf)
    ar, ac = np.divmod(a, C); br, bc = np.divmod(b, C)
    return np.maximum(np.abs(ar[:, None] - br[None, :]),
                      abs(ac[:, None] - bc[None, :])).min(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(a.selection)
    rows = []
    for _, row in sel.iterrows():
        obj, shot, split = str(row.object), int(row.shot), int(row["split"])
        _, _, tr, te = load_obj("r0", obj)
        tec = te["final"]
        ids = [int(v) for v in draw_images(tr["final"]["offsets"], shot, split, obj)]
        o = np.asarray(tec["offsets"])
        ot = np.asarray(tr["final"]["offsets"])
        n_tok = int(ot[1] - ot[0])
        types = np.asarray([str(v) for v in tec["types"]])
        C = int(tec["grids"][0][1])
        R = int(tec["grids"][0][0])
        sub_banks = {}
        for drop in range(len(ids)):
            keep = [j for j in ids if j != ids[drop]]
            sub_banks[drop] = l2n(torch.from_numpy(np.concatenate(
                [tr["final"]["feats"][int(ot[j]):int(ot[j + 1])] for j in keep]
            ).astype(np.float32)).to(DEV))

        # normal-only null at the SAME bank size
        null = []
        for si, sid in enumerate(ids):
            q = l2n(torch.from_numpy(
                tr["final"]["feats"][int(ot[sid]):int(ot[sid + 1])].astype(np.float32)).to(DEV))
            null.append(nnd(q, sub_banks[si]))
            del q
        null = np.concatenate(null)
        thr = float(np.percentile(null, 99.0))

        rng = np.random.default_rng(SEED + int(zlib.crc32(obj.encode()) % 100000))
        acc = {g: [] for g in ("D", "N_H", "N_R_far", "N_R_all")}
        for i in range(len(types)):
            if types[i] != "bad":
                continue
            g = np.asarray(tec["gt_frac"][int(o[i]):int(o[i + 1])], float)
            G = np.where(g > GT_THR)[0]
            clean = np.where(g == 0)[0]
            if not len(G) or not len(clean):
                continue
            farc = clean[cheb(clean, G, C) > FAR]
            if not len(farc):
                continue
            # matched 3-image bank score: average over the 4 drops
            qi = l2n(torch.from_numpy(
                tec["feats"][int(o[i]):int(o[i + 1])].astype(np.float32)).to(DEV))
            dbar = np.mean([nnd(qi, sub_banks[d]) for d in range(len(ids))], axis=0)
            del qi
            k = max(1, int(round(ALPHA * len(dbar))))
            tail = np.argpartition(dbar, len(dbar) - k)[-k:]
            NH = np.intersect1d(tail, farc)
            if not len(NH):
                continue
            pick_far = rng.choice(np.setdiff1d(farc, tail),
                                  min(len(NH), len(np.setdiff1d(farc, tail))), replace=False)
            pick_all = rng.choice(clean, min(len(NH), len(clean)), replace=False)
            for gname, idx in (("D", G), ("N_H", NH), ("N_R_far", pick_far),
                               ("N_R_all", pick_all)):
                if len(idx):
                    acc[gname].append(float(np.mean(dbar[idx] > thr)))
        for gname, v in acc.items():
            if v:
                rows.append(dict(object=obj, group=gname, n_images=len(v),
                                 p99_threshold=thr, frac_over_p99=float(np.mean(v))))
        for d in sub_banks.values():
            del d
        print(f"  {obj} done", flush=True)
        pd.DataFrame(rows).to_csv(out / "r6a1c_detectability_matched.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
