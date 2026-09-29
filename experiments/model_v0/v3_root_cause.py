# -*- coding: utf-8 -*-
"""
V3 root-cause diagnosis for the catastrophic refinement degradation.

Read-only with respect to the frozen implementation: it opens the stored
a0.npy / a2.npy / arms.npz / diagnostics.json and writes ONLY into
v3_root_cause_diagnosis/.  It never touches results/model_v0/v3/, never
re-runs the pipeline, and never changes HEAD.

Stage 1  independent metric recompute vs the aggregation CSV (must agree)
Stage 2  writeback sanity: do A3's changed patches lie inside the covered set?
Stage 3  score-chain trace on refined regions: A0 vs A2 vs A3 vs A5 on
         defect vs normal pixels -- this is where the mechanism shows up.

Usage: python experiments/model_v0/v3_root_cause.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\baseline")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from pixel_metrics_binned import dists2map  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from phase_m1_rescue import V1, VISA, gt_file  # noqa: E402

V3 = os.path.join(RESULTS, "v3")
OUT = os.path.join(RESULTS, "v3_root_cause_diagnosis")
CELLS = ["mvtec/capsule/k1/s0", "mvtec/cable/k8/s0", "mvtec/bottle/k1/s0"]
ARMS = ["a0", "a2", "a3", "a5"] + [f"a4s{j}" for j in range(10)]


def cell_dir(cid):
    return os.path.join(V3, *cid.split("/"))


def load_ctx(cid):
    d = cell_dir(cid)
    ds, obj = cid.split("/")[0], cid.split("/")[1]
    root = V1 if ds == "mvtec" else VISA
    te = np.load(os.path.join(RESULTS,
                              "cache_visa" if ds == "visa" else "cache_ml",
                              "final", f"{obj}_test.npz"), allow_pickle=True)
    names = [str(x) for x in te["names"]]
    types = [str(x) for x in te["types"]]
    meta = json.load(open(os.path.join(d, "metadata.json")))
    diag = json.load(open(os.path.join(d, "diagnostics.json")))
    a0 = np.load(os.path.join(d, "a0.npy"))
    a2 = np.load(os.path.join(d, "a2.npy"))
    z = np.load(os.path.join(d, "arms.npz"))
    gts = []
    for i, nm in enumerate(names):
        gp = None if types[i] == "good" else gt_file(root, obj, nm)
        if gp is not None:
            im = cv2.imread(os.path.join(root, obj, "test", *nm.split("/")),
                            cv2.IMREAD_COLOR)
            hw = im.shape[:2]
            g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            gp = None if (g is None or (g > 0).sum() == 0) else \
                (cv2.resize(g, (hw[1], hw[0]),
                            interpolation=cv2.INTER_NEAREST) > 0)
        gts.append(gp)
    return dict(d=d, diag=diag, meta=meta, names=names, types=types, a0=a0,
                a2=a2, z=z, gts=gts, root=root, obj=obj)


def stage1(cid, C):
    """Independent recompute of the three metrics, compared with the CSV."""
    from sklearn.metrics import roc_auc_score
    from pixel_metrics_binned import pixel_metrics_binned
    import tempfile, shutil
    y = np.array([0 if t == "good" else 1 for t in C["types"]])
    rec = []
    tmp = tempfile.mkdtemp(prefix="rc_")
    for arm in ARMS:
        if arm == "a0":
            M = C["a0"]
        elif arm == "a2":
            M = C["a2"]
        elif arm in C["z"].files:
            M = C["z"][arm]
        else:
            continue
        sub = os.path.join(tmp, arm)
        os.makedirs(sub, exist_ok=True)
        jobs = []
        for i in range(M.shape[0]):
            p = os.path.join(sub, f"{i}.npy")
            np.save(p, M[i])
            hw = None
            for nm in [C["names"][i]]:
                im = cv2.imread(os.path.join(C["root"], C["obj"], "test",
                                             *nm.split("/")), cv2.IMREAD_COLOR)
                hw = im.shape[:2]
            gp = None
            if y[i]:
                gp = gt_file(C["root"], C["obj"], C["names"][i])
                if gp is not None:
                    g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
                    if g is None or (g > 0).sum() == 0:
                        gp = None
            jobs.append((p, gp, hw))
        mm = pixel_metrics_binned(jobs, pro_limit=0.05)
        sc = np.array([float(m.mean()) for m in M])
        rec.append(dict(cell_id=cid, arm=arm,
                        img_AUROC=roc_auc_score(y, sc) * 100,
                        px_AUROC=mm["px_AUROC"] * 100, AUPRO=mm["AUPRO"] * 100))
        shutil.rmtree(sub, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return rec


def stage2(cid, C):
    """Do A3's deviations from A0 sit inside the covered patches?"""
    a0, a3 = C["a0"], C["z"]["a3"]
    diff = np.abs(a3 - a0).reshape(a3.shape[0], -1)
    changed = diff > 1e-9
    rows = []
    for i in range(a3.shape[0]):
        m2 = C["diag"]["a3"][i]["m2"]
        n_ch = int(changed[i].sum())
        rows.append(dict(cell_id=cid, image=C["names"][i], m2=m2,
                         n_changed=n_ch,
                         within_covered=(n_ch > 0) == (m2 > 0)))
    return rows


def stage3(cid, C):
    """On refined patches: how do A0 / A2 / A3 / A5 score defect vs normal?"""
    from scipy.stats import spearmanr
    out = []
    for i in range(C["a0"].shape[0]):
        if C["diag"]["a3"][i]["m2"] == 0:
            continue
        gp = C["gts"][i]
        if gp is None:
            continue
        # a0/a3/a5 are on the 448 grid, a2 is on the 672 grid -- resizing the
        # GT to a0's grid and reusing it for a2 crashed the first run.
        gh, gw = C["a0"][i].shape
        gt_g = cv2.resize(gp.astype(np.uint8), (gw, gh),
                          interpolation=cv2.INTER_NEAREST) > 0
        if gt_g.sum() == 0 or (~gt_g).sum() == 0:
            continue
        row = dict(cell_id=cid, image=C["names"][i], n_def=int(gt_g.sum()),
                   n_norm=int((~gt_g).sum()))
        for nm, M in (("a0", C["a0"]), ("a2", C["a2"]), ("a3", C["z"]["a3"]),
                      ("a5", C["z"]["a5"])):
            n = M[i].size
            side = int(round(np.sqrt(n)))
            # use each arm's OWN grid; only the 448-grid arms share the GT mask
            if side * side == n and side != gh:
                gg = np.ones((side, side), dtype=bool)     # a2: no per-pixel GT
                d_, n_ = M[i].reshape(side, side).ravel(), None
                row[f"{nm}_def_med"] = np.nan
                row[f"{nm}_norm_med"] = float(np.median(d_))
                row[f"{nm}_gap"] = np.nan
                continue
            g = M[i].reshape(gh, gw)
            d_, n_ = g[gt_g], g[~gt_g]
            row[f"{nm}_def_med"] = float(np.median(d_))
            row[f"{nm}_norm_med"] = float(np.median(n_))
            row[f"{nm}_gap"] = float(np.median(d_) - np.median(n_))
        out.append(row)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    csv = pd.read_csv(os.path.join(RESULTS, "v3_agg", "merged_partial.csv"))
    print(f"output dir {OUT}\n")
    all1, all2, all3 = [], [], []
    for cid in CELLS:
        if not os.path.exists(os.path.join(cell_dir(cid), "DONE.json")):
            print(f"  {cid}: no DONE, skip"); continue
        C = load_ctx(cid)
        r1 = stage1(cid, C)
        bad = []
        for r in r1:
            m = csv[(csv.cell_id == cid) & (csv.arm == r["arm"])]
            if m.empty:
                continue
            for k in ("img_AUROC", "px_AUROC", "AUPRO"):
                dv = abs(float(m.iloc[0][k]) - r[k])
                if dv > 1e-6:
                    bad.append((r["arm"], k, dv))
        a3v = [r for r in r1 if r["arm"] == "a3"][0]
        a0v = [r for r in r1 if r["arm"] == "a0"][0]
        a2v = [r for r in r1 if r["arm"] == "a2"][0]
        a5v = [r for r in r1 if r["arm"] == "a5"][0]
        print(f"  {cid}")
        print(f"    AUPRO  a0 {a0v['AUPRO']:7.2f}  a2 {a2v['AUPRO']:7.2f}  "
              f"a3 {a3v['AUPRO']:7.2f}  a5 {a5v['AUPRO']:7.2f}   "
              f"(a3-a0 {a3v['AUPRO'] - a0v['AUPRO']:+.2f})")
        print(f"    recheck vs CSV: {'MATCH' if not bad else f'{len(bad)} DIFF {bad[:3]}'}")
        all1.extend(r1)
        all2.extend(stage2(cid, C))
        all3.extend(stage3(cid, C))
    pd.DataFrame(all1).to_csv(os.path.join(OUT, "01_metric_recheck.csv"), index=False)
    pd.DataFrame(all2).to_csv(os.path.join(OUT, "02_writeback_audit.csv"), index=False)
    pd.DataFrame(all3).to_csv(os.path.join(OUT, "03_score_chain.csv"), index=False)
    print(f"\n  wrote 01_metric_recheck.csv ({len(all1)}) "
          f"02_writeback_audit.csv ({len(all2)}) "
          f"03_score_chain.csv ({len(all3)})")


if __name__ == "__main__":
    main()
