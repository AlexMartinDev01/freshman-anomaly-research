# -*- coding: utf-8 -*-
"""R6-A1 -- Tail Geometry & Calibration Audit (mechanism decomposition).

Frozen plan: R6A_Smoke_Audit_NextPlan/R6A_NEXT_PLAN_FROZEN.md
Diagnostic decomposition only.  No new method.  Test labels are used ONLY for
oracle arms (A2) and for GT-purity accounting; every calibration in A3/A4 is
fitted on support-normal data only.

A1  tail occupancy / purity at a fixed alpha grid
A2  oracle extent-matched pairwise diagnostic (m = that image's defect count)
A3  cross-image calibration: raw vs robust relative z
A4  normal-only monotonic calibration (support-normal LOO CDF), preserves patch
    ranking by construction -- so any image-level change isolates aggregation
A5  decomposition of the fixed top-1% set

Usage
    python r6a1_tail_audit/r6a1_tail_audit.py            # 1NN arms (fast)
    python r6a1_tail_audit/r6a1_tail_audit.py --probe    # + frozen probe refit
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "model_v0"))
sys.path.insert(0, str(ROOT / "experiments" / "baseline"))
sys.path.insert(0, str(ROOT / "third_party" / "AnomalyDINO"))

from phase_m1_rescue import load_obj                      # noqa: E402
from gate_r2_layer_confirm import draw_images, l2n, nn_dist  # noqa: E402

DEV = "cuda"
GT_THR = 0.10
ALPHAS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
SELECTION = ROOT / "r6a_supervised_smoke" / "R6A_SMOKE_SELECTION.csv"
OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1_tail_audit"


def make_bank(c, ids):
    o = c["offsets"]
    x = np.concatenate([c["feats"][int(o[i]):int(o[i + 1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))


def score_maps(trc, tec, shot, split, obj):
    """1NN maps for every test image + the support-LOO reference distances."""
    ids = [int(v) for v in draw_images(trc["offsets"], shot, split, obj)]
    offs = np.asarray(tec["offsets"])
    bank = make_bank(trc, ids)
    maps = []
    for i in range(len(tec["types"])):
        x = tec["feats"][int(offs[i]):int(offs[i + 1])].astype(np.float32)
        with torch.inference_mode():
            maps.append(nn_dist(l2n(torch.from_numpy(x).to(DEV)), bank).cpu().numpy().astype(np.float64))
    del bank
    # support-normal LOO distances -> the normal-only reference distribution F_N
    ref = []
    for sid in ids:
        loo = [j for j in ids if j != sid]
        b = make_bank(trc, loo)
        x = trc["feats"][int(trc["offsets"][sid]):int(trc["offsets"][sid + 1])].astype(np.float32)
        with torch.inference_mode():
            ref.append(nn_dist(l2n(torch.from_numpy(x).to(DEV)), b).cpu().numpy().astype(np.float64))
        del b
    return ids, maps, np.concatenate(ref)


def labels_of(tec, i):
    o = np.asarray(tec["offsets"])
    gt = np.asarray(tec["gt_frac"][int(o[i]):int(o[i + 1])], float)
    y = np.full(len(gt), -1, np.int8)
    y[gt == 0] = 0
    y[gt > GT_THR] = 1
    return y


def kmax(v, alpha):
    return max(1, int(round(alpha * len(v))))


def topk_mean(v, k):
    return float(np.partition(v, len(v) - k)[-k:].mean())


def auc(y, s):
    return float(roc_auc_score(y, np.asarray(s)) * 100)


def analyse_object(obj, shot, split, tec, maps, ref, do_probe, probe_scores):
    types = np.asarray([str(v) for v in tec["types"]])
    yimg = (types == "bad").astype(int)
    labels = [labels_of(tec, i) for i in range(len(types))]
    n_def = np.array([int((lab == 1).sum()) for lab in labels])
    good = np.where(yimg == 0)[0]
    bad = np.where(yimg == 1)[0]

    t_rows, d_rows, o_rows, c_rows = [], [], [], []

    # ---------------- A1 / A5 : fixed-alpha tail occupancy and decomposition
    for a in ALPHAS:
        for i in range(len(types)):
            v = maps[i]
            k = kmax(v, a)
            sel = np.argpartition(v, len(v) - k)[-k:]
            lab = labels[i]
            nd = int((lab[sel] == 1).sum())
            nclean = k - nd
            t_rows.append(dict(
                object=obj, arm="1nn", image=types[i], image_id=i, alpha=a, k=k, n_patches=len(v),
                defect_in_image=int(n_def[i]),
                selected_defect=nd, selected_clean=nclean,
                gt_precision=nd / k, gt_recall=(nd / n_def[i]) if n_def[i] else np.nan,
                defect_over_selected=nd / k,
                clean_selected_mean=float(v[sel][lab[sel] == 0].mean()) if nclean else np.nan,
                defect_selected_mean=float(v[sel][lab[sel] == 1].mean()) if nd else np.nan,
                tail_mean=float(v[sel].mean())))
        for i in bad:
            v = maps[i]
            k = kmax(v, a)
            sel = np.argpartition(v, len(v) - k)[-k:]
            lab = labels[i]
            dsel = v[sel][lab[sel] == 1]
            csel = v[sel][lab[sel] == 0]
            d_rows.append(dict(
                object=obj, alpha=a, image_id=i, k=k, n_defect=int(n_def[i]),
                n_defect_selected=len(dsel), n_clean_selected=len(csel),
                defect_selected_mean=float(dsel.mean()) if len(dsel) else np.nan,
                clean_selected_mean=float(csel.mean()) if len(csel) else np.nan,
                tail_mean=float(v[sel].mean())))

    # ---------------- A2 : oracle extent-matched pairwise diagnostic
    for i in bad:
        m = int(n_def[i])
        if m < 1:
            continue
        tb = topk_mean(maps[i], m)
        tg = np.array([topk_mean(maps[j], m) for j in good])
        o_rows.append(dict(object=obj, image_id=i, m=m, t_bad=tb,
                           n_good=len(good), n_good_below=int((tg < tb).sum()),
                           n_good_tied=int((tg == tb).sum())))
    # same construction at the frozen 1% for comparison
    for i in bad:
        v = maps[i]
        k = kmax(v, 0.01)
        tb = topk_mean(v, k)
        tg = np.array([topk_mean(maps[j], k) for j in good])
        o_rows.append(dict(object=obj, image_id=i, m=-1, t_bad=tb,
                           n_good=len(good), n_good_below=int((tg < tb).sum()),
                           n_good_tied=int((tg == tb).sum())))

    # ---------------- A3 : cross-image calibration arms
    z_maps = []
    for v in maps:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        s = 1.4826 * mad
        if not np.isfinite(s) or s <= 1e-10:
            s = float(np.std(v))
        z_maps.append((v - med) / max(s, 1e-8))
    # ---------------- A4 : normal-only monotonic calibration (F_N from support LOO)
    rs = np.sort(ref)
    def cdf(x):
        return np.searchsorted(rs, x, side="right") / len(rs)
    u_maps = [cdf(v) for v in maps]
    from scipy.special import erfinv
    zz_maps = [np.sqrt(2) * erfinv(np.clip(2 * u - 1, -1 + 1e-9, 1 - 1e-9))
               for u in u_maps]

    for name, mm in (("raw_1nn", maps), ("robust_z", z_maps),
                     ("normal_cdf", u_maps), ("normal_gauss", zz_maps)):
        c_rows.append(dict(object=obj, arm=name,
                           patch_auc=auc(np.concatenate([labels[i][labels[i] >= 0]
                                                         for i in range(len(types))]),
                                         np.concatenate([mm[i][labels[i] >= 0]
                                                         for i in range(len(types))])),
                           image_auc=auc(yimg, [topk_mean(mm[i], kmax(mm[i], 0.01))
                                                for i in range(len(types))]),
                           image_auc_top05=auc(yimg, [topk_mean(mm[i], kmax(mm[i], 0.005))
                                                      for i in range(len(types))])))

    # ---------------- A1 for the probe arm (needs the refit)
    if do_probe and probe_scores is not None:
        for a in ALPHAS:
            for i in range(len(types)):
                v = probe_scores[i]
                k = kmax(v, a)
                sel = np.argpartition(v, len(v) - k)[-k:]
                lab = labels[i]
                nd = int((lab[sel] == 1).sum())
                t_rows.append(dict(
                    object=obj, arm="probe", image=types[i], image_id=i, alpha=a, k=k,
                    n_patches=len(v), defect_in_image=int(n_def[i]),
                    selected_defect=nd, selected_clean=k - nd,
                    gt_precision=nd / k,
                    gt_recall=(nd / n_def[i]) if n_def[i] else np.nan,
                    defect_over_selected=nd / k,
                    clean_selected_mean=float(v[sel][lab[sel] == 0].mean()) if (k - nd) else np.nan,
                    defect_selected_mean=float(v[sel][lab[sel] == 1].mean()) if nd else np.nan,
                    tail_mean=float(v[sel].mean())))
        c_rows.append(dict(object=obj, arm="probe",
                           patch_auc=auc(np.concatenate([labels[i][labels[i] >= 0]
                                                         for i in range(len(types))]),
                                         np.concatenate([probe_scores[i][labels[i] >= 0]
                                                         for i in range(len(types))])),
                           image_auc=auc(yimg, [topk_mean(probe_scores[i],
                                                          kmax(probe_scores[i], 0.01))
                                                for i in range(len(types))]),
                           image_auc_top05=np.nan))
    return t_rows, d_rows, o_rows, c_rows


def refit_probe(obj, tec, images_X, images_y, folds, nfold, probe):
    """Frozen R6-A probe recipe, returning OOF patch scores for every test image."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    oof = [None] * len(images_X)
    for f in range(nfold):
        tr_ids = np.where(folds != f)[0]
        rng = np.random.default_rng(20260930 + f)
        Xs, ys = [], []
        for i in tr_ids:
            X, y = images_X[i], images_y[i]
            for cls in (0, 1):
                ids = np.where(y == cls)[0]
                if not len(ids):
                    continue
                if len(ids) > 512:
                    ids = rng.choice(ids, 512, replace=False)
                Xs.append(X[ids]); ys.append(np.full(len(ids), cls, np.int8))
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
        sc = StandardScaler().fit(Xtr)
        Xz = sc.transform(Xtr)
        if probe == "logreg":
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                   solver="liblinear", random_state=20260930)
        else:
            m = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000,
                          random_state=20260930)
        m.fit(Xz, ytr)
        for i in np.where(folds == f)[0]:
            Xt = sc.transform(images_X[i])
            oof[i] = (m.predict_proba(Xt)[:, 1] if probe == "logreg"
                      else m.decision_function(Xt))
        del Xs, ys, Xtr, ytr, Xz
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(a.selection)

    T, Dg, O, C, PL = [], [], [], [], []
    for _, r in sel.iterrows():
        obj, shot, split = str(r.object), int(r.shot), int(r["split"])
        t0 = time.perf_counter()
        _, _, tr, te = load_obj("r0", obj)
        trc, tec = tr["final"], te["final"]
        ids, maps, ref = score_maps(trc, tec, shot, split, obj)

        probe_scores = None
        if a.probe:
            offs = np.asarray(tec["offsets"])
            images_X = [np.asarray(tec["feats"][int(offs[i]):int(offs[i + 1])], np.float32)
                        for i in range(len(tec["types"]))]
            images_y = [labels_of(tec, i) for i in range(len(tec["types"]))]
            folds = pd.read_csv(ROOT / "results" / "model_v0" / "metrics" / "r6a_smoke"
                                / "folds.csv")
            fo = folds[folds.object == obj].sort_values("image_id").fold.values
            nfold = int(fo.max()) + 1
            for probe in ("logreg", "linsvm"):
                oof = refit_probe(obj, tec, images_X, images_y, fo, nfold, probe)
                for i, s in enumerate(oof):
                    for p in range(len(s)):
                        PL.append(dict(object=obj, probe=probe, image_id=i, patch=p,
                                       score=float(s[p]), y=int(images_y[i][p])))
            # A1 probe arm uses the logreg scores (both probes agree closely on patches)
            probe_scores = [None] * len(maps)
            lg = [x for x in PL if x["object"] == obj and x["probe"] == "logreg"]
            tmp = {}
            for x in lg:
                tmp.setdefault(x["image_id"], {})[x["patch"]] = x["score"]
            probe_scores = [np.array([tmp[i][p] for p in range(len(maps[i]))]) for i in range(len(maps))]
            del images_X

        t, d, o, c = analyse_object(obj, shot, split, tec, maps, ref, a.probe, probe_scores)
        T += t; Dg += d; O += o; C += c
        for name, rows, fn in (("a1_tail_occupancy", T, "r6a1_tail_occupancy.csv"),
                               ("a5_tail_decomposition", Dg, "r6a1_tail_decomposition.csv"),
                               ("a2_oracle_extent", O, "r6a1_oracle_extent.csv"),
                               ("a3a4_calibration", C, "r6a1_calibration_arms.csv")):
            pd.DataFrame(rows).to_csv(out / fn, index=False)
        print(f"  {obj} done ({time.perf_counter()-t0:.1f}s)", flush=True)
    if PL:
        pd.DataFrame(PL).to_csv(out / "r6a1_probe_oof_scores.csv", index=False)
    print("DONE", out)


if __name__ == "__main__":
    main()
