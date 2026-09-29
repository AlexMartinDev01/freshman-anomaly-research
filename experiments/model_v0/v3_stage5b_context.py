# -*- coding: utf-8 -*-
"""
Stage 5B -- corrected support-normal context fidelity, all 27 categories.

Fixes four defects in Stage 5 (see ROOT_CAUSE_REPORT):

  1. SELF-MATCH.  Stage 5 built the bank from exactly the k support images,
     then queried a token FROM one of those images against it.  nn_dist finds
     the query itself, so d_full ~ 0 and every relative error divided by ~0
     (the 10^5-10^6 values in 05_context_fidelity_support.csv).  The Spearman
     was computed against a near-constant reference vector and is meaningless.
     Here both queries use the SAME eligible bank with the query's own-image
     3x3 physical neighbourhood excluded -- the rule already frozen in
     addendum 2 for calibration.
  2. NO POSITIVE CONTROL.  Before claiming crop-DINO differs from full-672,
     prove a full re-forward reproduces the cache.  Runs first, fails fast.
  3. Spearman was per-25-token-core.  Now pooled per (dataset, object, n_ctx,
     layer) over all support images x anchors x core tokens.
  4. Per-layer fidelity, not just `all3`, plus trigger profile duplicated 15x.

Also extends budget feasibility to all 324 cells (reads diagnostics only) and
conditions the cap rate on images that ACTUALLY TRIGGER.

Support-only throughout: no test anomaly, no GT, no AUPRO, no A5.

Usage: OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5b_context.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from scipy.stats import spearmanr  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n  # noqa: E402
from phase_m1_rescue import V1, VISA  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_calibration as cal  # noqa: E402
import v3_check_scale as cs  # noqa: E402

OUT = os.path.join(RESULTS, "v3_root_cause_diagnosis")
N_CTX = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
CORE = 5
ANCH = 2                      # ANCH x ANCH anchors per support image
MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def excl_mask(u, v, gh, gw, base, n, bank_size):
    """allowed[token] for a query at (u,v): ban its own image's 3x3 physical
    block (addendum 2 rule); other images and far same-image patches stay."""
    rows, cols = cal.excl_for_672(u, v, gh, gw)
    allow = np.ones(bank_size, dtype=bool)
    if rows.size and cols.size:
        ban = base + (rows[:, None] * gw + cols[None, :]).ravel()
        ban = ban[(ban >= base) & (ban < base + n)]
        allow[ban] = False
    if not allow.any():
        raise AssertionError(
            f"no legal candidate for 672 token ({u},{v}) on a {gh}x{gw} grid: "
            f"the exclusion is never silently relaxed (addendum 2 / 3).")
    return allow


def budget_all324():
    """Budget feasibility over all 324 cells.  Reads diagnostics + grids only,
    so it can be regenerated without redoing the fidelity sweep."""
    brows = []
    for ds, objs in (("mvtec", MVTEC), ("visa", VISA_CATS)):
        for obj in objs:
            for shot in (1, 2, 4, 8):
                for sp in range(3):
                    d = os.path.join(RESULTS, "v3", ds, obj, f"k{shot}", f"s{sp}")
                    fp = os.path.join(d, "diagnostics.json")
                    if not os.path.exists(fp):
                        continue
                    dg = json.load(open(fp))
                    ms = np.array([x["m"] for x in dg.get("a3", [])])
                    trig = ms[ms > 0]
                    # B = 0.5 * gh672 * gw672, read from the 672 cache itself.
                    # It must NOT be derived from the 448 grid: the 448 cache
                    # stores no `grids` at all, and ceil(1.5 * gh448) is not the
                    # 672 grid -- VisA test grids are (48,52)..(48,78), whose
                    # second dims are not 3k, so 1.5x of the true 448 dim is not
                    # even an integer.  Deriving it silently mis-sizes B (and
                    # therefore K_max) for all 12 VisA objects.
                    with np.load(os.path.join(
                            RESULTS, "cache_m1_dino672", "final",
                            f"dino672_{obj}_test.npz"), allow_pickle=True) as z:
                        g6 = z["grids"]
                    uniq = {tuple(int(v) for v in r) for r in g6}
                    if len(uniq) != 1:
                        raise SystemExit(
                            f"{ds}/{obj}: test images span several 672 grids "
                            f"{sorted(uniq)}; B is not a single number.")
                    gh6, gw6 = uniq.pop()
                    B = 0.5 * gh6 * gw6
                    for n in N_CTX:
                        K = int(B // (n * n))
                        brows.append(dict(
                            dataset=ds, object=obj, shot=shot, split=sp,
                            n_ctx=n, B=B, gh672=gh6, gw672=gw6, Kmax=K,
                            n_images=len(ms),
                            n_triggered=int(len(trig)),
                            mean_trigger_m=float(ms.mean()),
                            median_trigger_m=float(np.median(ms)),
                            pct_capped_all=float((ms > K).mean() * 100),
                            pct_capped_triggered=(
                                float((trig > K).mean() * 100) if len(trig) else 0.0),
                            mean_retained_all=float(np.minimum(ms, K).sum() / max(ms.sum(), 1)),
                            mean_retained_triggered=(
                                float(np.minimum(trig, K).sum() / max(trig.sum(), 1))
                                if len(trig) else 0.0),
                            token_utilization=float(
                                np.minimum(ms, K).sum() * n * n / (len(ms) * B))))
    pd.DataFrame(brows).to_csv(os.path.join(OUT, "05b_budget_all324.csv"), index=False)
    print(f"  wrote 05b_budget_all324.csv ({len(brows)} rows)")
    return brows


def write_fidelity(fid):
    """Post-loop aggregation and writes.  Separated from main() so the whole
    tail can be exercised on synthetic rows in a second, instead of being
    validated by a 45-minute sweep that may not reach it."""
    # The vectors go to disk BEFORE any pandas work, so the expensive part can
    # never be lost again.
    np.savez_compressed(
        os.path.join(OUT, "05b_fidelity_vectors.npz"),
        keys=np.array([f"{r['dataset']}|{r['object']}|{r['n_ctx']}|{r['layer']}"
                       for r in fid]),
        dH=np.array([r["_dH"] for r in fid], dtype=np.float32),
        dF=np.array([r["_dF"] for r in fid], dtype=np.float32))

    df = pd.DataFrame(fid)
    # Pooled Spearman per (dataset, object, n_ctx, layer).
    # NOT groupby.apply: numpy walks a pandas Series positionally (seq[0]) while
    # Series.__getitem__ with an integer key is LABEL-based, so any group whose
    # index does not contain the label 0 raises KeyError: 0 -- and it silently
    # "works" for as long as the first group happens to start at row 0.  A plain
    # dict has no such ambiguity.
    acc = {}
    for r in fid:
        k = (r["dataset"], r["object"], r["n_ctx"], r["layer"])
        acc.setdefault(k, ([], []))
        acc[k][0].append(r["_dH"])
        acc[k][1].append(r["_dF"])
    sp = pd.DataFrame([
        dict(dataset=k[0], object=k[1], n_ctx=k[2], layer=k[3],
             nn_spearman=float(spearmanr(np.concatenate(h),
                                         np.concatenate(f)).statistic))
        for k, (h, f) in acc.items()])
    df = df.drop(columns=["_dH", "_dF"]).merge(
        sp, on=["dataset", "object", "n_ctx", "layer"], validate="many_to_one")
    df.to_csv(os.path.join(OUT, "05b_context_fidelity_corrected.csv"), index=False)
    (df.groupby(["dataset", "object", "shot", "n_ctx"])
       .agg(n_positions=("cos_median", "size"), cos_mean=("cos_mean", "mean"),
            cos_median=("cos_median", "median"), cos_p10=("cos_p10", "median"),
            feature_rel_error=("feature_rel_error", "median"),
            nn_spearman=("nn_spearman", "median")).reset_index()
       .to_csv(os.path.join(OUT, "05b_context_fidelity_by_category.csv"), index=False))
    (df.groupby(["dataset", "layer", "n_ctx"])
       .agg(n_positions=("cos_median", "size"), cos_mean=("cos_mean", "mean"),
            cos_median=("cos_median", "median"), cos_p10=("cos_p10", "median"),
            feature_rel_error=("feature_rel_error", "median"),
            nn_spearman=("nn_spearman", "median")).reset_index()
       .to_csv(os.path.join(OUT, "05b_context_fidelity_by_layer.csv"), index=False))
    part = os.path.join(OUT, "_partial_fidelity.csv")
    if os.path.exists(part):
        os.remove(part)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("S5B_BUDGET_ONLY") == "1":
        budget_all324()
        return
    model, resize, totensor, norm = cs.build()

    # ---------- 1. positive control: full re-forward == cache? ----------
    print("=== positive control: full re-forward vs cache ===")
    ident = []
    for ds, obj in (("mvtec", "bottle"), ("mvtec", "capsule"),
                    ("visa", "pcb2"), ("visa", "macaroni1")):
        c672 = rf.load_672(obj, "train")
        o6 = c672[rf.LAYERS[0]]["offsets"]
        ck = c672["final"]["feats"].shape[-1]
        p0 = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train", "good",
                          str(c672[rf.LAYERS[0]]["names"][0]))
        big = rf.resized_grid(cv2.imread(p0, cv2.IMREAD_COLOR), resize)
        gh6, gw6 = big.size[1] // rf.STRIDE, big.size[0] // rf.STRIDE
        t = norm(totensor(big))
        with torch.no_grad():
            tk = model.model.get_intermediate_layers(
                t.unsqueeze(0).to(model.device), n=[5, 8, 11])
        cached = c672["final"]["feats"][o6[0]:o6[0 + 1]]
        rer = tk[-1].squeeze(0).cpu().numpy().astype(np.float16)
        a = rer.astype(np.float32).reshape(-1, ck)
        b = cached.astype(np.float32)
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        cos = float((an * bn).sum(1).mean())
        mad = float(np.abs(a - b).max())
        ok = cos > 0.999 and mad < 0.05
        ident.append(dict(dataset=ds, object=obj, grid=f"{gh6}x{gw6}",
                          n_tokens=int(a.shape[0]), cos_mean=cos,
                          max_abs_diff=mad, pass_=bool(ok)))
        print(f"  {ds}/{obj}: cos={cos:.6f} maxdiff={mad:.5f} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    pd.DataFrame(ident).to_csv(os.path.join(OUT, "05b_full_cache_identity.csv"),
                               index=False)
    if not all(r["pass_"] for r in ident):
        raise SystemExit("POSITIVE CONTROL FAILED -- cache/preprocess mismatch; "
                         "context fidelity numbers would be uninterpretable.")

    # ---------- 2. corrected fidelity, all 27 categories ----------
    fid, prof = [], []
    for objs, ds in ((MVTEC, "mvtec"), (VISA_CATS, "visa")):
        for obj in objs:
            _, c448 = sel.load_split(obj, "train")
            c672 = rf.load_672(obj, "train")
            tro = c448[rf.LAYERS[0]]["offsets"]
            idx = draw_images(tro, 8, 0, obj)
            o6 = c672[rf.LAYERS[0]]["offsets"]
            gs = {tuple(int(v) for v in c672[rf.LAYERS[0]]["grids"][i])
                  for i in idx}
            if len(gs) != 1:
                print(f"  {ds}/{obj}: differing support grids, skip"); continue
            gh6, gw6 = gs.pop()
            n_tok = gh6 * gw6
            bank = {l: l2n(torch.from_numpy(np.concatenate(
                [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
            ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}
            for si, i in enumerate(idx):
                # bank block holding THIS image: the self-exclusion must ban
                # columns of the query's own image, not image 0's (v3_calibration
                # guard_from_feats passes base = o0 for the same reason).
                base_off = si * n_tok
                assert base_off + n_tok <= len(idx) * n_tok
                p = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train",
                                 "good",
                                 str(c672[rf.LAYERS[0]]["names"][i]))
                big = rf.resized_grid(cv2.imread(p, cv2.IMREAD_COLOR), resize)
                full = {l: c672[l]["feats"][o6[i]:o6[i + 1]].reshape(gh6, gw6, -1)
                        for l in rf.LAYERS}
                for ax in range(ANCH):
                    for ay in range(ANCH):
                        ar = int((ax + 1) / (ANCH + 1) * (gh6 - 8))
                        ac = int((ay + 1) / (ANCH + 1) * (gw6 - 8))
                        for n in N_CTX:
                            h = n // 2
                            r0 = min(max(0, ar - h), gh6 - n)
                            c0 = min(max(0, ac - h), gw6 - n)
                            crop = rf.crop_feats(model, big, r0, c0, n,
                                                 norm, totensor)
                            k0 = (n - CORE) // 2
                            for li, lay in enumerate(rf.LAYERS):
                                fh = crop[li][k0:k0 + CORE, k0:k0 + CORE].reshape(
                                    -1, crop[li].shape[-1])
                                ff = full[lay][r0 + k0:r0 + k0 + CORE,
                                               c0 + k0:c0 + k0 + CORE].reshape(-1, crop[li].shape[-1])
                                an = fh / (np.linalg.norm(fh, axis=1, keepdims=True) + 1e-12)
                                bn = ff / (np.linalg.norm(ff, axis=1, keepdims=True) + 1e-12)
                                cos = (an * bn).sum(1)
                                rel = np.linalg.norm(fh - ff, axis=1) / (
                                    np.linalg.norm(ff, axis=1) + 1e-12)
                                # corrected NN: same eligible bank for both
                                qh = l2n(torch.from_numpy(fh.astype(np.float32)).to(DEV))
                                qf = l2n(torch.from_numpy(ff.astype(np.float32)).to(DEV))
                                B = bank[lay]
                                Dh = (1.0 - qh @ B.T).cpu().numpy()
                                Df = (1.0 - qf @ B.T).cpu().numpy()
                                dH, dF = [], []
                                rr, cc = np.mgrid[r0 + k0:r0 + k0 + CORE,
                                                  c0 + k0:c0 + k0 + CORE]
                                for t in range(CORE * CORE):
                                    al = excl_mask(int(rr.ravel()[t]),
                                                   int(cc.ravel()[t]), gh6, gw6,
                                                   base_off, n_tok, Df.shape[1])
                                    dH.append(Dh[t][al].min())
                                    dF.append(Df[t][al].min())
                                dH, dF = np.array(dH), np.array(dF)
                                fid.append(dict(
                                    dataset=ds, object=obj, shot=8, split=0,
                                    support_image_id=int(i), layer=lay, n_ctx=n,
                                    core_size=CORE, halo_tokens=n * n,
                                    n_tokens=CORE * CORE,
                                    cos_mean=float(cos.mean()),
                                    cos_median=float(np.median(cos)),
                                    cos_p10=float(np.percentile(cos, 10)),
                                    feature_rel_error=float(np.median(rel)),
                                    d_full_median=float(np.median(dF)),
                                    d_halo_median=float(np.median(dH)),
                                    nn_rel_error=float(np.median(
                                        np.abs(dH - dF) / (np.abs(dF) + 1e-12))),
                                    _dH=dH.tolist(), _dF=dF.tolist()))
            v3d = os.path.join(RESULTS, "v3", ds, obj, "k8", "s0")
            ms = []
            for r, _, f in os.walk(v3d):
                if "diagnostics.json" in f:
                    dg = json.load(open(os.path.join(r, "diagnostics.json")))
                    ms += [x["m"] for x in dg.get("a3", [])]
            if ms:
                prof.append(dict(dataset=ds, object=obj, shot=8, split=0,
                                 n_images=len(ms), n_triggered=int(sum(1 for x in ms if x > 0)),
                                 m_mean=float(np.mean(ms)), m_max=int(np.max(ms))))
            print(f"  {ds}/{obj} done (grid {gh6}x{gw6}, k={len(idx)})", flush=True)
            # Crash-safe partial, rewritten per object.  The previous run threw
            # away a full 45-minute sweep because every write happened after the
            # loop, and the loop was not the part that failed.
            pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in fid]).to_csv(
                os.path.join(OUT, "_partial_fidelity.csv"), index=False)

    df = write_fidelity(fid)
    pd.DataFrame(prof).drop_duplicates().to_csv(
        os.path.join(OUT, "V3_TRIGGER_PROFILE_ALL324.csv"), index=False)

    # ---------- 3. budget feasibility over all 324 cells ----------
    budget_all324()
    print(f"\n  wrote 05b_* ({len(df)} fidelity rows)")


if __name__ == "__main__":
    main()
