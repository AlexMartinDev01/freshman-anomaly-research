# -*- coding: utf-8 -*-
"""Stage 5C -- sealed-confound rerun of the halo feasibility study.

This is a CLOSING experiment, not further exploration.  It fixes the three
things that stopped Stage 5B from being quotable, and adds nothing else:

  A. FIXED PHYSICAL CORE.  In Stage 5/5B the crop origin was clamped
     (`r0 = min(max(0, ar - h), gh6 - n)`) so that the 5x5 core actually
     COMPARED drifted with n -- e.g. for the first anchor on a 48-row grid the
     core top row was 11 at n=5 but 14 at n=33.  The late rise in the curve
     therefore mixed "more halo" with "core moved toward the image centre".
     Here core centres are chosen up front so that no n in N_CTX can ever
     clip, and the core is bit-identical across all 15 halo sizes.

  B. THREE-LAYER POSITIVE CONTROL.  Stage 5B proved a full re-forward
     reproduces the cache for `final` only, yet `mid`/`midlate` are exactly
     the layers whose fidelity is in question.  All three are checked now.

  C. DETECTOR-CONSISTENT AGGREGATE.  Reporting three per-layer Spearmans
     invites "maybe the agg_all3 the detector actually consumes recovers".
     `agg3` is computed explicitly and reported alongside.

Also: the trigger profile is renamed to say what it is (k8/s0 cohort, 27
rows -- the 324-cell budget table lives in 05b_budget_all324.csv).

Support-only throughout: no test anomaly, no GT, no AUPRO, no A5.
n_ctx is NOT extended beyond 33.

Usage: OMP_NUM_THREADS=2 python experiments/model_v0/v3_stage5c_context.py
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
ANCH = 2                      # ANCH x ANCH fixed core centres per support image
SHOT, SPLIT = 8, 0            # same support cohort as Stage 5B, for comparability
CASES = [("mvtec", "bottle"), ("mvtec", "capsule"),
         ("visa", "pcb2"), ("visa", "macaroni1")]

# ---- pre-registered stop rule, fixed BEFORE this script was run ----
DECISION_RHO = 0.50           # median over objects of the agg3 Spearman
DECISION_RET = 0.50           # mean over the 324 cells of trigger-mass retention

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def excl_mask(u, v, gh, gw, base, n, bank_size):
    """allowed[token] for a query at (u,v): ban its own image's 3x3 physical
    block (addendum 2 rule); other images and far same-image patches stay.

    `base` is the column offset of the query image's own bank block -- passing
    0 here (instead of si * n_tok) leaves 7 of 8 samples self-matched, which is
    the bug Stage 5B carried until it was caught.
    """
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


def core_centers(extent, k=ANCH):
    """Core centres at which NO n in N_CTX clips.

    The crop is centred on the core: r0 = cr - (n-1)//2, spanning [r0, r0+n).
    Demanding 0 <= r0 and r0+n <= extent for EVERY n in N_CTX gives

        cr >= (n_max - 1) // 2        and        cr <= extent - (n_max + 1) // 2

    -- both independent of which n is under evaluation.  Because r0 is never
    clamped, the 5x5 core compared is bit-identical for all 15 halo sizes.
    """
    h = (max(N_CTX) - 1) // 2
    lo, hi = h, extent - h - 1
    if hi < lo:
        raise SystemExit(
            f"extent {extent} is too small to hold an unclipped core for "
            f"n up to {max(N_CTX)}: valid centres would be [{lo}, {hi}].")
    span = hi - lo
    return [lo + span * (i + 1) // (k + 1) for i in range(k)]


def agg3_spearman(fid, anchor=None):
    """Detector-consistent aggregate fidelity.

    The detector consumes agg_all3: per-layer 1-NN distance, per-layer
    z-score, then the mean over layers.  Fidelity is scored the same way --
    each layer z-scored with FULL-REFERENCE statistics (mu, sd of the pooled
    d_full for that layer) so that halo and full are put on one scale, then
    averaged over the three layers, then Spearman'd.  Reporting only per-layer
    values would leave open whether the aggregate happens to recover.

    `anchor` restricts the pool to one anchor INDEX (i_r, i_c), ranked within
    each object.  Absolute core coordinates are NOT comparable across objects
    (the valid band depends on each object's grid: MVTec cores sit at 21/26,
    VisA ones at 22/28, 29/42, 31/46, ...), so filtering by coordinate silently
    compares different OBJECT SUBSETS.  Pooling anchors before z-scoring is
    also not the same as scoring each anchor separately, because between-anchor
    offsets enter the pooled rank vector.  Both conventions are reported by
    evaluate_decision(); neither changes the verdict under trigger-mass
    retention.
    """
    if anchor is not None:
        keep = {}
        for r in fid:
            key = (r["dataset"], r["object"])
            keep.setdefault(key, [set(), set()])
        for r in fid:
            rs, cs = keep[(r["dataset"], r["object"])]
            rs.add(r["core_r"]); cs.add(r["core_c"])
        fid = [r for r in fid
               if (sorted(keep[(r["dataset"], r["object"])][0]).index(r["core_r"]),
                   sorted(keep[(r["dataset"], r["object"])][1]).index(r["core_c"]))
               == tuple(anchor)]
    g = {}
    for r in fid:
        g.setdefault((r["dataset"], r["object"], r["n_ctx"]), {}) \
         .setdefault(r["sample_id"], {})[r["layer"]] = (
             np.asarray(r["_dH"], dtype=np.float64),
             np.asarray(r["_dF"], dtype=np.float64))
    rows = []
    for (ds, obj, n), samples in g.items():
        layers = sorted({lay for s in samples.values() for lay in s})
        stat = {}
        for lay in layers:
            allf = np.concatenate([s[lay][1] for s in samples.values() if lay in s])
            stat[lay] = (float(allf.mean()), float(allf.std() + 1e-12))
        zh, zf = [], []
        for _, s in sorted(samples.items()):
            if len(s) != len(layers):
                raise SystemExit(f"{ds}/{obj}/n={n}: incomplete sample {sorted(s)}")
            zh.append(np.mean([(s[lay][0] - stat[lay][0]) / stat[lay][1]
                               for lay in layers], axis=0))
            zf.append(np.mean([(s[lay][1] - stat[lay][0]) / stat[lay][1]
                               for lay in layers], axis=0))
        zh, zf = np.concatenate(zh), np.concatenate(zf)
        rows.append(dict(dataset=ds, object=obj, n_ctx=n,
                         n_samples=len(samples), n_tokens=int(zh.size),
                         agg3_spearman=float(spearmanr(zh, zf).statistic)))
    return pd.DataFrame(rows)


def write_outputs(fid, prof):
    """Post-loop aggregation and writes.  Kept separate from main() so the
    whole tail can be exercised on synthetic rows in a second, instead of
    being validated by a 13-minute sweep that may not reach it."""
    # Vectors first: the sweep is the expensive part and must never depend on
    # the pandas work below succeeding.
    np.savez_compressed(
        os.path.join(OUT, "05c_fidelity_vectors.npz"),
        keys=np.array([f"{r['dataset']}|{r['object']}|{r['n_ctx']}|{r['layer']}"
                       f"|{r['sample_id']}" for r in fid]),
        dH=np.array([r["_dH"] for r in fid], dtype=np.float32),
        dF=np.array([r["_dF"] for r in fid], dtype=np.float32))

    df = pd.DataFrame(fid)
    # Pooled Spearman per (dataset, object, n_ctx, layer).
    # NOT groupby.apply: numpy walks a pandas Series positionally (seq[0]) while
    # Series.__getitem__ with an integer key is LABEL-based, so any group whose
    # index does not contain the label 0 raises KeyError: 0 -- and it silently
    # "works" for as long as the first group happens to start at row 0.
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
    df.to_csv(os.path.join(OUT, "05c_context_fidelity_fixed_core.csv"), index=False)

    agg = df.groupby(["dataset", "object", "shot", "n_ctx"]).agg(
        n_positions=("cos_median", "size"), cos_mean=("cos_mean", "mean"),
        cos_median=("cos_median", "median"), cos_p10=("cos_p10", "median"),
        feature_rel_error=("feature_rel_error", "median"),
        nn_spearman=("nn_spearman", "median")).reset_index()
    agg.to_csv(os.path.join(OUT, "05c_context_fidelity_by_category.csv"), index=False)

    df.groupby(["dataset", "layer", "n_ctx"]).agg(
        n_positions=("cos_median", "size"), cos_mean=("cos_mean", "mean"),
        cos_median=("cos_median", "median"), cos_p10=("cos_p10", "median"),
        feature_rel_error=("feature_rel_error", "median"),
        nn_spearman=("nn_spearman", "median")).reset_index().to_csv(
        os.path.join(OUT, "05c_context_fidelity_by_layer.csv"), index=False)

    agg3_spearman(fid).to_csv(os.path.join(OUT, "05c_agg3_fidelity.csv"), index=False)

    # 27 rows at k8/s0 -- named for what it is.  The 324-cell budget table is
    # 05b_budget_all324.csv and is reused, not recomputed.
    pd.DataFrame(prof).drop_duplicates().to_csv(
        os.path.join(OUT, "V3_TRIGGER_PROFILE_K8S0.csv"), index=False)

    part = os.path.join(OUT, "_partial_fidelity.csv")
    if os.path.exists(part):
        os.remove(part)
    return df


def retentions_from_diagnostics():
    """(trigger-mass, naive per-image) retention per n_ctx, recomputed straight
    from the frozen diagnostics.json files -- not read back from any CSV, so the
    number the stop rule rests on is independently sourced."""
    rows = []
    for ds, objs in (("mvtec", MVTEC), ("visa", VISA_CATS)):
        for obj in objs:
            for shot in (1, 2, 4, 8):
                for sp in range(3):
                    fp = os.path.join(RESULTS, "v3", ds, obj, f"k{shot}", f"s{sp}",
                                      "diagnostics.json")
                    if not os.path.exists(fp):
                        continue
                    with open(fp) as f:
                        ms = np.array([x["m"] for x in json.load(f).get("a3", [])])
                    trig = ms[ms > 0]
                    with np.load(os.path.join(
                            RESULTS, "cache_m1_dino672", "final",
                            f"dino672_{obj}_test.npz"), allow_pickle=True) as z:
                        u = {tuple(int(v) for v in r) for r in z["grids"]}
                    gh6, gw6 = u.pop()
                    B = 0.5 * gh6 * gw6
                    for n in N_CTX:
                        K = int(B // (n * n))
                        if len(trig):
                            rows.append(dict(
                                dataset=ds, object=obj, shot=shot, split=sp, n_ctx=n,
                                trigger_mass_retained=float(
                                    np.minimum(trig, K).sum() / trig.sum()),
                                naive_mean_per_image=float(
                                    (np.minimum(trig, K) / np.maximum(trig, 1)).mean())))
                        else:
                            rows.append(dict(
                                dataset=ds, object=obj, shot=shot, split=sp, n_ctx=n,
                                trigger_mass_retained=0.0, naive_mean_per_image=0.0))
    return pd.DataFrame(rows)


def evaluate_decision(fid, ret):
    """Apply the pre-registered stop rule.  No knob is tuned afterwards.

    Both aggregation conventions (pooled vs per-anchor) AND both readings of
    "retained" (trigger-mass vs mean-per-image) are reported, because each was
    a live ambiguity.  A verdict that depended on the choice would not be a
    verdict.
    """
    conv = {"agg3_pooled": agg3_spearman(fid)}
    for i_r in range(ANCH):
        for i_c in range(ANCH):
            conv[f"agg3_anchor{i_r}{i_c}"] = agg3_spearman(fid, anchor=(i_r, i_c))
    rows = []
    for n in N_CTX:
        d = dict(n_ctx=n)
        for name, a in conv.items():
            d[name] = float(a[a.n_ctx == n].agg3_spearman.median())
        d["retention_trigger_mass"] = float(
            ret[ret.n_ctx == n].trigger_mass_retained.mean())
        d["retention_naive_per_image"] = float(
            ret[ret.n_ctx == n].naive_mean_per_image.mean())
        d["meets_agg3_pooled"] = bool(d["agg3_pooled"] >= DECISION_RHO)
        d["meets_retention_mass"] = bool(d["retention_trigger_mass"] >= DECISION_RET)
        d["BOTH_pooled_mass"] = bool(d["meets_agg3_pooled"] and d["meets_retention_mass"])
        rows.append(d)
    t = pd.DataFrame(rows)
    # verdict under every convention x reading combination
    acols = [c for c in t.columns if c.startswith("agg3_")]
    combos = {}
    for a in acols:
        for r in ("retention_trigger_mass", "retention_naive_per_image"):
            hit = t[(t[a] >= DECISION_RHO) & (t[r] >= DECISION_RET)]
            combos[f"{a} & {r}"] = list(map(int, hit.n_ctx)) if len(hit) else []
    t.to_csv(os.path.join(OUT, "05c_halo_decision.csv"), index=False)
    print("\n  === pre-registered stop rule ===")
    print(f"    need agg3 Spearman median >= {DECISION_RHO:.2f} "
          f"AND retention >= {DECISION_RET:.2f}")
    print(t.round(3).to_string(index=False))
    print("\n  --- feasible n_ctx under every convention x retention reading ---")
    for k, v in combos.items():
        print(f"    {k:52s} -> {v if v else 'NONE'}")
    stop = all(len(v) == 0 for v in combos.values())
    if stop:
        print("\n    STOP: no n_ctx <= 33 satisfies both conditions under ANY "
              "convention.")
        print("    => the pure-halo route is closed; halo is NOT extended further,")
        print("       and no defect performance is consulted.")
    else:
        print("\n    A feasible halo exists under at least one convention -- "
              "resolve the convention BEFORE proceeding.")
    return t, stop


def load_fid_from_npz():
    """Rebuild the row dicts from 05c_fidelity_vectors.npz so the decision can
    be re-derived without re-running the sweep and without a model."""
    z = np.load(os.path.join(OUT, "05c_fidelity_vectors.npz"), allow_pickle=True)
    out = []
    for k, h, f in zip([str(x) for x in z["keys"]], z["dH"], z["dF"]):
        ds, obj, n, lay, sid = k.split("|")
        i, cr, cc = sid.split("_")
        out.append(dict(dataset=ds, object=obj, n_ctx=int(n), layer=lay,
                        sample_id=sid, support_image_id=int(i),
                        core_r=int(cr), core_c=int(cc),
                        _dH=h.tolist(), _dF=f.tolist()))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("S5C_DECIDE_ONLY") == "1":
        evaluate_decision(load_fid_from_npz(), retentions_from_diagnostics())
        return
    model, resize, totensor, norm = cs.build()

    # ---------- 1. positive control: all three layers ----------
    print("=== positive control: full re-forward vs cache, mid/midlate/final ===")
    ident = []
    for ds, obj in CASES:
        c672 = rf.load_672(obj, "train")
        p0 = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train", "good",
                          str(c672[rf.LAYERS[0]]["names"][0]))
        big = rf.resized_grid(cv2.imread(p0, cv2.IMREAD_COLOR), resize)
        with torch.no_grad():
            tk = model.model.get_intermediate_layers(
                norm(totensor(big)).unsqueeze(0).to(model.device), n=[5, 8, 11])
        for li, lay in enumerate(rf.LAYERS):
            o = c672[lay]["offsets"]
            ck = c672[lay]["feats"].shape[-1]
            cached = c672[lay]["feats"][o[0]:o[1]]
            rer = tk[li].squeeze(0).cpu().numpy().astype(np.float16)
            a = rer.astype(np.float32).reshape(-1, ck)
            b = cached.astype(np.float32)
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
            bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
            cos = float((an * bn).sum(1).mean())
            mad = float(np.abs(a - b).max())
            ok = cos > 0.999 and mad < 0.05
            ident.append(dict(dataset=ds, object=obj, layer=lay,
                              n_intermediate=[5, 8, 11][li], n_tokens=int(a.shape[0]),
                              cos_mean=cos, max_abs_diff=mad, pass_=bool(ok)))
            print(f"  {ds}/{obj:9s} {lay:8s} cos={cos:.6f} maxdiff={mad:.5f} "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
    pd.DataFrame(ident).to_csv(
        os.path.join(OUT, "05c_full_cache_identity_all_layers.csv"), index=False)
    if not all(r["pass_"] for r in ident):
        raise SystemExit("POSITIVE CONTROL FAILED -- cache/preprocess mismatch; "
                         "context fidelity numbers would be uninterpretable.")

    # ---------- 2. fixed-core fidelity, all 27 categories ----------
    fid, prof = [], []
    for objs, ds in ((MVTEC, "mvtec"), (VISA_CATS, "visa")):
        for obj in objs:
            _, c448 = sel.load_split(obj, "train")
            c672 = rf.load_672(obj, "train")
            tro = c448[rf.LAYERS[0]]["offsets"]
            idx = draw_images(tro, SHOT, SPLIT, obj)
            o6 = c672[rf.LAYERS[0]]["offsets"]
            gs = {tuple(int(v) for v in c672[rf.LAYERS[0]]["grids"][i]) for i in idx}
            if len(gs) != 1:
                print(f"  {ds}/{obj}: differing support grids, skip"); continue
            gh6, gw6 = gs.pop()
            n_tok = gh6 * gw6
            bank = {l: l2n(torch.from_numpy(np.concatenate(
                [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
            ).astype(np.float32)).to(DEV)) for l in rf.LAYERS}
            crs, ccs = core_centers(gh6), core_centers(gw6)
            for si, i in enumerate(idx):
                base_off = si * n_tok
                assert base_off + n_tok <= len(idx) * n_tok
                p = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train",
                                 "good", str(c672[rf.LAYERS[0]]["names"][i]))
                big = rf.resized_grid(cv2.imread(p, cv2.IMREAD_COLOR), resize)
                full = {l: c672[l]["feats"][o6[i]:o6[i + 1]].reshape(gh6, gw6, -1)
                        for l in rf.LAYERS}
                for cr in crs:
                    for cc in ccs:
                        # the reference 5x5 -- one array, reused for every n
                        ref = {l: full[l][cr - CORE // 2:cr + CORE // 2 + 1,
                                         cc - CORE // 2:cc + CORE // 2 + 1]
                               for l in rf.LAYERS}
                        for n in N_CTX:
                            hh, k0 = (n - 1) // 2, (n - CORE) // 2
                            r0, c0 = cr - hh, cc - hh
                            assert 0 <= r0 and r0 + n <= gh6 and 0 <= c0 and c0 + n <= gw6
                            assert r0 + k0 == cr - CORE // 2
                            assert c0 + k0 == cc - CORE // 2
                            crop = rf.crop_feats(model, big, r0, c0, n, norm, totensor)
                            sid = f"{int(i)}_{cr}_{cc}"
                            for li, lay in enumerate(rf.LAYERS):
                                fh = crop[li][k0:k0 + CORE, k0:k0 + CORE].reshape(
                                    -1, crop[li].shape[-1])
                                ff = ref[lay].reshape(-1, crop[li].shape[-1])
                                an = fh / (np.linalg.norm(fh, axis=1, keepdims=True) + 1e-12)
                                bn = ff / (np.linalg.norm(ff, axis=1, keepdims=True) + 1e-12)
                                cos = (an * bn).sum(1)
                                rel = np.linalg.norm(fh - ff, axis=1) / (
                                    np.linalg.norm(ff, axis=1) + 1e-12)
                                qh = l2n(torch.from_numpy(fh.astype(np.float32)).to(DEV))
                                qf = l2n(torch.from_numpy(ff.astype(np.float32)).to(DEV))
                                B = bank[lay]
                                Dh = (1.0 - qh @ B.T).cpu().numpy()
                                Df = (1.0 - qf @ B.T).cpu().numpy()
                                dH, dF = [], []
                                for t in range(CORE * CORE):
                                    al = excl_mask(cr - CORE // 2 + t // CORE,
                                                   cc - CORE // 2 + t % CORE,
                                                   gh6, gw6, base_off, n_tok,
                                                   Df.shape[1])
                                    dH.append(Dh[t][al].min())
                                    dF.append(Df[t][al].min())
                                dH, dF = np.array(dH), np.array(dF)
                                fid.append(dict(
                                    dataset=ds, object=obj, shot=SHOT, split=SPLIT,
                                    support_image_id=int(i), sample_id=sid,
                                    core_r=cr, core_c=cc, layer=lay, n_ctx=n,
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
            v3d = os.path.join(RESULTS, "v3", ds, obj, f"k{SHOT}", f"s{SPLIT}")
            ms = []
            for r, _, f in os.walk(v3d):
                if "diagnostics.json" in f:
                    dg = json.load(open(os.path.join(r, "diagnostics.json")))
                    ms += [x["m"] for x in dg.get("a3", [])]
            if ms:
                prof.append(dict(dataset=ds, object=obj, shot=SHOT, split=SPLIT,
                                 n_images=len(ms),
                                 n_triggered=int(sum(1 for x in ms if x > 0)),
                                 m_mean=float(np.mean(ms)), m_max=int(np.max(ms))))
            print(f"  {ds}/{obj} done (grid {gh6}x{gw6}, cores r={crs} c={ccs}, "
                  f"k={len(idx)})", flush=True)
            pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in fid]).to_csv(
                os.path.join(OUT, "_partial_fidelity.csv"), index=False)

    df = write_outputs(fid, prof)
    ret = retentions_from_diagnostics()
    ret.to_csv(os.path.join(OUT, "05c_budget_decision_input.csv"), index=False)
    evaluate_decision(fid, ret)
    print(f"\n  wrote 05c_* ({len(df)} fidelity rows)")


if __name__ == "__main__":
    main()
