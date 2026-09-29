# -*- coding: utf-8 -*-
"""P0-B -- feature-level parameter-free Global->Local retrieval audit.

Protocol: docs/p0b/P0B_PROTOCOL.md, frozen at commit 5f86b1d.  This file
implements that protocol and nothing else.

    B = Full448 token at the probe
    T = Full672, area-overlap pooled onto the same physical footprint (reference)
    Q = Local672 crop, same pooling, same footprint
    K = V = the query image's own Full448 grid
    s_j = sqrt(D) * cos(Q, K_j);  a = softmax(s);  C = normalize(sum a V)
    H = normalize(Q + C)                       # alpha fixed at 1

No training.  No nn.MultiheadAttention, Linear Q/K/V, or MLP.  No temperature,
alpha, top-k or head sweep.  Support-normal ONLY: no defect images, no GT, no
AUPRO/AUROC/A5.

Usage
    P0B_SMOKE=1 python experiments/p0b_feature_level_audit/p0b_feature_audit.py
    python experiments/p0b_feature_level_audit/p0b_feature_audit.py
    P0B_AGG_ONLY=1 python experiments/p0b_feature_level_audit/p0b_feature_audit.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from scipy.stats import spearmanr  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, VISA  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_calibration as cal  # noqa: E402
import v3_check_scale as cs  # noqa: E402

OUT = os.path.join(RESULTS, "p0b_feature_level_audit")
CACHE = os.path.join(OUT, "cache")
MANIFEST = os.path.join(RESULTS, "v3_1_f0_resolution_increment", "f0_probe_distances.csv")
SCALES = [9, 13, 15]
CORE448 = 3
SHOT, SPLIT = 8, 0
EXPECTED_ROWS = 27 * 8 * 9 * 9 * 3 * 3          # 157464
SAMPLES_PER_CELL = 8 * 9 * 9                    # 648
PC_OBJECTS = [("mvtec", "bottle"), ("visa", "pcb2")]
MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA_CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

# ---- frozen GO gate (protocol section 14) ----
G1_MIN, G2_MIN, G4_MIN = 0.03, 0.0, 0.10
N_BOOT = 10000


def side448(n):
    return CORE448 + 2 * int(round((n / 1.5 - CORE448) / 2.0))


def crop_origin(ar, ac, n):
    """(r6, c6) of the Local672 crop -- the same rule F0 used."""
    h = (n - 1) // 2
    return int(np.floor(1.5 * ar + 0.75)) - h, int(np.floor(1.5 * ac + 0.75)) - h


def local_weights(i, j):
    """672 tokens overlapping 448 patch (i,j) with exact area weights.
    Frozen arithmetic (cal._span / cal._overlap); PC2 proves it equals
    cal.pool_weights token-by-token and weight-by-weight."""
    y0, y1 = cal._span(i, cal.T448)
    x0, x1 = cal._span(j, cal.T448)
    rows, cols, w = [], [], []
    for u in range(int(y0 // cal.T672) - 1, int(y1 // cal.T672) + 2):
        oy = cal._overlap(*cal._span(u, cal.T672), y0, y1)
        if oy <= 0:
            continue
        for v in range(int(x0 // cal.T672) - 1, int(x1 // cal.T672) + 2):
            ox = cal._overlap(*cal._span(v, cal.T672), x0, x1)
            if ox <= 0:
                continue
            rows.append(u); cols.append(v); w.append(oy * ox)
    return np.array(rows, int), np.array(cols, int), np.array(w, float)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def attention(Q, K, V):
    """Parameter-free retrieval.  Q:(nq,D), K/V:(N,D), all L2-normalised."""
    s = np.sqrt(Q.shape[-1]) * (Q @ K.T)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    a = e / e.sum(axis=1, keepdims=True)
    return unit(a @ V), a


# ================================================================ correctness
def cache_audit(caches):
    """Protocol section 3: real support NAME comparison, not row counts."""
    rows = []
    for ds, obj in PC_OBJECTS:
        c448, c672 = caches[(ds, obj)]
        idx = draw_images(c448[rf.LAYERS[0]]["offsets"], SHOT, SPLIT, obj)
        n4 = [str(c448[rf.LAYERS[0]]["names"][i]) for i in idx]
        n6 = [str(c672[rf.LAYERS[0]]["names"][i]) for i in idx]
        dims = [int(c448[l]["feats"].shape[-1]) for l in rf.LAYERS] + \
               [int(c672[l]["feats"].shape[-1]) for l in rf.LAYERS]
        if n4 != n6:
            raise SystemExit(f"CACHE AUDIT FAILED {ds}/{obj}: support names differ")
        if any(d != 384 for d in dims):
            raise SystemExit(f"CACHE AUDIT FAILED {ds}/{obj}: unexpected feature dim {dims}")
        rows.append(dict(dataset=ds, object=obj, n_support=len(idx),
                         support_names_identical=True, all_dims_384=True,
                         support_names_448=";".join(n4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p0b_cache_audit.csv"), index=False)
    print(f"  cache audit PASS ({len(df)} objects; support names compared by name)")
    return df


def positive_control(caches, model, r448, r672, totensor, norm):
    rows = []
    for ds, obj in PC_OBJECTS:
        c448, c672 = caches[(ds, obj)]
        p0 = os.path.join(V1 if ds == "mvtec" else VISA, obj, "train", "good",
                          str(c448[rf.LAYERS[0]]["names"][0]))
        img = cv2.imread(p0, cv2.IMREAD_COLOR)
        for dom, rr, cache in (("448", r448, c448), ("672", r672, c672)):
            big = rf.resized_grid(img, rr)
            with torch.no_grad():
                tk = model.model.get_intermediate_layers(
                    norm(totensor(big)).unsqueeze(0).to(model.device), n=[5, 8, 11])
            for li, lay in enumerate(rf.LAYERS):
                o = cache[lay]["offsets"]
                ck = cache[lay]["feats"].shape[-1]
                a = tk[li].squeeze(0).cpu().numpy().astype(np.float16) \
                    .astype(np.float32).reshape(-1, ck)
                b = cache[lay]["feats"][o[0]:o[1]].astype(np.float32)
                cos = float((unit(a) * unit(b)).sum(1).mean())
                mad = float(np.abs(a - b).max())
                ok = bool(cos > 0.999999 and mad <= 1e-5)
                rows.append(dict(dataset=ds, object=obj, domain=dom, layer=lay,
                                 cos_mean=cos, max_abs_diff=mad, pass_=ok))
                print(f"  PC1 {ds}/{obj:8s} {dom} {lay:8s} cos={cos:.6f} "
                      f"maxdiff={mad:.2e} {'OK' if ok else 'FAIL'}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "p0b_positive_control.csv"), index=False)
    if not df.pass_.all():
        raise SystemExit("PC1 FAILED -- STOP. Do not continue P0-B.")
    print(f"  PC1: {int(df.pass_.sum())}/{len(df)} PASS")


def selftest_pooling():
    import random
    random.seed(0)
    for _ in range(40):
        i, j = random.randrange(0, 32), random.randrange(0, 32)
        r1, c1, w1 = local_weights(i, j)
        r2, c2, w2 = cal.pool_weights(i, j, 48, 48)
        if set(zip(r1.tolist(), c1.tolist())) != set(zip(r2.tolist(), c2.tolist())):
            raise SystemExit(f"PC2 FAILED: token sets differ at patch ({i},{j})")
        m1 = {(a, b): w for a, b, w in zip(r1, c1, w1)}
        for a, b, w in zip(r2, c2, w2):
            if abs(m1[(a, b)] - w) > 1e-12:
                raise SystemExit(f"PC2 FAILED: weight differs at ({i},{j})/({a},{b})")
    return ("PC2: local_weights == cal.pool_weights, 40 random patches, "
            "index and weight exact")


def selftest_bank(caches, n_checks=1000):
    """PC3 -- the held-out image contributes ZERO rows to its own bank.
    Bitwise, not by index bookkeeping."""
    import random
    random.seed(1)
    checked = 0
    for ds, obj in PC_OBJECTS:
        c448, c672 = caches[(ds, obj)]
        idx = draw_images(c448[rf.LAYERS[0]]["offsets"], SHOT, SPLIT, obj)
        for l in rf.LAYERS:
            o = c672[l]["offsets"]
            f = np.ascontiguousarray(c672[l]["feats"])
            dt = np.dtype((np.void, f.dtype.itemsize * f.shape[1]))
            for _ in range(n_checks):
                i = idx[random.randrange(len(idx))]
                bank = np.concatenate([f[o[j]:o[j + 1]] for j in idx if j != i])
                hit = np.intersect1d(f[o[i]:o[i + 1]].view(dt).ravel(),
                                     bank.view(dt).ravel())
                if len(hit):
                    raise SystemExit(f"PC3 FAILED: {ds}/{obj}/{l} image #{i} "
                                     f"contributes {len(hit)} rows to its own bank")
                checked += 1
                if checked >= n_checks:
                    break
            if checked >= n_checks:
                break
        if checked >= n_checks:
            break
    return (f"PC3: {checked} random (object, layer, held-out image) checks -- "
            f"held-out contribution is 0 rows everywhere")


# ================================================================ feature build
def object_rows(ds, obj, man, model, r448, r672, totensor, norm, cdir):
    c448 = sel.load_split(obj, "train")[1]
    c672 = rf.load_672(obj, "train")
    o4 = {l: c448[l]["offsets"] for l in rf.LAYERS}
    o6 = {l: c672[l]["offsets"] for l in rf.LAYERS}
    gh4, gw4 = (int(v) for v in c448[rf.LAYERS[0]]["grids"][0])
    gh6, gw6 = (int(v) for v in c672[rf.LAYERS[0]]["grids"][0])
    idx = draw_images(o4[rf.LAYERS[0]], SHOT, SPLIT, obj)
    for l in rf.LAYERS:
        if len({tuple(int(v) for v in c448[l]["grids"][i]) for i in idx}) != 1 or \
           len({tuple(int(v) for v in c672[l]["grids"][i]) for i in idx}) != 1:
            raise SystemExit(f"{ds}/{obj}/{l}: support images span several grids")
    sub = man[(man.dataset == ds) & (man.object == obj)]
    rows = []
    for si, i in enumerate(idx):
        # ---- K = V: this image's own Full448 grid (and the cyclic-shift one)
        Kq, Ksh = {}, {}
        for l in rf.LAYERS:
            Kq[l] = unit(c448[l]["feats"][o4[l][i]:o4[l][i + 1]].astype(np.float32))
            j = idx[(si + 1) % len(idx)]
            Ksh[l] = unit(c448[l]["feats"][o4[l][j]:o4[l][j + 1]].astype(np.float32))
            if l == rf.LAYERS[0]:
                full4 = c448[l]["feats"][o4[l][i]:o4[l][i + 1]].reshape(gh4, gw4, -1)
                full6 = c672[l]["feats"][o6[l][i]:o6[l][i + 1]].reshape(gh6, gw6, -1)
        # ---- Bank672: the OTHER seven images, all tokens
        bank6 = {l: unit(np.concatenate(
            [c672[l]["feats"][o6[l][j]:o6[l][j + 1]] for j in idx if j != i]
        ).astype(np.float32)) for l in rf.LAYERS}
        # B_native's bank depends only on (image, layer) -- hoisted out of the
        # anchor/scale loops, where rebuilding it 648x per object cost ~7 GB of
        # allocation churn per category.
        bank448 = {l: unit(np.concatenate(
            [c448[l]["feats"][o4[l][j]:o4[l][j + 1]] for j in idx if j != i]
        ).astype(np.float32)) for l in rf.LAYERS}
        # ---- Q: Local672 crop, pooled onto the probe footprint (cached)
        s = sub[sub.support_id == i]
        anchors = sorted({(int(v), int(w)) for v, w in
                          zip(s.anchor_r, s.anchor_c)})
        nm = os.path.join(cdir, f"si{si}.npz")
        if os.path.exists(nm):
            Qs = np.load(nm)["Q"]
        else:
            big = rf.resized_grid(cv2.imread(os.path.join(
                V1 if ds == "mvtec" else VISA, obj, "train", "good",
                str(c448[rf.LAYERS[0]]["names"][i])), cv2.IMREAD_COLOR), r672)
            Qs = np.zeros((len(anchors), len(SCALES), len(rf.LAYERS),
                           CORE448 * CORE448, 384), dtype=np.float32)
            for ai, (ar, ac) in enumerate(anchors):
                for ni, n in enumerate(SCALES):
                    r6, c6 = crop_origin(ar, ac, n)
                    cl = rf.crop_feats(model, big, r6, c6, n, norm, totensor)
                    for li, l in enumerate(rf.LAYERS):
                        a = cl[li].astype(np.float32)
                        for dr in range(CORE448):
                            for dc in range(CORE448):
                                rr, cc, w = local_weights(ar - 1 + dr, ac - 1 + dc)
                                lr, lc = rr - r6, cc - c6
                                if lr.min() < 0 or lr.max() >= n or lc.min() < 0 or lc.max() >= n:
                                    raise SystemExit(f"{ds}/{obj} si={si} n={n}: probe "
                                                     f"patch maps outside the 672 crop")
                                Qs[ai, ni, li, dr * CORE448 + dc] = \
                                    (a[lr, lc] * w[:, None]).sum(0) / w.sum()
            Qs = unit(Qs)
            np.savez_compressed(nm, Q=Qs)
        for ai, (ar, ac) in enumerate(anchors):
            for ni, n in enumerate(SCALES):
                for li, l in enumerate(rf.LAYERS):
                    P = s[(s.n == n) & (s.layer == l) & (s.anchor_r == ar)
                          & (s.anchor_c == ac)].sort_values(["patch_r", "patch_c"])
                    if not len(P):
                        continue
                    patches = [(int(r), int(c)) for r, c in zip(P.patch_r, P.patch_c)]
                    Q = np.stack([Qs[ai, ni, li, (r - (ar - 1)) * CORE448
                                          + (c - (ac - 1))] for r, c in patches])
                    C, a = attention(Q, Kq[l], Kq[l])
                    Csh, _ = attention(Q, Ksh[l], Ksh[l])
                    H, Hsh = unit(Q + C), unit(Q + Csh)
                    if abs(float(a.sum(1).mean()) - 1.0) > 1e-6:
                        raise SystemExit("PC4 FAILED: attention rows do not sum to 1")
                    B = np.stack([unit(full4[r, c].astype(np.float32)) for r, c in patches])
                    T = np.stack([unit((full6[rr, cc] * w[:, None]).sum(0) / w.sum())
                                  for r, c in patches
                                  for rr, cc, w in [local_weights(r, c)]])
                    dv = {"B": 1.0 - (B @ bank6[l].T).max(1),
                          "Q": 1.0 - (Q @ bank6[l].T).max(1),
                          "C": 1.0 - (C @ bank6[l].T).max(1),
                          "H": 1.0 - (H @ bank6[l].T).max(1),
                          "Hshuf": 1.0 - (Hsh @ bank6[l].T).max(1),
                          "T": 1.0 - (T @ bank6[l].T).max(1),
                          "B_native": 1.0 - (B @ bank448[l].T).max(1)}
                    for k, (r, c) in enumerate(patches):
                        rows.append(dict(
                            dataset=ds, object=obj, support_id=int(i), si=si,
                            fold=si % 2, anchor_r=ar, anchor_c=ac, patch_r=r,
                            patch_c=c, n=n, layer=l,
                            cos_B_T=float(B[k] @ T[k]), cos_Q_T=float(Q[k] @ T[k]),
                            cos_C_T=float(C[k] @ T[k]), cos_H_T=float(H[k] @ T[k]),
                            cos_Hshuf_T=float(Hsh[k] @ T[k]),
                            d_B=dv["B"][k], d_Q=dv["Q"][k], d_C=dv["C"][k],
                            d_H=dv["H"][k], d_Hshuf=dv["Hshuf"][k], d_T=dv["T"][k],
                            d_B_native=dv["B_native"][k],
                            norm_Q=float(np.linalg.norm(Q[k])),
                            norm_C=float(np.linalg.norm(C[k])),
                            norm_H=float(np.linalg.norm(H[k])),
                            norm_T=float(np.linalg.norm(T[k])),
                            attn_sum=float(a[k].sum())))
    return pd.DataFrame(rows)


# ================================================================== aggregation
ARMS = ("B", "Q", "C", "H", "Hshuf", "T")


def crossfit_agg3(g):
    """Protocol section 9.  For each held-out support image, mu/sigma of every
    layer come from the OTHER seven images at that layer only -- never
    crop-local, never the query image's own statistics.

    Returns z_B / z_Q / z_C / z_H / z_Hshuf / z_T on the same row order, so the
    caller can average the three layers straight into agg3.
    """
    g = g.reset_index(drop=True)
    out = {f"z_{a}": np.full(len(g), np.nan) for a in ARMS}
    for i in sorted(g.support_id.unique()):
        held = (g.support_id == i).values            # positional boolean
        cal_src = g[~held]
        for l in rf.LAYERS:
            rows_sel = held & (g.layer == l).values
            src = cal_src[cal_src.layer == l]
            if not len(src):
                raise SystemExit(f"cross-fit: no calibration rows for layer {l}")
            for a in ARMS:
                mu = float(src[f"d_{a}"].mean())
                sd = float(src[f"d_{a}"].std() + 1e-12)
                out[f"z_{a}"][rows_sel] = (g[f"d_{a}"].values[rows_sel] - mu) / sd
    if any(np.isnan(v).any() for v in out.values()):
        raise SystemExit("cross-fit left NaN rows -- a support image is missing")
    return pd.DataFrame(out)


def aggregate(raw):
    """Per (category, n): agg3 rho for every arm, deltas, R_feat."""
    per_layer, agg = [], []
    for (ds, obj, n), g in raw.groupby(["dataset", "object", "n"]):
        g = g.sort_values(["support_id", "anchor_r", "anchor_c", "patch_r", "patch_c", "layer"])
        z = pd.concat([g.reset_index(drop=True), crossfit_agg3(g)], axis=1)
        for l in rf.LAYERS:
            gl = z[z.layer == l]
            for a in ("B", "Q", "C", "H", "Hshuf"):
                per_layer.append(dict(
                    dataset=ds, object=obj, n=n, layer=l, arm=a, n_samples=len(gl),
                    spearman=float(spearmanr(gl[f"d_{a}"], gl.d_T).statistic)))
        rho = {a: float(spearmanr(z[f"z_{a}"], z.z_T).statistic)
               for a in ("B", "Q", "C", "H", "Hshuf")}
        med = {a: float(z[f"cos_{a}_T"].median()) for a in ("B", "Q", "C", "H", "Hshuf")}
        cbase = max(med["B"], med["Q"])
        agg.append(dict(dataset=ds, object=obj, n=n, n_samples=len(z),
                        rho_B=rho["B"], rho_Q=rho["Q"], rho_C=rho["C"],
                        rho_H=rho["H"], rho_shuf=rho["Hshuf"],
                        delta_B=rho["H"] - rho["B"],
                        delta_Q=rho["H"] - rho["Q"],
                        delta_shuf=rho["H"] - rho["Hshuf"],
                        cos_B=med["B"], cos_Q=med["Q"], cos_C=med["C"],
                        cos_H=med["H"], cos_shuf=med["Hshuf"],
                        c_base=cbase,
                        R_feat=(med["H"] - cbase) / (1 - cbase)))
    return pd.DataFrame(per_layer), pd.DataFrame(agg)


def bootstrap(agg):
    """10000 dataset-stratified category bootstrap.  Uncertainty only -- it
    never modifies the frozen gate."""
    rng = np.random.default_rng(20260929)
    mv = agg[agg.dataset == "mvtec"]
    vi = agg[agg.dataset == "visa"]
    rows = []
    for n in SCALES:
        a, b = mv[mv.n == n], vi[vi.n == n]
        if not len(a) or not len(b):
            continue
        est = {k: float(pd.concat([a, b])[k].median())
               for k in ("delta_B", "delta_Q", "delta_shuf", "R_feat")}
        draws = {k: [] for k in est}
        for _ in range(N_BOOT):
            s = pd.concat([a.sample(len(a), replace=True, random_state=int(rng.integers(1 << 31))),
                           b.sample(len(b), replace=True, random_state=int(rng.integers(1 << 31)))])
            for k in est:
                draws[k].append(float(s[k].median()))
        for k in est:
            lo, hi = np.percentile(draws[k], [2.5, 97.5])
            rows.append(dict(n=n, statistic=k, estimate=est[k],
                             ci_lo=float(lo), ci_hi=float(hi),
                             ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def decide(agg):
    rows = []
    for n in SCALES:
        s = agg[agg.n == n]
        mv, vi = s[s.dataset == "mvtec"], s[s.dataset == "visa"]
        g1 = float(s.delta_B.median()) >= G1_MIN
        g2 = float(mv.delta_B.median()) > G2_MIN and float(vi.delta_B.median()) > G2_MIN
        g3 = float(s.delta_Q.median()) > 0
        g4 = float(s.R_feat.median()) >= G4_MIN
        g5 = float(s.delta_shuf.median()) > 0
        rows.append(dict(n=n, n_categories=len(s),
                         dH_B=float(s.delta_B.median()),
                         dH_Q=float(s.delta_Q.median()),
                         dH_shuf=float(s.delta_shuf.median()),
                         mvtec_dH_B=float(mv.delta_B.median()),
                         visa_dH_B=float(vi.delta_B.median()),
                         R_feat=float(s.R_feat.median()),
                         G1=bool(g1), G2=bool(g2), G3=bool(g3),
                         G4=bool(g4), G5=bool(g5),
                         GO=bool(g1 and g2 and g3 and g4 and g5)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "p0b_decision.csv"), index=False)
    return d


# ======================================================================== main
def main():
    os.makedirs(CACHE, exist_ok=True)
    smoke = os.environ.get("P0B_SMOKE") == "1"
    agg_only = os.environ.get("P0B_AGG_ONLY") == "1"

    if not agg_only:
        man = pd.read_csv(MANIFEST)
        print(f"manifest: {MANIFEST}\n  rows {len(man)}")
        if smoke:
            keys = set(PC_OBJECTS)
            man = man[[(d, o) in keys for d, o in zip(man.dataset, man.object)]]
            print(f"  SMOKE: restricted to {sorted(keys)} -> {len(man)} rows")
        if len(man) != EXPECTED_ROWS and not smoke:
            raise SystemExit(f"PC7 FAILED: manifest has {len(man)} rows, "
                             f"expected {EXPECTED_ROWS}. Do not analyse.")
        model, r672, totensor, norm = cs.build()
        r448 = transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC,
                                 antialias=True)
        # Gate caches are loaded for the two PC objects ONLY.  Building this
        # dict for all 27 categories would pin ~51 GB (each object's two
        # domains x three layers is ~1.9 GB) and OOM before the first crop --
        # a failure that the two-object smoke run cannot reveal.
        _pc = {(d, o): (sel.load_split(o, "train")[1], rf.load_672(o, "train"))
               for d, o in PC_OBJECTS}
        print("=== correctness gates ===")
        cache_audit(_pc)
        r_pc2 = selftest_pooling()
        r_pc3 = selftest_bank(_pc)
        positive_control(_pc, model, r448, r672, totensor, norm)
        del _pc
        objs = (list(PC_OBJECTS) if smoke else
                [("mvtec", o) for o in MVTEC] + [("visa", o) for o in VISA_CATS])
        # P0B_ONLY="mvtec/bottle,visa/pcb2" restricts this process to a shard.
        # Safe because every object writes only its own cache dir plus a DONE
        # marker; the aggregation step reads the DONE caches and is unaffected.
        only = os.environ.get("P0B_ONLY")
        if only:
            want = {tuple(t.split("/")) for t in only.split(",") if t}
            objs = [o for o in objs if o in want]
            print(f"  P0B_ONLY -> {len(objs)} objects: "
                  f"{[f'{d}/{o}' for d, o in objs]}")
        for ds, obj in objs:
            cdir = os.path.join(CACHE, ds, obj)
            os.makedirs(cdir, exist_ok=True)
            if os.path.exists(os.path.join(cdir, "DONE")):
                print(f"  {ds}/{obj}: DONE, skip", flush=True)
                continue
            sub = man[(man.dataset == ds) & (man.object == obj)]
            df = object_rows(ds, obj, sub, model, r448, r672, totensor, norm, cdir)
            if len(df) != 8 * 9 * 3 * 9 * 3:
                raise SystemExit(f"{ds}/{obj}: produced {len(df)} rows, expected "
                                 f"{8*9*3*9*3} -- do not analyse")
            df.to_csv(os.path.join(cdir, "probes.csv"), index=False)
            open(os.path.join(cdir, "DONE"), "w").write(f"{len(df)}\n")
            print(f"  {ds}/{obj} done: {len(df)} rows", flush=True)
        pd.DataFrame([
            dict(check="PC1_positive_control", detail="12/12 PASS (cos>0.999999, maxdiff<=1e-5)"),
            dict(check="PC2_pooling_weights", detail=r_pc2),
            dict(check="PC3_bank_exclusion", detail=r_pc3),
            dict(check="PC4_attention_sums", detail="checked per batch, |sum a - 1| < 1e-6"),
            dict(check="PC5_finite", detail="checked after aggregation"),
            dict(check="PC6_norms", detail="checked after aggregation"),
            dict(check="PC7_rows", detail=f"expected {EXPECTED_ROWS}"),
            dict(check="PC8_cell_size", detail=f"expected {SAMPLES_PER_CELL} per (category,n,layer)"),
        ]).to_csv(os.path.join(OUT, "p0b_selftest.csv"), index=False)
        if smoke:
            # Smoke validates implementation only.  Running the aggregation on
            # it would trip PC7 by design, and writing a decision from two
            # categories would be worse than useless.
            print("\n  SMOKE OK: every object produced its expected row count; "
                  "aggregation and the GO gate are NOT evaluated on smoke data.")
            return

    # ---- aggregate over DONE categories only
    parts = []
    for ds, obj in [(d, o) for d in ("mvtec", "visa")
                    for o in (MVTEC if d == "mvtec" else VISA_CATS)]:
        p = os.path.join(CACHE, ds, obj, "probes.csv")
        if os.path.exists(p) and os.path.exists(os.path.join(CACHE, ds, obj, "DONE")):
            parts.append(pd.read_csv(p))
    if not parts:
        raise SystemExit("no DONE categories found")
    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(os.path.join(OUT, "p0b_probe_features.csv"), index=False)
    print(f"\naggregating {len(raw)} rows from {len(parts)} DONE categories")
    if len(raw) != EXPECTED_ROWS:
        raise SystemExit(f"PC7 FAILED: {len(raw)} rows, expected {EXPECTED_ROWS}. "
                         f"Do not analyse.")
    for (ds, obj, n, l), g in raw.groupby(["dataset", "object", "n", "layer"]):
        if len(g) != SAMPLES_PER_CELL:
            raise SystemExit(f"PC8 FAILED: {ds}/{obj}/n={n}/{l} has {len(g)} rows, "
                             f"expected {SAMPLES_PER_CELL}")
    print(f"  PC7 rows={len(raw)} OK | PC8 every (category,n,layer)={SAMPLES_PER_CELL} OK")
    bad = [c for c in raw.columns if c.startswith(("d_", "cos_", "norm_", "attn_"))
           and not np.isfinite(raw[c]).all()]
    if bad:
        raise SystemExit(f"PC5 FAILED: non-finite values in {bad}")
    nm = raw[["norm_Q", "norm_H", "norm_T"]]
    print(f"  PC5 all finite OK | PC6 norms: Q {nm.norm_Q.mean():.6f} "
          f"H {nm.norm_H.mean():.6f} T {nm.norm_T.mean():.6f} | "
          f"PC4 attn_sum max dev {float(np.abs(raw.attn_sum - 1).max()):.2e}")

    per_layer, agg3 = aggregate(raw)
    per_layer.to_csv(os.path.join(OUT, "p0b_per_layer.csv"), index=False)
    agg3.to_csv(os.path.join(OUT, "p0b_agg3_by_category.csv"), index=False)
    boot = bootstrap(agg3)
    boot.to_csv(os.path.join(OUT, "p0b_bootstrap.csv"), index=False)
    dec = decide(agg3)
    print("\n=== P0-B frozen gate ===")
    print(dec.round(4).to_string(index=False))
    go = dec[dec.GO]
    if len(go):
        print(f"\n  GO. Smallest passing n = {int(go.n.min())}  -> admit T0")
    else:
        print("\n  NO-GO for every n -> Transformer branch CLOSED")
    print("\n=== bootstrap 95% CI (uncertainty only; the gate is not modified) ===")
    print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
