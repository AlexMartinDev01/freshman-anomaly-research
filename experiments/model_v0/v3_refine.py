# -*- coding: utf-8 -*-
"""
V3 Stage 3/4 -- high-resolution crop refinement and writeback.

Implements docs/MODEL_V3_PREREGISTRATION.md sections 2.3 / 2.4 and
docs/MODEL_V3_PREREGISTRATION_ADDENDUM_1.md A1.2 / A1.3.

Two invariants drive the whole module:

  GEOMETRY   a crop is a token-aligned window of the GLOBALLY RESIZED image, so
             its tokens cover the same ground area as global-672 tokens and no
             second resize can creep in (v3_check_scale.py proves both).
  RESOLUTION the crop's query is matched against a 672 bank.  Matching a
             672-density query against the 448 bank would be a silent
             resolution-domain mismatch, so the cache key carries the
             resolution and `assert_resolution` guards it.

Normal-only affine calibration (section 2.4): the refined map's median and IQR
are matched to the 448 map's on NORMAL reference patches.  The obvious
construction -- the support images' own global-672 maps -- is degenerate,
because those images ARE the bank, so their distance to it is ~0.  The
calibration therefore reuses the same 3x3 self-neighbourhood exclusion as
section 2.2.2: a support patch may match anything except its own 3x3 block in
its own image.  Still normal-only: no GT, no test anomaly, no test-time tuning.

Usage: python experiments/model_v0/v3_refine.py --check
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\third_party\AnomalyDINO")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from src.backbones import get_model  # noqa: E402
from tail_calib import RESULTS  # noqa: E402
from gate_r2_layer_confirm import DEV, draw_images, l2n, nn_dist  # noqa: E402
from phase_m1_rescue import V1, VISA, load_cached  # noqa: E402
from v3_selector import rank_truncate  # noqa: E402

LAYERS = ["mid", "midlate", "final"]
C672 = os.path.join(RESULTS, "cache_m1_dino672")
PX_EDGE, STRIDE = 672, 14


def resolve(cache_dir, resolution):
    """Every 672 cache must advertise itself as such -- see the module docstring."""
    assert resolution == 672, f"only 672 caches are read here, got {resolution}"
    assert "dino672" in os.path.basename(cache_dir), \
        f"cache dir {cache_dir} is not a 672 cache"
    return cache_dir


def resized_grid(img_bgr, resize):
    """The globally resized image, cropped to a multiple of the stride.

    This is the ONE resize a crop may ever see; crops are token-aligned windows
    of THIS image, never re-resized (v3_check_scale.py proves both).
    """
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    big = resize(Image.fromarray(rgb))
    W, H = big.size
    cw, ch = W - W % STRIDE, H - H % STRIDE
    return big.crop((0, 0, cw, ch))


def load_672(obj, which):
    src = resolve(C672, 672)
    return {l: load_cached(src, l, f"dino672_{obj}_{which}") for l in LAYERS}


def window_672(r448, c448, n, gh672, gw672):
    """3x3 448-window starting at (r448, c448) -> a centred n x n 672 window.

    A 448 token spans [k*14, (k+1)*14) px at the 448 resize, i.e. [21k, 21k+21)
    px at the 672 resize (factor 1.5), so its centre sits at 672 px 21k+10.5,
    i.e. 672 token index (21k+10.5)/14.  The 3x3 window's centre is therefore
    at 672 token index 1.5*k + 2.25 for both axes.
    """
    cy = int(np.floor(1.5 * r448 + 2.25))
    cx = int(np.floor(1.5 * c448 + 2.25))
    h = (n - 1) // 2
    r0 = min(max(0, cy - h), gh672 - n)
    c0 = min(max(0, cx - h), gw672 - n)
    assert r0 >= 0 and c0 >= 0 and r0 + n <= gh672 and c0 + n <= gw672
    return r0, c0


def crop_feats(model, big, r0, c0, n, norm, totensor):
    """Features of an n x n token-aligned window of the resized image."""
    box = (c0 * STRIDE, r0 * STRIDE, (c0 + n) * STRIDE, (r0 + n) * STRIDE)
    t = norm(totensor(big.crop(box)))
    with torch.no_grad():
        tk = model.model.get_intermediate_layers(t.unsqueeze(0).to(model.device),
                                                 n=[5, 8, 11])
    return [x.squeeze(0).cpu().numpy().astype(np.float32).reshape(n, n, -1)
            for x in tk]


def agg3(feats_per_layer, bank_per_layer):
    """agg_all3 for one crop: per-layer 1-NN distance, z-score over the crop's
    own pixels, then average.  Self-normalisation keeps the three layers
    comparable without touching any statistic outside this crop."""
    Z = []
    for f, bank in zip(feats_per_layer, bank_per_layer):
        n = f.shape[0] * f.shape[1]
        q = l2n(torch.from_numpy(f.reshape(n, -1)).to(DEV))
        with torch.no_grad():
            d = nn_dist(q, bank).cpu().numpy()
        Z.append((d - d.mean()) / (d.std() + 1e-12))
    return np.mean(Z, axis=0)


def writeback(m0_flat, grid448, patches, refined, a, b):
    """Replace 448 patches covered by >=1 crop; overlaps take the MEAN.

    `patches` is a list of (idx_array_into_flat, values) aligned with `refined`.
    """
    out = np.array(m0_flat, dtype=np.float32, copy=True)
    acc = {}
    for idx, val in zip(patches, refined):
        v = a * np.asarray(val, dtype=np.float64) + b
        for j, f in enumerate(idx):
            acc.setdefault(int(f), []).append(float(v[j]))
    for f, vs in acc.items():
        out[f] = float(np.mean(vs))
    return out, len(acc)


def fit_affine(ref_norm, m0_norm):
    """a, b matching median and IQR of the refined normal map to the 448 one."""
    r, m = np.asarray(ref_norm, float), np.asarray(m0_norm, float)
    q1r, q3r = np.percentile(r, [25, 75])
    q1m, q3m = np.percentile(m, [25, 75])
    ir, im = q3r - q1r, q3m - q1m
    a = im / ir if ir > 1e-9 else 1.0
    b = np.median(m) - a * np.median(r)
    return float(a), float(b)


def normal_reference_672(tr672, idx):
    """672 guard distances on the k support images, 3x3 self-exclusion applied.

    Mirrors section 2.2.2's exclusion: without it the reference is ~0 because
    the support images are themselves the bank.
    """
    any_l = LAYERS[0]
    grids = tr672[any_l]["grids"]
    gset = {tuple(int(v) for v in grids[i]) for i in idx}
    assert len(gset) == 1, f"support images have differing grids: {gset}"
    gh, gw = gset.pop()
    rows = np.arange(gh * gw)
    rr, cc = rows // gw, rows % gw
    per_layer = {}
    for l in LAYERS:
        feats = np.concatenate([tr672[l]["feats"][tr672[l]["offsets"][i]:
                                                  tr672[l]["offsets"][i + 1]]
                                for i in idx])
        B = l2n(torch.from_numpy(feats.astype(np.float32)).to(DEV))
        maps = []
        for m in range(len(idx)):
            base = m * gh * gw
            Q = l2n(torch.from_numpy(
                feats[base:base + gh * gw].astype(np.float32)).to(DEV))
            D = (1.0 - Q @ B.T).cpu().numpy()
            allowed = np.ones_like(D, dtype=bool)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r2, c2 = rr + dr, cc + dc
                    ok = (r2 >= 0) & (r2 < gh) & (c2 >= 0) & (c2 < gw)
                    allowed[np.where(ok)[0], base + (r2[ok] * gw + c2[ok])] = False
            if not allowed.any(axis=1).all():
                allowed[:] = True
                allowed[rows, base + rows] = False
            maps.append(np.where(allowed, D, np.inf).min(axis=1))
        per_layer[l] = maps
    Z = []
    for l in LAYERS:
        a = np.concatenate(per_layer[l])
        mu, sd = a.mean(), a.std() + 1e-12
        Z.append([(m - mu) / sd for m in per_layer[l]])
    return [sum(zs) / len(LAYERS) for zs in zip(*Z)], (gh, gw)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_writeback():
    """E: synthetic map -> position, no transpose, no flattening, overlap = mean."""
    g = (6, 7)
    m0 = np.zeros(g[0] * g[1], dtype=np.float32)
    patches, refined = [], []
    patches.append(np.array([10, 11, 17, 18]))       # (row1,col3..4)
    refined.append(np.array([1.0, 1.0, 1.0, 1.0]))
    out, ncov = writeback(m0, g, patches, refined, 1.0, 0.0)
    o = out.reshape(g)
    assert ncov == 4 and abs(o[1, 3] - 1.0) < 1e-12 and o[1, 4] == 1.0
    assert o.sum() == 4.0, f"writeback touched {o.sum()} patches, expected 4"
    assert out.shape == m0.shape, "writeback changed the array shape"

    # overlap -> MEAN, and the affine calibration is applied
    patches = [np.array([10]), np.array([10])]
    refined = [np.array([0.0]), np.array([2.0])]
    out, _ = writeback(m0, g, patches, refined, 2.0, 1.0)
    assert abs(out[10] - (2.0 * 1.0 + 1.0)) < 1e-12, \
        f"overlap/affine wrong: {out[10]}"
    # a transposed write would have put the value at (3, 1) instead of (1, 3)
    o2 = out.reshape(g)
    assert o2[1, 3] == 3.0 and o2[3, 1] == 0.0, "writeback transposed rows/cols"
    print("  E writeback: position OK, shape preserved, overlap = mean, "
          "affine applied, no transpose")


def check_bank_resolution():
    """D: the 672 bank must be 672, and a 448 bank must be refused."""
    tr = load_672("bottle", "train")
    shape = tr["final"]["feats"].shape[-1]
    print(f"  D bank: 672 train cache loaded, feature dim {shape}, "
          f"{len(tr['final']['names'])} normal images")
    try:
        resolve(os.path.join(RESULTS, "cache_ml_img"), 672)
        raise AssertionError("a 448 cache was accepted as a 672 bank")
    except AssertionError as e:
        if "not a 672 cache" not in str(e):
            raise
    print("  D bank: a 448 cache is refused -> no silent resolution mixing")


def check_support_only(tr):
    """A: the calibration may touch exactly the k drawn support images."""
    for shot in (1, 4):
        idx = draw_images(tr[LAYERS[0]]["offsets"], shot, 0, "bottle")
        touched = set(int(i) for i in idx)
        assert len(touched) == shot, f"shot={shot} drew {len(touched)} images"
        print(f"  A support-only: shot={shot} touches image ids {sorted(touched)}"
              f" ({len(touched)} of the pool) -- calibration never sees the rest")
    print("  A support-only: OK")


def check_geometry(obj="bottle"):
    """C/D bridge: a real crop's 448<->672 token mapping at corners and edges."""
    _, tr = load_split_448(obj, "train")
    gh672, gw672 = 48, 48
    cases = {"centre": (15, 15), "top-left": (0, 0), "bottom-right": (29, 29),
             "edge": (0, 20)}
    for nm, (r, c) in cases.items():
        for n in (5, 13):
            r0, c0 = window_672(r, c, n, gh672, gw672)
            assert 0 <= r0 and r0 + n <= gh672 and 0 <= c0 and c0 + n <= gw672
            print(f"  geom {nm:<13} 448 win ({r:>2},{c:>2}) n={n:>2} -> "
                  f"672 win ({r0:>2},{c0:>2})..({r0 + n - 1:>2},{c0 + n - 1:>2})")
    print("  geom: all windows stay inside the 672 grid, no clipping")


def load_split_448(obj, which):
    from v3_selector import load_split
    return load_split(obj, which)


N_SEEDS = 10


def seed_of(dataset, obj, shot, split, seed_id, arm="A4"):
    """Deterministic A4 seed.

        seed = SHA256("dataset|object|shot|split|seed_id|arm")[:8] as an int

    Everything that identifies the draw goes in, so two workers computing the
    same cell derive the identical draw whatever order they run in, and no two
    cells share a draw by accident.  Consciously NOT `hash()` (randomised per
    process) and NOT a positional index (depends on how many objects happened
    to precede this one).

    `dataset` and the `arm` namespace are in the key because MVTec and VisA can
    both grow a category of the same name, and because a future second random
    arm must not collide with A4's.
    """
    import hashlib
    key = f"{dataset}|{obj}|{shot}|{split}|{seed_id}|{arm}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def choose(mode, pool, m2, hot, gap, rows, cols, seed_key=None, gt=None):
    """POSITION-ONLY selection, `pool` = this arm's candidate set.

    `m2` is produced by the single frozen allocate() call and handed in, so
    every arm provably gets the same crop count, the same side n and the same
    token budget -- the arms differ only in WHERE.  Sharing this function (and
    the budget) is what makes A3 > A4 mean "the selector picks better places"
    rather than "the arms got different geometry".

    THE POOLS MATTER.  Addendum 1 A1.3 draws A4 from the windows with
    `hot >= tau_hot` -- the LARGER set -- not from the ones that also satisfy
    `gap <= tau_gap`.  `gap` is the selector's actual contribution, so drawing
    A4 from the fully-triggered set would hand it A3's answer: whenever
    m <= K_max the budget covers every candidate and A4 would be forced to
    equal A3 exactly, leaving G3 with no power on most images.  A5 is the
    oracle of that same task and therefore uses the same pool.
    """
    if m2 == 0:
        return np.zeros(0, dtype=int)          # addendum 2 A2.7
    assert len(pool) >= m2, f"{len(pool)} candidates < m2={m2}"
    if mode == "a3":
        order = rank_truncate(hot[pool], gap[pool], rows[pool], cols[pool],
                              len(pool))
        return pool[order][:m2]
    if mode == "a4":
        assert seed_key is not None, "a4 needs a seed key"
        rng = np.random.default_rng(seed_of(*seed_key))
        return rng.choice(pool, size=m2, replace=False)
    if mode == "a5":
        score = np.zeros(len(pool)) if gt is None else np.asarray(gt)[pool]
        order = np.lexsort((cols[pool], rows[pool], -score))
        return pool[order][:m2]
    raise ValueError(mode)


def check_budget_equality(objects, shot=1, splits=1):
    """Section 10-F: per test image, m' / n / T = m'*n^2 must be identical
    across A3, all 10 A4 seeds, and A5 -- and A3's positions must actually
    differ from A4's, otherwise the random arm may be reusing A3's choice."""
    import v3_selector as sel
    for obj in objects:
        _, tr = load_split_448(obj, "train")
        _, te = load_split_448(obj, "test")
        names = [str(x) for x in te[LAYERS[0]]["names"]]
        for sp in range(splits):
            tau_hot, tau_gap, _ = sel.calibrate(obj, shot, sp, tr)
            S = sel.test_maps(obj, tr, te, shot, sp)
            teg = te[LAYERS[0]]["grids"]
            n_chk = n_diff = n_zero = n_room = 0
            ms = []
            for i, m0 in enumerate(S):
                gh, gw = int(teg[i][0]), int(teg[i][1])
                h, g = sel.window_stats(m0.reshape(gh, gw))
                fire = (h >= tau_hot) & (g <= tau_gap)
                hot_only = h >= tau_hot          # A4/A5's larger pool (A1.3)
                m = int(fire.sum())
                rows, cols = np.mgrid[0:h.shape[0], 0:h.shape[1]]
                cand = np.where(fire.ravel())[0]
                pool = np.where(hot_only.ravel())[0]
                grid672 = (int(np.ceil(gh * 1.5)), int(np.ceil(gw * 1.5)))
                B = sel.budget(grid672)
                m2, n = sel.allocate(m, B)
                if m == 0:
                    n_zero += 1
                Ts, pos_a3 = [], None
                for mode in ["a3"] + [f"a4s{k}" for k in range(N_SEEDS)] + ["a5"]:
                    use = cand if mode == "a3" else pool
                    si = int(mode[3:]) if mode.startswith("a4") else 0
                    p = choose("a3" if mode == "a3" else
                               ("a4" if mode.startswith("a4") else "a5"),
                               use, m2, h.ravel(), g.ravel(),
                               rows.ravel(), cols.ravel(),
                               seed_key=("mvtec" if
                                         os.path.isdir(os.path.join(V1, obj))
                                         else "visa", obj, shot, sp, si))
                    if mode == "a3":
                        pos_a3 = p
                    assert len(p) == m2, f"{mode}: {len(p)} != m2={m2}"
                    Ts.append(len(p) * n * n)
                assert len(set(Ts)) == 1, f"T differs across arms: {set(Ts)}"
                n_chk += 1
                # G3 has power on an image only when A4 has somewhere else to
                # go: the hot pool must contain more windows than the budget
                # covers. (m > K_max is NOT the criterion -- it described the
                # buggy pool where A4 drew from the fully-triggered set.)
                if len(pool) > m2:
                    n_room += 1
                ms.append(m)
                if m2 > 0 and not np.array_equal(np.sort(pos_a3), np.sort(
                        choose("a4", pool, m2, h.ravel(), g.ravel(),
                               rows.ravel(), cols.ravel(),
                               seed_key=("mvtec" if
                                         os.path.isdir(os.path.join(V1, obj))
                                         else "visa", obj, shot, sp, 0)))):
                    n_diff += 1
            ms = np.asarray(ms)
            print(f"  F {obj:<12} {shot}-shot s{sp}: {n_chk} images, "
                  f"m'/n/T identical across A3 + {N_SEEDS} A4 seeds + A5; "
                  f"m=0 on {n_zero}; A3!=A4 on {n_diff}/{n_chk - n_zero} "
                  f"refined", flush=True)
            print(f"      G3 POWER: A4 has room to differ on {n_room}/{n_chk} "
                  f"images (hot pool larger than the budget); on the rest A4 "
                  f"is forced to reproduce A3.  m: mean {ms.mean():.1f}, "
                  f"p90 {np.percentile(ms, 90):.0f}, max {ms.max()}", flush=True)


def selftest_seed():
    """A4 seed derivation: same cell -> same draw, any component -> new draw."""
    base = seed_of("mvtec", "bottle", 1, 0, 3)
    assert base == seed_of("mvtec", "bottle", 1, 0, 3), "not deterministic"
    variants = [("visa", "bottle", 1, 0, 3),      # dataset
                ("mvtec", "screw", 1, 0, 3),      # object
                ("mvtec", "bottle", 2, 0, 3),     # shot
                ("mvtec", "bottle", 1, 1, 3),     # split
                ("mvtec", "bottle", 1, 0, 4)]     # seed_id
    for v in variants:
        assert seed_of(*v) != base, f"component ignored: {v}"
    assert seed_of("mvtec", "bottle", 1, 0, 3, "A5") != base, "namespace ignored"
    # the 10 seeds of one cell are all distinct
    assert len({seed_of("mvtec", "bottle", 1, 0, j) for j in range(10)}) == 10
    # and no two cells collide across a full 27 x 4 x 3 x 10 sweep
    seen = set()
    for ds, objs in (("mvtec", ["bottle", "screw"]), ("visa", ["pcb2"])):
        for o in objs:
            for k in (1, 2, 4, 8):
                for sp in range(3):
                    for j in range(10):
                        s = seed_of(ds, o, k, sp, j)
                        assert s not in seen, f"seed collision at {ds}/{o}"
                        seen.add(s)
    print(f"  seed selftest OK.  anchor seed_of('mvtec','bottle',1,0,3) = "
          f"{base}")
    print(f"  (run this twice: the anchor must print the same value, proving "
          f"cross-process determinism)")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-budget", action="store_true")
    ap.add_argument("--selftest-seed", action="store_true")
    ap.add_argument("--objects", default="bottle")
    a = ap.parse_args()
    if a.selftest_seed:
        selftest_seed()
        return
    if a.check_budget:
        print("V3 section 10-F -- per-image budget equality")
        check_budget_equality(a.objects.split(","))
        return
    if not a.check:
        raise SystemExit("use --check or --check-budget")
    print("V3 refine -- implementation checks")
    check_writeback()
    check_bank_resolution()
    _, tr = load_split_448("bottle", "train")
    check_support_only(tr)
    check_geometry()


if __name__ == "__main__":
    main()
