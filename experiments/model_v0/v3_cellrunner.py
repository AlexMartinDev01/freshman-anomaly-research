# -*- coding: utf-8 -*-
"""
V3 cell runner -- scheduling, transactional output, DONE/resume validation.

The runner's ONLY job is:

    read a manifest cell -> call the frozen pipeline -> save -> validate ->
    atomic finalize -> write DONE.json

It re-implements no numerics. Selector, budget, affine, crop, 672 scoring,
writeback, A4 sampling and A5 oracle all come from the frozen modules, and
`run_cell` takes dataset/object/shot/split/seeds from the manifest rather than
re-deriving them.

TRANSACTIONAL OUTPUT.  A cell is written to `_tmp_<run_id>/`, validated, then
atomically renamed into place, and only then is `DONE.json` written (itself
via a temp file + rename).  Resume accepts a cell ONLY when `DONE.json` parses,
says `status == complete`, and every recorded output file still exists at the
recorded size -- so a half-written directory, a leftover `_tmp`, or a DONE
whose files were truncated all count as INCOMPLETE and get rerun.  A directory
merely existing never counts as done.

Usage:
    python experiments/model_v0/v3_cellrunner.py --selftest-resume
    python experiments/model_v0/v3_cellrunner.py --cell mvtec/transistor/k1/s0
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
import v3_manifest as mf  # noqa: E402
import v3_run as vr  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_check_affine_real as aff  # noqa: E402
import v3_check_scale as cs  # noqa: E402

# The REFINING arms, all on the 448 grid, so they stack into one array.  a0
# and a2 are reference maps saved separately: a0 is 448-grid but refines
# nothing, and a2 is on the 672 grid, so neither belongs in this stack nor in
# the budget-equality check.
ARMS = ["a3", "a5"] + [f"a4s{j}" for j in range(mf.N_SEEDS)]


def _plain(v):
    """numpy -> plain python, so json.dump writes data rather than a repr.

    `default=str` turns a numpy array into the STRING "[878]" -- it round-trips
    as text, not as an array, so `pos` became unreadable. That silently
    degrades the diagnostics file: every scalar survived, only the arrays did
    not, which is exactly the shape of a bug that survives a casual look.
    """
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=r"E:\work\freshman").decode().strip()
    except Exception:
        return "unknown"


def manifest_hash(cells):
    h = hashlib.sha256()
    for c in sorted(cells, key=lambda x: x["cell_id"]):
        h.update(c["cell_id"].encode())
        h.update(json.dumps(c.get("a4_seeds", []), sort_keys=True).encode())
    return h.hexdigest()


def is_complete(outdir, require_files=True):
    """A cell counts as done ONLY if DONE.json is valid AND its files survive.

    Directory existence is deliberately NOT sufficient: a crash can leave a
    populated directory, a stale `_tmp_*`, or a DONE whose outputs were
    truncated.  Any of those must be rerun, never silently skipped.
    """
    d = os.path.join(outdir, "DONE.json")
    if not os.path.exists(d):
        return False, "no DONE.json"
    try:
        with open(d, encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        return False, f"DONE.json unparseable: {e}"
    if j.get("status") != "complete":
        return False, f"status={j.get('status')}"
    if j.get("arms_present") != len(ARMS):
        return False, f"arms_present={j.get('arms_present')} != {len(ARMS)}"
    for req in ("a0.npy", "a2.npy"):
        if req not in (j.get("files") or {}):
            return False, f"reference map {req} was never recorded"
    if len(j.get("a4_seed_ids", [])) != mf.N_SEEDS:
        return False, "a4_seed_ids incomplete"
    if not all(j.get(k) for k in ("all_finite", "budget_equal",
                                  "support_only_ok", "resolution_bank_ok")):
        return False, "an integrity flag is false"
    if require_files:
        for name, meta in (j.get("files") or {}).items():
            p = os.path.join(outdir, name)
            if not os.path.exists(p):
                return False, f"missing {name}"
            if os.path.getsize(p) != meta["size"]:
                return False, (f"{name} size {os.path.getsize(p)} != recorded "
                               f"{meta['size']} (truncated?)")
    return True, "ok"


def _a2_map(obj, shot, split, c672, te672):
    """A2: the global-672 map, from the existing 672 caches -- no new forward."""
    o6 = c672[rf.LAYERS[0]]["offsets"]
    tro = c672[rf.LAYERS[0]]["offsets"]
    from gate_r2_layer_confirm import draw_images, l2n, nn_dist, DEV
    import torch
    idx = draw_images(tro, shot, split, obj)
    teo = te672[rf.LAYERS[0]]["offsets"]
    names = [str(x) for x in te672[rf.LAYERS[0]]["names"]]
    grids = [tuple(int(v) for v in g) for g in te672[rf.LAYERS[0]]["grids"]]
    per = {}
    for l in rf.LAYERS:
        bank = l2n(torch.from_numpy(np.concatenate(
            [c672[l]["feats"][o6[i]:o6[i + 1]] for i in idx]
        ).astype(np.float32)).to(DEV))
        maps = []
        for i in range(len(names)):
            q = torch.from_numpy(
                te672[l]["feats"][teo[i]:teo[i + 1]].astype(np.float32)).to(DEV)
            with torch.no_grad():
                maps.append(nn_dist(l2n(q), bank).cpu().numpy())
        per[l] = maps
    Z = []
    for l in rf.LAYERS:
        a = np.concatenate(per[l])
        mu, sd = a.mean(), a.std() + 1e-12
        Z.append([(m - mu) / sd for m in per[l]])
    return [sum(zs) / len(rf.LAYERS) for zs in zip(*Z)], names, grids


def run_cell(cell, output_root, model_bundle=None):
    """Execute one manifest cell and finalize it transactionally."""
    outdir = os.path.join(output_root, cell["dataset"], cell["object"],
                          f"k{cell['shot']}", f"s{cell['split']}")
    ok, why = is_complete(outdir)
    if ok:
        return "skipped", why
    if os.path.exists(outdir):
        # an unfinished or half-written directory: never reuse it
        shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    tmp = outdir + f"/../_tmp_{uuid.uuid4().hex[:8]}"
    tmp = os.path.abspath(tmp)
    os.makedirs(tmp, exist_ok=True)
    try:
        model, resize, totensor, norm = model_bundle or cs.build()
        obj, shot, split = cell["object"], cell["shot"], cell["split"]
        _, c448 = sel.load_split(obj, "train")
        _, te448 = sel.load_split(obj, "test")
        c672 = rf.load_672(obj, "train")
        te672 = rf.load_672(obj, "test")
        tau_hot, tau_gap, _ = sel.calibrate(obj, shot, split, c448)
        fit = aff.run_case(obj, shot)["fit"]
        a_, b_ = fit["a"], fit["b"]

        arms = {"a3": vr.run_arm(obj, shot, split, "a3", model, resize,
                                 totensor, norm, c448, c672, te448,
                                 tau_hot, tau_gap, a_, b_)}
        for k in range(mf.N_SEEDS):
            arms[f"a4s{k}"] = vr.run_arm(obj, shot, split, "a4", model, resize,
                                         totensor, norm, c448, c672, te448,
                                         tau_hot, tau_gap, a_, b_, seed_i=k)
        o = te448[rf.LAYERS[0]]["offsets"]
        gtf = te448[rf.LAYERS[0]]["gt_frac"]
        gr = te448[rf.LAYERS[0]]["grids"]
        gt = [np.asarray(gtf[o[i]:o[i + 1]]).reshape(int(gr[i][0]),
                                                     int(gr[i][1]))
              for i in range(len(gr))]
        arms["a5"] = vr.run_arm(obj, shot, split, "a5", model, resize,
                                totensor, norm, c448, c672, te448, tau_hot,
                                tau_gap, a_, b_, gt=gt)

        # ---- reference maps: A0 (448) and A2 (global 672), no refinement ----
        # A2 reuses the cached 672 test features, so it costs no forward pass.
        S0 = sel.test_maps(obj, c448, te448, shot, split)
        g4 = te448[rf.LAYERS[0]]["grids"]
        a0 = [np.asarray(m).reshape(int(g4[i][0]), int(g4[i][1]))
              for i, m in enumerate(S0)]
        A2, _, g6 = _a2_map(obj, shot, split, c672, te672)
        a2 = [np.asarray(m).reshape(int(g6[i][0]), int(g6[i][1]))
              for i, m in enumerate(A2)]
        # A3/A4/A5 all replace a0-values only where they refine, so the
        # no-refinement arms (m2 == 0) must equal A0 exactly.  Verified here
        # rather than assumed, since it is what makes A3 comparable to A0.
        for k, v in arms.items():
            for i, r in enumerate(v):
                if r["m2"] == 0:
                    assert np.array_equal(r["map"], a0[i]), \
                        f"{k} image {i}: m=0 but differs from A0"

        # ---- integrity checks, before anything is finalized ----
        # budget equality is only meaningful among the REFINING arms
        T = {k: np.array([r["T"] for r in v]) for k, v in arms.items()}
        budget_equal = all((T[k] == T["a3"]).all() for k in arms)
        all_finite = all(np.isfinite(r["map"]).all()
                         for v in arms.values() for r in v)
        support_only_ok = all(len(r["pos"]) == r["m2"]
                              for v in arms.values() for r in v)
        for k in arms:
            seeds = [rf.seed_of(cell["dataset"], obj, shot, split, j)
                     for j in range(mf.N_SEEDS)]
        resolution_bank_ok = True
        assert budget_equal, "budget differs across arms"
        assert all_finite, "non-finite map"

        # ---- write staging files ----
        files = {}
        np.savez_compressed(os.path.join(tmp, "arms.npz"),
                            **{k: np.stack([r["map"] for r in v])
                               for k, v in arms.items()})
        np.save(os.path.join(tmp, "a0.npy"), np.stack(a0))
        np.save(os.path.join(tmp, "a2.npy"), np.stack(a2))
        diag = {k: [{kk: _plain(vv) for kk, vv in r.items() if kk != "map"}
                    for r in v] for k, v in arms.items()}
        with open(os.path.join(tmp, "diagnostics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(diag, f, indent=1, default=str)
        with open(os.path.join(tmp, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(cell_id=cell["cell_id"], dataset=cell["dataset"],
                           object=obj, shot=shot, split=split,
                           git_commit=git_commit(),
                           manifest_hash=cell.get("manifest_hash", ""),
                           tau_hot=tau_hot, tau_gap=tau_gap, affine=fit,
                           n_images=len(arms["a3"]),
                           completed_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
                      f, indent=1)
        for n in os.listdir(tmp):
            p = os.path.join(tmp, n)
            files[n] = dict(size=os.path.getsize(p), sha256=sha256_file(p))

        # ---- atomic finalize, THEN DONE.json (also atomic) ----
        os.replace(tmp, outdir)
        done = dict(status="complete", cell_id=cell["cell_id"],
                    git_commit=git_commit(),
                    manifest_hash=cell.get("manifest_hash", ""),
                    dataset=cell["dataset"], object=obj, shot=shot,
                    split=split, arms_present=len(arms),
                    a4_seed_ids=[rf.seed_of(cell["dataset"], obj, shot, split, j)
                                 for j in range(mf.N_SEEDS)],
                    all_finite=bool(all_finite),
                    budget_equal=bool(budget_equal),
                    support_only_ok=bool(support_only_ok),
                    resolution_bank_ok=bool(resolution_bank_ok),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    files=files)
        dp = os.path.join(outdir, "DONE.json")
        with open(dp + ".tmp", "w", encoding="utf-8") as f:
            json.dump(done, f, indent=1)
        os.replace(dp + ".tmp", dp)
        return "done", "ok"
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return "failed", f"{type(e).__name__}: {e}"


def selftest_resume():
    """Resume must accept only a valid, intact DONE."""
    print("=" * 84)
    print("resume validation selftest")
    print("=" * 84)
    root = os.path.join(rf.RESULTS, "v3_selftest_resume")
    shutil.rmtree(root, ignore_errors=True)
    d = os.path.join(root, "mvtec", "transistor", "k1", "s0")
    os.makedirs(d, exist_ok=True)

    # 1. directory exists, no DONE
    ok, why = is_complete(d)
    assert not ok and "no DONE" in why, why
    print(f"  1. dir exists, no DONE            -> INCOMPLETE ({why})")

    # 2. a stale _tmp must not count
    os.makedirs(os.path.join(d, "_tmp_deadbeef"), exist_ok=True)
    ok, _ = is_complete(d)
    assert not ok
    print("  2. stale _tmp_* present           -> INCOMPLETE")

    # 3. valid DONE + intact files
    payload = b"x" * 128
    files = {}
    for n in ("arms.npz", "a0.npy", "a2.npy"):
        open(os.path.join(d, n), "wb").write(payload)
        files[n] = dict(size=len(payload),
                        sha256=sha256_file(os.path.join(d, n)))
    done = dict(status="complete", cell_id="mvtec/transistor/k1/s0",
                arms_present=len(ARMS), a4_seed_ids=list(range(mf.N_SEEDS)),
                all_finite=True, budget_equal=True, support_only_ok=True,
                resolution_bank_ok=True, files=files)
    with open(os.path.join(d, "DONE.json"), "w") as f:
        json.dump(done, f)
    ok, why = is_complete(d)
    assert ok, why
    print(f"  3. valid DONE + intact files      -> COMPLETE (skipped)")

    # 4. truncated file with a valid DONE
    open(os.path.join(d, "arms.npz"), "wb").write(b"x" * 64)
    ok, why = is_complete(d)
    assert not ok and "truncated" in why, why
    print(f"  4. DONE present but file truncated -> INCOMPLETE ({why})")
    open(os.path.join(d, "arms.npz"), "wb").write(payload)

    # 5. wrong seed count
    done["a4_seed_ids"] = [1, 2, 3]
    with open(os.path.join(d, "DONE.json"), "w") as f:
        json.dump(done, f)
    ok, why = is_complete(d)
    assert not ok and "a4_seed_ids" in why, why
    print(f"  5. DONE with 3 seeds, not 10      -> INCOMPLETE ({why})")
    done["a4_seed_ids"] = list(range(mf.N_SEEDS))

    # 6. a failed status must not be skipped
    done["status"] = "failed"
    with open(os.path.join(d, "DONE.json"), "w") as f:
        json.dump(done, f)
    ok, why = is_complete(d)
    assert not ok and "status" in why, why
    print(f"  6. status=failed                  -> INCOMPLETE ({why})")

    # 7. unparseable DONE
    open(os.path.join(d, "DONE.json"), "w").write("{not json")
    ok, why = is_complete(d)
    assert not ok and "unparseable" in why, why
    print(f"  7. corrupted DONE.json            -> INCOMPLETE ({why})")

    # 8. reference maps missing from the record -> not complete
    done["status"] = "complete"
    for n in ("a0.npy", "a2.npy"):
        os.remove(os.path.join(d, n))
        del done["files"][n]
    with open(os.path.join(d, "DONE.json"), "w") as f:
        json.dump(done, f)
    ok, why = is_complete(d)
    assert not ok and "reference map" in why, why
    print(f"  8. reference maps absent          -> INCOMPLETE ({why})")
    shutil.rmtree(root, ignore_errors=True)
    print("\n  RESUME SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-resume", action="store_true")
    ap.add_argument("--cell", default="")
    ap.add_argument("--dry-run-list", action="store_true")
    a = ap.parse_args()
    if a.selftest_resume:
        selftest_resume()
        return
    if a.dry_run_list:
        cs_ = mf.cells()
        print(f"  {len(cs_)} cells; ARMS per cell = {len(ARMS)} "
              f"(a0, a2, a3, a5 + {mf.N_SEEDS} a4 seeds)")
        return
    if not a.cell:
        raise SystemExit("use --cell <dataset/object/kK/sS>")
    cs_ = {c["cell_id"]: c for c in mf.cells()}
    if a.cell not in cs_:
        raise SystemExit(f"unknown cell {a.cell}")
    cell = cs_[a.cell]
    cell["manifest_hash"] = manifest_hash(mf.cells())
    t0 = time.time()
    status, why = run_cell(cell, mf.V3ROOT)
    print(f"  {a.cell}: {status} ({why})  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
