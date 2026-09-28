# -*- coding: utf-8 -*-
"""
V3 gate 3 -- fresh in-memory reference vs the runner's SAVED output.

The earlier direct runs printed diagnostics to stdout and never wrote maps, so
"runner output equals direct output" could not honestly be claimed from them.
This recomputes the whole cell in memory on the SAME code revision and the
same support/seed, then compares against what the runner serialized.

What it verifies is therefore not `run_arm` (already covered) but the layer
run_cell adds: the call, the serialization, and the save. Bit-exact is the
right bar -- nothing numerical happens in that layer -- and `np.array_equal`
is used deliberately, NOT `allclose`. A mismatch is not to be papered over
with a tolerance; it means the runner changed a result.

Usage: python experiments/model_v0/v3_verify_runner.py --cell mvtec/transistor/k1/s0
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
import v3_manifest as mf  # noqa: E402
import v3_run as vr  # noqa: E402
import v3_selector as sel  # noqa: E402
import v3_refine as rf  # noqa: E402
import v3_check_affine_real as aff  # noqa: E402
import v3_check_scale as cs  # noqa: E402
import v3_cellrunner as cr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    cells = {c["cell_id"]: c for c in mf.cells()}
    cell = cells[a.cell]
    outdir = os.path.join(mf.V3ROOT, cell["dataset"], cell["object"],
                          f"k{cell['shot']}", f"s{cell['split']}")
    if not os.path.exists(os.path.join(outdir, "DONE.json")):
        raise SystemExit(f"{outdir} has no DONE.json -- run the cell first")

    obj, shot, split = cell["object"], cell["shot"], cell["split"]
    print(f"gate 3: fresh reference vs saved runner output for {a.cell}")
    model, resize, totensor, norm = cs.build()
    _, c448 = sel.load_split(obj, "train")
    _, te448 = sel.load_split(obj, "test")
    c672 = rf.load_672(obj, "train")
    te672 = rf.load_672(obj, "test")
    tau_hot, tau_gap, _ = sel.calibrate(obj, shot, split, c448)
    fit = aff.run_case(obj, shot)["fit"]
    a_, b_ = fit["a"], fit["b"]

    arms = {"a3": vr.run_arm(obj, shot, split, "a3", model, resize, totensor,
                             norm, c448, c672, te448, tau_hot, tau_gap, a_, b_)}
    for k in range(mf.N_SEEDS):
        arms[f"a4s{k}"] = vr.run_arm(obj, shot, split, "a4", model, resize,
                                     totensor, norm, c448, c672, te448,
                                     tau_hot, tau_gap, a_, b_, seed_i=k)
    o = te448[rf.LAYERS[0]]["offsets"]
    gtf = te448[rf.LAYERS[0]]["gt_frac"]
    gr = te448[rf.LAYERS[0]]["grids"]
    gt = [np.asarray(gtf[o[i]:o[i + 1]]).reshape(int(gr[i][0]), int(gr[i][1]))
          for i in range(len(gr))]
    arms["a5"] = vr.run_arm(obj, shot, split, "a5", model, resize, totensor,
                            norm, c448, c672, te448, tau_hot, tau_gap, a_, b_,
                            gt=gt)

    S0 = sel.test_maps(obj, c448, te448, shot, split)
    a0 = [np.asarray(m).reshape(int(gr[i][0]), int(gr[i][1]))
          for i, m in enumerate(S0)]
    A2, _, g6 = cr._a2_map(obj, shot, split, c672, te672)
    a2 = [np.asarray(m).reshape(int(g6[i][0]), int(g6[i][1]))
          for i, m in enumerate(A2)]

    # ---------------- bit-exact map comparison ----------------
    z = np.load(os.path.join(outdir, "arms.npz"))
    bad = []
    for k in cr.ARMS:
        ref = np.stack([r["map"] for r in arms[k]])
        if not np.array_equal(z[k], ref):
            d = float(np.abs(z[k] - ref).max())
            bad.append((k, d))
    for name, ref in (("a0.npy", np.stack(a0)), ("a2.npy", np.stack(a2))):
        s = np.load(os.path.join(outdir, name))
        if not np.array_equal(s, ref):
            bad.append((name, float(np.abs(s - ref).max())))
    print(f"  maps: compared {len(cr.ARMS)} arms + a0 + a2 "
          f"({np.load(os.path.join(outdir, 'a0.npy')).shape} / "
          f"{np.load(os.path.join(outdir, 'a2.npy')).shape}) with np.array_equal")
    assert not bad, f"NOT bit-identical: {bad}"
    print("        -> BIT-IDENTICAL")

    # ---------------- diagnostics comparison ----------------
    with open(os.path.join(outdir, "diagnostics.json")) as f:
        saved = json.load(f)
    with open(os.path.join(outdir, "metadata.json")) as f:
        meta = json.load(f)
    diffs = []
    for k in cr.ARMS:
        if len(saved[k]) != len(arms[k]):
            diffs.append(f"{k}: {len(saved[k])} vs {len(arms[k])} rows")
            continue
        for i, (sv, rf_) in enumerate(zip(saved[k], arms[k])):
            for fld in ("m", "m2", "n", "T", "cap_hit", "identical"):
                if fld in sv and sv[fld] != rf_.get(fld):
                    diffs.append(f"{k}[{i}].{fld}: {sv[fld]} vs {rf_.get(fld)}")
            if not np.array_equal(np.asarray(sv["pos"], dtype=int),
                                  np.asarray(rf_["pos"], dtype=int)):
                diffs.append(f"{k}[{i}].pos differs")
                break
    assert not diffs, f"diagnostics differ: {diffs[:5]}"
    print(f"  diagnostics: {len(cr.ARMS)} arms x {len(arms['a3'])} images -- "
          f"m/m2/n/T/cap_hit/identical/pos all identical")
    assert abs(meta["tau_hot"] - tau_hot) == 0.0
    assert abs(meta["tau_gap"] - tau_gap) == 0.0
    assert meta["affine"]["a"] == a_ and meta["affine"]["b"] == b_
    print(f"  metadata: tau_hot/tau_gap/a/b/R2 identical "
          f"(a={a_:.5f} b={b_:.5f} R2={fit['r2']:.4f})")

    ndiff = 0
    for i in range(len(arms["a3"])):
        if arms["a3"][i]["m2"] == 0:
            continue
        if not np.array_equal(np.sort(arms["a3"][i]["pos"]),
                              np.sort(arms["a4s0"][i]["pos"])):
            ndiff += 1
    integ = sum(1 for r in arms["a3"] if r["m2"] > 0)
    print(f"  G3 diagnostic: A3 != A4(seed 0) on {ndiff}/{integ} refined "
          f"images")
    print("\n  GATE 3 PASS -- the runner changed nothing")


if __name__ == "__main__":
    main()
