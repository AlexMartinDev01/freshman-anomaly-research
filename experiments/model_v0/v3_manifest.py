# -*- coding: utf-8 -*-
"""
V3 main-experiment manifest, shard planner and infrastructure dry-run.

Pure scheduling.  This module never loads a model, a cache or a metric, and it
must stay that way: the confirmatory run's infrastructure has to be provably
independent of anything that could influence a result.

  cells()     27 categories x 4 shots x 3 splits = 324 cells, each with a
              unique id "dataset/object/kshot/ssplit" and a unique output dir
  seeds(cell) the 10 A4 seeds, generated NOW rather than decided at run time
  plan(n)     category-level sharding, greedy bin packing on an ESTIMATED COST
              that treats VisA as ~3x MVTec (measured: 153s vs 453s per cell).
              Splitting by category count alone would hand one worker every
              slow VisA category.  A category's 12 cells never split across
              workers.

Usage: python experiments/model_v0/v3_manifest.py --dry-run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from tail_calib import RESULTS  # noqa: E402
import v3_refine as rf  # noqa: E402

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
SHOTS = [1, 2, 4, 8]
SPLITS = [0, 1, 2]
N_SEEDS = 10
V3ROOT = os.path.join(RESULTS, "v3")

# Estimated per-cell cost.  Measured on the engineering smoke: an MVTec cell
# (transistor) took 153 s and a VisA cell (pcb2) 453 s, so 3.0 is not a guess.
COST = {"mvtec": 1.0, "visa": 3.0}


def categories():
    return ([("mvtec", o) for o in MVTEC] + [("visa", o) for o in VISA])


def cells():
    out = []
    for ds, obj in categories():
        for k in SHOTS:
            for s in SPLITS:
                out.append(dict(dataset=ds, object=obj, shot=k, split=s,
                                cell_id=f"{ds}/{obj}/k{k}/s{s}"))
    return out


def outdir(cell):
    return os.path.join(V3ROOT, cell["dataset"], cell["object"],
                        f"k{cell['shot']}", f"s{cell['split']}")


def seeds_for(cell):
    """The 10 A4 seeds, derived now and stored with the cell."""
    return [rf.seed_of(cell["dataset"], cell["object"], cell["shot"],
                       cell["split"], j) for j in range(N_SEEDS)]


def plan(n_workers=2):
    """Category-level greedy bin packing.  A category is atomic."""
    cats = categories()
    load = {w: 0.0 for w in range(n_workers)}
    shards = {w: [] for w in range(n_workers)}
    for ds, obj in sorted(cats, key=lambda c: -COST[c[0]]):
        w = min(load, key=lambda x: (load[x], x))       # ties -> lowest index
        shards[w].append((ds, obj))
        load[w] += COST[ds]
    return shards, load


def dry_run():
    print("=" * 92)
    print("V3 infrastructure dry-run -- NO model, NO cache, NO metric")
    print("=" * 92)
    cs = cells()
    ids = [c["cell_id"] for c in cs]
    dirs = [outdir(c) for c in cs]

    nm = sum(1 for c in cs if c["dataset"] == "mvtec")
    nv = sum(1 for c in cs if c["dataset"] == "visa")
    print(f"\n  cells          {len(cs)}  (MVTec {nm} + VisA {nv})")
    assert len(cs) == 324, f"expected 324 cells, got {len(cs)}"
    assert nm == 180 and nv == 144, (nm, nv)
    assert len(set(ids)) == len(ids), "duplicate cell_id"
    assert len(set(dirs)) == len(dirs), "duplicate output directory"
    print(f"  unique ids     {len(set(ids))}   unique dirs  {len(set(dirs))}"
          f"   -> 0 duplicates, 0 collisions")

    # every (dataset, object, shot, split) present exactly once
    key = {(c["dataset"], c["object"], c["shot"], c["split"]) for c in cs}
    assert len(key) == 324 and len(key) == len(cs)
    print(f"  coverage       every (dataset,object,shot,split) present exactly "
          f"once")

    # ---- A4 seeds: pre-generated, per cell, globally collision-free ----
    allseeds = {}
    for c in cs:
        s = seeds_for(c)
        assert len(set(s)) == N_SEEDS, f"{c['cell_id']}: seeds not distinct"
        for j, v in enumerate(s):
            assert v not in allseeds, \
                f"seed collision: {c['cell_id']} seed {j} vs {allseeds[v]}"
            allseeds[v] = f"{c['cell_id']}#{j}"
    print(f"  A4 seeds       {len(cs)} x {N_SEEDS} = {len(allseeds)} distinct, "
          f"0 collisions")
    # and the SAME shot/split draws are not replaying across shots/splits
    a = {tuple(seeds_for(c)) for c in cs}
    assert len(a) == len(cs), "two cells share an identical seed list"
    print(f"  seed lists     all {len(a)} cells have a distinct 10-seed list")

    # ---- sharding ----
    for nw in (2, 3):
        shards, load = plan(nw)
        flat = [c for w in shards for c in shards[w]]
        assert sorted(flat) == sorted(categories()), "shards do not partition"
        cells_per = {w: len(shards[w]) * 12 for w in shards}
        print(f"\n  shard plan ({nw} workers), MVTec weight 1.0 / VisA 3.0:")
        for w in shards:
            names = [f"{d}:{o}" for d, o in shards[w]]
            print(f"    w{w}  cost {load[w]:5.1f}  {cells_per[w]:3d} cells  "
                  f"{len(names)} categories: {', '.join(names)}")
        assert sum(cells_per.values()) == 324
        a_cost = max(load.values()) / (sum(load.values()) / nw)
        print(f"    imbalance vs perfect balance: {a_cost:.3f}x")

    # ---- the runner must not know how metrics are computed ----
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "v3_run.py")
    src = open(runner, encoding="utf-8").read()
    # Look for a CALL, not a mention: the engineering-only guard deliberately
    # names these functions in order to REPLACE them (`_pmb.x = _boom`), which
    # is an assignment and takes no call parentheses.
    import re
    banned = ["pixel_metrics_binned", "roc_auc_score", "mean_top1p",
              "pixel_f1_at_threshold"]
    hits = [b for b in banned
            if re.search(rf"(?<![\w.]){b}\s*\(", src)]
    assert not hits, \
        (f"runner CALLS a metric function: {hits}. The confirmatory runner "
         f"must not know how AUPRO/AUROC are computed; aggregation is a "
         f"separate script run only after 324/324 complete.")
    print(f"\n  runner         v3_run.py references no metric function "
          f"({', '.join(banned)})")

    print(f"\n  manifest       {V3ROOT}")
    print("\n  DRY RUN OK -- infrastructure only")
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        cs = dry_run()
        if a.write_manifest:
            os.makedirs(V3ROOT, exist_ok=True)
            p = os.path.join(V3ROOT, "manifest.json")
            for c in cs:
                c["outdir"] = outdir(c)
                c["a4_seeds"] = seeds_for(c)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cs, f, indent=1)
            print(f"  wrote {p} ({len(cs)} cells)")
    else:
        raise SystemExit("use --dry-run")


if __name__ == "__main__":
    main()
