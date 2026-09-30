# -*- coding: utf-8 -*-
"""
TSN formal confirmatory runner V2.
Run from the ORIGINAL project root, e.g. E:\\work\\freshman.

Frozen scientific protocol:
- shots: 2,4,8
- splits: 0,1,2
- TSN on raw final-layer 1NN distances
- q = log(max / mean_top1p)
- q_ref = median leave-one-support-out q
- test score = |q_test - q_ref|
- comparator = original agg_all3 image score
- no defect labels used for calibration/tuning

V2 only adds operational safety:
- repo-root = current working directory
- resume support
- per-cell timing
- smoke mode without formal gate
- expected-cell checks
Scientific formula and gates are unchanged.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0, str(ROOT/"experiments"/"baseline"))
sys.path.insert(0, str(ROOT/"third_party"/"AnomalyDINO"))

from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images, l2n, nn_dist
from tail_calib import mean_top1p, RESULTS

LAYERS=["mid","midlate","final"]
DEV="cuda"

def qshape(d):
    d=np.asarray(d,dtype=np.float64).ravel()
    if d.size == 0:
        raise ValueError("empty distance vector")
    eps=1e-12
    return float(np.log((float(d.max())+eps)/(float(mean_top1p(d))+eps)))

def make_bank(feats, offs, ids):
    x=np.concatenate([feats[int(offs[i]):int(offs[i+1])] for i in ids]).astype(np.float32)
    return l2n(torch.from_numpy(x).to(DEV))

def image_slice(cache,i):
    o=cache["offsets"]
    return cache["feats"][int(o[i]):int(o[i+1])]

def score_image(feats_i,bank):
    q=l2n(torch.from_numpy(np.asarray(feats_i,np.float32)).to(DEV))
    with torch.inference_mode():
        return nn_dist(q,bank).cpu().numpy().astype(np.float64)

def infer_objects():
    mv=Path(RESULTS)/"cache_ml_img"/"final"
    vi=Path(RESULTS)/"cache_visa"/"final"
    objs=set()
    for p in mv.glob("*_train.npz"):
        objs.add(p.name[:-len("_train.npz")])
    for p in vi.glob("*_train.npz"):
        objs.add(p.name[:-len("_train.npz")])
    return sorted(objs)

def dataset_of(obj):
    from phase_m1_rescue import V1, VISA
    if os.path.isdir(os.path.join(V1,obj)):
        return "mvtec"
    if os.path.isdir(os.path.join(VISA,obj)):
        return "visa"
    return "unknown"

def run_cell(obj,shot,split):
    t0=time.perf_counter()
    if shot < 2:
        raise ValueError("Formal TSN requires shot >= 2.")
    root,layers,tr,te=load_obj("r0",obj)
    if list(layers)!=LAYERS:
        raise RuntimeError(f"{obj}: layers={layers}, expected={LAYERS}")

    support_ids=[int(v) for v in draw_images(tr["final"]["offsets"],shot,split,obj)]
    if len(support_ids)!=shot:
        raise RuntimeError(f"{obj} k{shot} s{split}: support count {len(support_ids)} != {shot}")

    support_q=[]
    for sid in support_ids:
        loo=[j for j in support_ids if j!=sid]
        bank_loo=make_bank(tr["final"]["feats"],tr["final"]["offsets"],loo)
        d=score_image(image_slice(tr["final"],sid),bank_loo)
        support_q.append(qshape(d))
        del bank_loo
    q_ref=float(np.median(support_q))

    banks={}
    raw={}
    for layer in LAYERS:
        banks[layer]=make_bank(tr[layer]["feats"],tr[layer]["offsets"],support_ids)
        raw[layer]=[]
        n_img=len(te[layer]["offsets"])-1
        for i in range(n_img):
            raw[layer].append(score_image(image_slice(te[layer],i),banks[layer]))
        del banks[layer]

    Z={}
    for layer in LAYERS:
        allv=np.concatenate(raw[layer])
        mu=float(allv.mean())
        sd=float(allv.std()+1e-12)
        Z[layer]=[(d-mu)/sd for d in raw[layer]]
    agg=[(a+b+c)/3.0 for a,b,c in zip(Z["mid"],Z["midlate"],Z["final"])]

    types=[str(v) for v in te["final"]["types"]]
    y=np.array([1 if t=="bad" else 0 for t in types],dtype=np.int32)
    if y.min()==y.max():
        raise RuntimeError(f"{obj}: test labels have only one class")

    base_final=np.array([mean_top1p(d) for d in raw["final"]],dtype=np.float64)
    base_agg=np.array([mean_top1p(d) for d in agg],dtype=np.float64)
    tsn=np.array([abs(qshape(d)-q_ref) for d in raw["final"]],dtype=np.float64)

    auc_final=float(roc_auc_score(y,base_final)*100)
    auc_agg=float(roc_auc_score(y,base_agg)*100)
    auc_tsn=float(roc_auc_score(y,tsn)*100)

    return {
        "dataset":dataset_of(obj),
        "object":obj,"shot":shot,"split":split,
        "n_support":len(support_ids),
        "support_ids":";".join(map(str,support_ids)),
        "q_ref":q_ref,
        "q_support":";".join(f"{v:.12g}" for v in support_q),
        "baseline_final_img_AUROC":auc_final,
        "baseline_agg_img_AUROC":auc_agg,
        "tsn_img_AUROC":auc_tsn,
        "delta_tsn_vs_agg":auc_tsn-auc_agg,
        "delta_tsn_vs_final":auc_tsn-auc_final,
        "elapsed_sec":time.perf_counter()-t0,
    }

def make_gate(df):
    rows=[]
    for dataset,g in df.groupby("dataset"):
        delta=g["delta_tsn_vs_agg"]
        hard_thr=g["baseline_agg_img_AUROC"].quantile(.25)
        hard=g[g["baseline_agg_img_AUROC"]<=hard_thr]
        row={
            "dataset":dataset,
            "n_cells":len(g),
            "mean_baseline_agg":g["baseline_agg_img_AUROC"].mean(),
            "mean_tsn":g["tsn_img_AUROC"].mean(),
            "mean_delta":delta.mean(),
            "median_delta":delta.median(),
            "nonnegative_fraction":float((delta>=0).mean()),
            "hard_quartile_threshold":hard_thr,
            "hard_quartile_n":len(hard),
            "hard_quartile_mean_delta":hard["delta_tsn_vs_agg"].mean(),
            "G1_mean_delta_gt0":bool(delta.mean()>0),
            "G2_nonnegative_ge70pct":bool((delta>=0).mean()>=0.70),
            "G3_hard_quartile_gain_ge2":bool(hard["delta_tsn_vs_agg"].mean()>=2.0),
            "G4_no_mean_degradation":bool(g["tsn_img_AUROC"].mean()>=g["baseline_agg_img_AUROC"].mean()),
        }
        row["ALL_PASS"]=all(v for k,v in row.items() if k.startswith("G"))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--objects",default="")
    ap.add_argument("--shots",default="2,4,8")
    ap.add_argument("--splits",type=int,default=3)
    ap.add_argument("--out",default=r"results\model_v0\metrics\tsn_formal_v2.csv")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--smoke",action="store_true",help="Run a small subset and skip formal gate.")
    ap.add_argument("--max-cells",type=int,default=0)
    args=ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    print("ROOT =",ROOT)
    print("GPU  =",torch.cuda.get_device_name(0))
    print("CUDA =",torch.version.cuda)

    objs=[o for o in args.objects.split(",") if o] if args.objects else infer_objects()
    shots=[int(v) for v in args.shots.split(",")]
    keys=[(o,k,s) for o in objs for k in shots for s in range(args.splits)]
    if args.max_cells>0:
        keys=keys[:args.max_cells]

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    done=set()
    if args.resume and out.exists():
        old=pd.read_csv(out)
        rows=old.to_dict("records")
        done={(str(r["object"]),int(r["shot"]),int(r["split"])) for _,r in old.iterrows()}
        print(f"Resume: {len(done)} cells already complete.")

    total=len(keys)
    for n,(obj,shot,split) in enumerate(keys,1):
        key=(obj,shot,split)
        if key in done:
            print(f"[{n}/{total}] SKIP {obj} k{shot} s{split}")
            continue
        print(f"[{n}/{total}] RUN  {obj} k{shot} s{split}",flush=True)
        row=run_cell(obj,shot,split)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out,index=False)
        print(f"  agg={row['baseline_agg_img_AUROC']:.3f} TSN={row['tsn_img_AUROC']:.3f} "
              f"delta={row['delta_tsn_vs_agg']:+.3f} time={row['elapsed_sec']:.1f}s",flush=True)

    df=pd.DataFrame(rows)
    if args.smoke or args.max_cells>0 or args.objects:
        print("SMOKE/SUBSET COMPLETE. Formal gate intentionally not evaluated.")
        return

    expected=len(objs)*len(shots)*args.splits
    if len(df)!=expected:
        raise SystemExit(f"Expected {expected} unique cells, got {len(df)}.")
    if df[["object","shot","split"]].duplicated().any():
        raise SystemExit("Duplicate cell keys found.")

    gate=make_gate(df)
    gate_path=out.with_name(out.stem+"_gate.csv")
    gate.to_csv(gate_path,index=False)
    print("\n=== FROZEN TSN FORMAL GATE ===")
    print(gate.to_string(index=False))
    if set(gate.dataset)!={"mvtec","visa"}:
        raise SystemExit(f"Expected mvtec+visa, found {sorted(gate.dataset.unique())}")
    if not bool(gate.ALL_PASS.all()):
        raise SystemExit("TSN FORMAL GATE: NOT PASS")
    print("TSN FORMAL GATE: PASS")

if __name__=="__main__":
    main()
