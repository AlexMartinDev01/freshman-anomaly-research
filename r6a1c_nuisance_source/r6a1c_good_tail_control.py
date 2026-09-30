# -*- coding: utf-8 -*-
"""R6-A1C audit 4, the missing control.

N_H is the top 1% of a BAD image's distance map, so asking whether it exceeds a
distance threshold is partly circular -- the natural control is the top 1% of a
GOOD image's distance map.  If good-image tails exceed the normal-only threshold
just as often, then N_H is not special and the "detectable from normal data"
claim collapses.

Bank size is matched 3-vs-3 exactly as in the corrected audit.
"""
import sys, zlib
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"experiments"/"model_v0"))
sys.path.insert(0, str(ROOT/"experiments"/"baseline")); sys.path.insert(0, str(ROOT/"third_party"/"AnomalyDINO"))
from phase_m1_rescue import load_obj
from gate_r2_layer_confirm import draw_images, l2n
DEV="cuda"; GT_THR=.10; FAR=4; ALPHA=.01; CHUNK=512
sel = pd.read_csv(ROOT/"r6a_supervised_smoke"/"R6A_SMOKE_SELECTION.csv")
out = ROOT/"results"/"model_v0"/"metrics"/"r6a1c_nuisance_source"
def nnd(q,b):
    o=np.empty(len(q))
    for s in range(0,len(q),CHUNK):
        with torch.inference_mode():
            o[s:s+CHUNK]=(1.0-q[s:s+CHUNK]@b.T).max(1).values.cpu().numpy().astype(np.float64)
    return o
def cheb(a,b,C):
    if not len(a) or not len(b): return np.full(len(a),np.inf)
    ar,ac=np.divmod(a,C); br,bc=np.divmod(b,C)
    return np.maximum(np.abs(ar[:,None]-br[None,:]),np.abs(ac[:,None]-bc[None,:])).min(axis=1)
rows=[]
for _,r in sel.iterrows():
    obj=str(r.object); _,_,tr,te=load_obj("r0",obj); tec=te["final"]
    ids=[int(v) for v in draw_images(tr["final"]["offsets"],4,0,obj)]
    o=np.asarray(tec["offsets"]); ot=np.asarray(tr["final"]["offsets"])
    types=np.asarray([str(v) for v in tec["types"]]); C=int(tec["grids"][0][1])
    sub={d: l2n(torch.from_numpy(np.concatenate([tr["final"]["feats"][int(ot[j]):int(ot[j+1])] for j in ids if j!=ids[d]]).astype(np.float32)).to(DEV)) for d in range(len(ids))}
    null=[]
    for si,sid in enumerate(ids):
        q=l2n(torch.from_numpy(tr["final"]["feats"][int(ot[sid]):int(ot[sid+1])].astype(np.float32)).to(DEV))
        null.append(nnd(q,sub[si])); del q
    thr=float(np.percentile(np.concatenate(null),99.0))
    acc={"N_H":[], "good_tail":[], "good_tail_far":[]}
    for i in range(len(types)):
        q=l2n(torch.from_numpy(tec["feats"][int(o[i]):int(o[i+1])].astype(np.float32)).to(DEV))
        d=np.mean([nnd(q,sub[x]) for x in range(len(ids))],axis=0); del q
        k=max(1,int(round(ALPHA*len(d)))); tail=np.argpartition(d,len(d)-k)[-k:]
        if types[i]=="bad":
            g=np.asarray(tec["gt_frac"][int(o[i]):int(o[i+1])],float)
            G=np.where(g>GT_THR)[0]; clean=np.where(g==0)[0]
            if len(G) and len(clean):
                farc=clean[cheb(clean,G,C)>FAR]
                NH=np.intersect1d(tail,farc)
                if len(NH): acc["N_H"].append(float(np.mean(d[NH]>thr)))
        else:
            acc["good_tail"].append(float(np.mean(d[tail]>thr)))
    for gname,v in acc.items():
        if v: rows.append(dict(object=obj,group=gname,n=len(v),frac_over_p99=float(np.mean(v))))
    for x in sub.values(): del x
    print(f"  {obj} done",flush=True)
    pd.DataFrame(rows).to_csv(out/"r6a1c_good_tail_control.csv",index=False)
print("DONE")
