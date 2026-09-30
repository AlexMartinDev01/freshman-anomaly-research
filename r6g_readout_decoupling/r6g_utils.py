# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import importlib.util, zlib
import numpy as np
import torch

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

def stable_seed(name, base=20260930):
    return int(base + zlib.crc32(str(name).encode("utf-8")) % 1000000)

def l2n_np(x, eps=1e-12):
    x=np.asarray(x,np.float32)
    return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),eps,None)

def cosine_positionwise(a,b):
    a=l2n_np(a); b=l2n_np(b)
    return np.sum(a*b,axis=1)

def cache_img_features(c,i):
    o=np.asarray(c["offsets"])
    return np.asarray(c["feats"][int(o[i]):int(o[i+1])],np.float32)

def build_image_index(base: Path):
    files=[p.resolve() for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    by_base={}; by_stem={}
    for p in files:
        by_base.setdefault(p.name.lower(),[]).append(p)
        by_stem.setdefault(p.stem.lower(),[]).append(p)
    return files,by_base,by_stem

def resolve_name(name,index_tuple):
    files,by_base,by_stem=index_tuple
    s=str(name).replace("\\","/")
    p=Path(s)
    if p.exists(): return p.resolve()
    b=p.name.lower()
    if b in by_base and len(by_base[b])==1: return by_base[b][0]
    st=p.stem.lower()
    if st in by_stem and len(by_stem[st])==1: return by_stem[st][0]
    norm=s.lower().strip("/")
    hits=[q for q in files if str(q).replace("\\","/").lower().endswith(norm)]
    if len(hits)==1: return hits[0]
    raise RuntimeError(f"Cannot uniquely resolve {name}; hits={hits[:8]}")

def resolve_existing_root(project_root: Path, explicit: str, candidates):
    checks=[]
    if explicit:
        p=Path(explicit); checks.append(p if p.is_absolute() else project_root/p)
    for c in candidates:
        p=Path(c); checks.append(p if p.is_absolute() else project_root/p)
    for p in checks:
        if p.exists(): return p.resolve()
    raise FileNotFoundError("Dataset root not found. Checked:\n"+"\n".join(map(str,checks)))

def load_anomalydino_model(project_root: Path, cfg):
    ad=Path(cfg["anomalydino_dir"])
    if not ad.is_absolute(): ad=project_root/ad
    p=ad/"src"/"backbones.py"
    if not p.exists(): raise FileNotFoundError(p)
    spec=importlib.util.spec_from_file_location("r6g_backbones",p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.get_model(cfg["model_name"],cfg["device"],smaller_edge_size=int(cfg["resolution"]))

def extract_dense(model,tensors,layers=range(12)):
    batch=torch.stack(tensors,dim=0).to(model.device)
    with torch.inference_mode():
        z=model.model.get_intermediate_layers(batch,n=list(layers))
    # return per image: (L,N,D)
    out=[]
    for i in range(batch.shape[0]):
        out.append(np.stack([zz[i].float().cpu().numpy() for zz in z],axis=0).astype(np.float32))
    del batch,z
    return out

def score_1nn(feats, bank, chunk=1024):
    q=l2n_np(feats); b=l2n_np(bank)
    bt=torch.from_numpy(b)
    if torch.cuda.is_available(): bt=bt.cuda()
    ans=[]
    for s in range(0,len(q),chunk):
        qt=torch.from_numpy(q[s:s+chunk])
        if torch.cuda.is_available(): qt=qt.cuda()
        with torch.inference_mode():
            sim=qt@bt.T
            d=1.0-sim.max(dim=1).values
        ans.append(d.cpu().numpy())
    return np.concatenate(ans).astype(np.float64)

def topmean(v, frac=.01):
    v=np.asarray(v,float).ravel()
    if len(v)==0: return np.nan
    k=max(1,int(np.ceil(frac*len(v))))
    k=min(k,len(v))
    return float(np.partition(v,len(v)-k)[-k:].mean())

def tail_precision_recall(y,score,frac=.01):
    y=np.asarray(y,int); score=np.asarray(score,float)
    if len(y)==0: return np.nan,np.nan,0
    k=max(1,int(np.ceil(frac*len(y)))); k=min(k,len(y))
    idx=np.argpartition(score,len(score)-k)[-k:]
    tp=int((y[idx]==1).sum())
    prec=tp/k
    ndef=int((y==1).sum())
    rec=np.nan if ndef==0 else tp/ndef
    return float(prec),float(rec),k
