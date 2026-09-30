# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import importlib.util, os, zlib
import numpy as np, pandas as pd, torch
from scipy.ndimage import binary_dilation, distance_transform_cdt

IMAGE_EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
IMAGENET_MEAN=torch.tensor([0.485,0.456,0.406],dtype=torch.float32)[:,None,None]
IMAGENET_STD=torch.tensor([0.229,0.224,0.225],dtype=torch.float32)[:,None,None]

def stable_seed(name, base=20260930):
    return int(base + (zlib.crc32(str(name).encode("utf-8")) % 1000000))

def resolve_existing_root(project_root: Path, explicit: str, candidates):
    checks=[]
    if explicit:
        p=Path(explicit)
        checks.append(p if p.is_absolute() else project_root/p)
    for c in candidates:
        p=Path(c)
        checks.append(p if p.is_absolute() else project_root/p)
    for p in checks:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("No dataset root found. Checked:\n" + "\n".join(map(str,checks)))

def load_anomalydino_model(project_root: Path, cfg):
    ad=Path(cfg["anomalydino_dir"])
    if not ad.is_absolute(): ad=project_root/ad
    backbones=ad/"src"/"backbones.py"
    if not backbones.exists():
        raise FileNotFoundError(f"Missing AnomalyDINO backbones.py: {backbones}")
    spec=importlib.util.spec_from_file_location("r6a1e_anomalydino_backbones", backbones)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model(cfg["model_name"], cfg["device"], smaller_edge_size=int(cfg["resolution"]))

def build_image_index(base: Path):
    files=[p.resolve() for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    by_base={}
    by_stem={}
    for p in files:
        by_base.setdefault(p.name.lower(),[]).append(p)
        by_stem.setdefault(p.stem.lower(),[]).append(p)
    return files,by_base,by_stem

def resolve_name(name, index_tuple):
    files,by_base,by_stem=index_tuple
    s=str(name).replace("\\","/")
    p=Path(s)
    if p.exists(): return p.resolve()

    base=p.name.lower()
    if base in by_base and len(by_base[base])==1:
        return by_base[base][0]

    stem=p.stem.lower()
    if stem in by_stem and len(by_stem[stem])==1:
        return by_stem[stem][0]

    # suffix matching if cache stored relative path
    norm=s.lower().strip("/")
    hits=[q for q in files if str(q).replace("\\","/").lower().endswith(norm)]
    if len(hits)==1: return hits[0]

    raise RuntimeError(f"Cannot uniquely resolve cached image name: {name}; candidates={hits[:5]}")

def l2n_np(x,eps=1e-12):
    x=np.asarray(x,np.float32)
    return x/np.clip(np.linalg.norm(x,axis=1,keepdims=True),eps,None)

def cosine_positionwise(a,b):
    a=l2n_np(a); b=l2n_np(b)
    return np.sum(a*b,axis=1)

def cache_img_features(c,i):
    o=np.asarray(c["offsets"])
    return np.asarray(c["feats"][int(o[i]):int(o[i+1])],np.float32)

def make_support_bank(c,ids):
    return l2n_np(np.concatenate([cache_img_features(c,int(i)) for i in ids],axis=0))

def score_1nn(feats,bank,batch=1024):
    q=l2n_np(feats); b=np.asarray(bank,np.float32)
    out=[]
    bt=torch.from_numpy(b)
    use_cuda=torch.cuda.is_available()
    if use_cuda: bt=bt.cuda()
    for s in range(0,len(q),batch):
        qt=torch.from_numpy(q[s:s+batch])
        if use_cuda: qt=qt.cuda()
        with torch.inference_mode():
            sim=qt@bt.T
            d=1.0-sim.max(dim=1).values
        out.append(d.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def topmean(values,valid,alpha):
    v=np.asarray(values)[np.asarray(valid,bool)]
    if len(v)==0: return np.nan
    k=max(1,int(round(float(alpha)*len(v))))
    k=min(k,len(v))
    return float(np.partition(v,len(v)-k)[-k:].mean())

def dilate_patch_mask(mask,r):
    mask=np.asarray(mask,bool)
    if r==0: return mask.copy()
    return binary_dilation(mask,structure=np.ones((2*r+1,2*r+1),bool))

def patchmask_to_pixel(mask,patch_size,tensor_hw):
    H,W=mask.shape
    pix=np.repeat(np.repeat(mask,int(patch_size),axis=0),int(patch_size),axis=1)
    th,tw=map(int,tensor_hw)
    if pix.shape!=(th,tw):
        raise RuntimeError(f"pixel-mask shape {pix.shape} != prepared tensor {(th,tw)}")
    return pix

def replace_tensor(target,donor,pixmask):
    if tuple(target.shape)!=tuple(donor.shape):
        raise RuntimeError(f"tensor shapes differ: {target.shape} vs {donor.shape}")
    out=target.clone()
    m=torch.from_numpy(np.asarray(pixmask,bool))
    out[:,m]=donor[:,m]
    return out

def extract_batch(model,tensors,batch_size=8):
    ans=[]
    for s in range(0,len(tensors),batch_size):
        batch=torch.stack(tensors[s:s+batch_size],dim=0).to(model.device)
        with torch.inference_mode():
            # Same final-layer mechanism used by AnomalyDINO wrapper.
            z=model.model.get_intermediate_layers(batch)[0]
        ans.extend([x.detach().cpu().numpy().astype(np.float32) for x in z])
    return ans

def feature_drift(raw,cf,valid):
    raw=l2n_np(raw); cf=l2n_np(cf)
    d=1.0-np.sum(raw*cf,axis=1)
    v=np.asarray(valid,bool).ravel()
    return float(np.mean(d[v])) if v.any() else np.nan, d

def cheb_distance_from_gt(G):
    # scipy CDT chessboard distance to nearest TRUE cell.
    # distance_transform_cdt computes distance of nonzero cells to zero,
    # so use ~G: defect cells are zero, outside gets distance.
    return distance_transform_cdt(~np.asarray(G,bool),metric="chessboard").astype(int)

def ring_label(d):
    if 1<=d<=2: return "1-2"
    if 3<=d<=4: return "3-4"
    if 5<=d<=8: return "5-8"
    if 9<=d<=16: return "9-16"
    if d>16: return "17+"
    return "GT"

def denormalize_to_uint8(t):
    x=t.detach().cpu().float()*IMAGENET_STD+IMAGENET_MEAN
    x=x.clamp(0,1)
    return (x.permute(1,2,0).numpy()*255).round().astype(np.uint8)
