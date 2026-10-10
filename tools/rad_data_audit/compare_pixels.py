#!/usr/bin/env python3
"""RAD foreground-masked vs unmasked RGB pixel check (sampled, metadata-only)."""
import csv,hashlib,io,json,sys,zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from compare_archives import parse_zip

def pixels(zipf, rec):
 with Image.open(io.BytesIO(zipf.read(rec["path"]))) as im:
  return np.array(im.convert("RGB"),dtype=np.uint8)

def main():
 p,q,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
 mz,m,mm,_=parse_zip(p);uz,u,um,_=parse_zip(q)
 shared=set(m)&set(u)
 groups=defaultdict(list)
 for key in sorted(shared):
  inst,split,defect,stem=key
  groups[(inst,split,defect)].append(key)
 keys=[records[0] for _,records in sorted(groups.items())]
 rows=[]
 for key in keys:
  aa=pixels(mz,m[key]);bb=pixels(uz,u[key])
  if aa.shape != bb.shape:raise RuntimeError("paired RGB shapes differ: "+repr(key))
  gap=np.abs(aa.astype(np.int16)-bb.astype(np.int16))
  pixelchange=np.any(gap>0,axis=2);pixelchange10=np.any(gap>10,axis=2)
  rows.append(dict(instance=key[0],split=key[1],defect=key[2],stem=key[3],
    width=aa.shape[1],height=aa.shape[0],exact_rgb_equal=bool(np.array_equal(aa,bb)),
    mean_channel_absolute_diff=float(gap.mean()),fraction_pixels_any_diff=float(pixelchange.mean()),
    fraction_pixels_gt10_diff=float(pixelchange10.mean()),
    masked_nearblack_fraction=float(np.all(aa<=3,axis=2).mean()),
    masked_nearwhite_fraction=float(np.all(aa>=252,axis=2).mean()),
    unmasked_nearblack_fraction=float(np.all(bb<=3,axis=2).mean()),
    unmasked_nearwhite_fraction=float(np.all(bb>=252,axis=2).mean())))
 # Exact ground-truth file correspondence and compressed or raw mask byte equality.
 def maskindex(files):
  d={}
  for name in files:
   parts=PurePosixPath(name).parts
   p=next((i for i,x in enumerate(parts) if x=="ground_truth"),None)
   if p is None or p==0:continue
   k=(parts[p-1],parts[p+1],parts[p+2]) if len(parts)>p+2 else None
   if k:d[k]=name
  return d
 md=maskindex(mm);ud=maskindex(um)
 mk=set(md)&set(ud)
 masks_equal=0;sample_disagree=[]
 # The masks are small enough to check the full 2,823 pair set.
 for key in sorted(mk):
  ah=hashlib.sha256(mz.read(md[key])).digest()
  bh=hashlib.sha256(uz.read(ud[key])).digest()
  if ah==bh:masks_equal+=1
  else:
   if len(sample_disagree)<15:sample_disagree.append(key)
 with open(out/"PAIRED_RGB_PIXEL_SAMPLES.csv","w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=dict(
  sampled_rgb_pairs=len(rows),
  all_sampled_rgb_equal=all(r["exact_rgb_equal"] for r in rows),
  mean_fraction_any_pixel_change=float(np.mean([r["fraction_pixels_any_diff"] for r in rows])),
  median_fraction_any_pixel_change=float(np.median([r["fraction_pixels_any_diff"] for r in rows])),
  mean_masked_nearblack=float(np.mean([r["masked_nearblack_fraction"] for r in rows])),
  mean_unmasked_nearblack=float(np.mean([r["unmasked_nearblack_fraction"] for r in rows])),
  mean_masked_nearwhite=float(np.mean([r["masked_nearwhite_fraction"] for r in rows])),
  mean_unmasked_nearwhite=float(np.mean([r["unmasked_nearwhite_fraction"] for r in rows])),
  masked_mask_count=len(md),unmasked_mask_count=len(ud),
  masks_common=len(mk), masks_same_sha256=masks_equal,
  masked_only_masks=len(set(md)-set(ud)),unmasked_only_masks=len(set(ud)-set(md)),
  mask_sha_mismatch_sample=sample_disagree,
  caveat="RGB comparison samples one image per instance/split/defect; mask hash comparison is exhaustive. This does NOT verify every RGB pair identical/different.",
  model_inference_performed=False,
 )
 (out/"RAD_RGB_AND_GT_DIFFERENCE.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print("RAD_PIXEL_SAMPLE_CHECK",json.dumps(report,ensure_ascii=False))
 mz.close();uz.close()
if __name__=="__main__":main()
