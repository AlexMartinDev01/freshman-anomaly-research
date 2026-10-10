#!/usr/bin/env python3
"""Compare RAD masked and unmasked ZIPs by source-file identity only.
No models, no anomaly scores, no original image redistribution.
Usage: python compare_archives.py MASKED.zip NONMASKED.zip OUTPUT_DIR
"""
import csv, hashlib, io, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from PIL import Image

INSTANCES={"binderclip","binderclip2","bowl_upright","box","can","charger",
           "cup1_upright","cup2_upright","cup2_upright2","cup2_upright3",
           "gluebottle","gluebottle2","phonecase","phonecase2","rubberduck",
           "spoon_upright","spraybottle2","tennisball"}
IMAGE={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()

def parse_zip(p):
 z=zipfile.ZipFile(p)
 records={}
 extra=[]
 masks=[]
 for info in z.infolist():
  if info.is_dir():continue
  path=info.filename
  parts=PurePosixPath(path.replace("\\","/")).parts
  found=next((i for i,x in enumerate(parts) if x in INSTANCES),None)
  if found is None:
   extra.append(path);continue
  inst=parts[found];rest=parts[found+1:]
  if len(rest)>=3 and rest[0]=="ground_truth":
   masks.append(path);continue
  if len(rest)!=3 or rest[0] not in ("train","test") or Path(rest[-1]).suffix.lower() not in IMAGE:
   extra.append(path);continue
  split,defect,name=rest
  key=(inst,split,defect,Path(name).stem)
  # Same key + different extension should not silently overwrite.
  if key in records:raise RuntimeError("duplicate logical image key "+repr(key))
  records[key]={"path":path,"size":info.file_size,"ext":Path(name).suffix.lower(),"instance":inst,"split":split,"defect":defect}
 return z,records,masks,extra

def image_dimension(z,entry):
 with Image.open(io.BytesIO(z.read(entry["path"]))) as im:
  im.verify()
 with Image.open(io.BytesIO(z.read(entry["path"]))) as im:
  return list(im.size)

def main():
 maskp,fullp,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
 mz,m,masks,masked_other=parse_zip(maskp)
 uz,u,umasks,unmasked_other=parse_zip(fullp)
 a=set(m);b=set(u);both=sorted(a&b);monly=sorted(a-b);uonly=sorted(b-a)
 counts=[]
 for inst in sorted(INSTANCES):
  row={"instance":inst,"masked":sum(k[0]==inst for k in a),"unmasked":sum(k[0]==inst for k in b),
       "shared":sum(k[0]==inst for k in a&b),"masked_only":sum(k[0]==inst for k in a-b),
       "unmasked_only":sum(k[0]==inst for k in b-a)}
  counts.append(row)
 with open(out/"ARCHIVE_COUNTS_BY_INSTANCE.csv","w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
 with open(out/"ARCHIVE_FILE_DIFFERENCES.csv","w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(["instance","split","defect","stem","status","masked_path","unmasked_path"])
  for k in monly:w.writerow([*k,"MASKED_ONLY",m[k]["path"],""])
  for k in uonly:w.writerow([*k,"UNMASKED_ONLY","",u[k]["path"]])
 # Sample dimensions by instance, by split and by defect without reading every full-resolution image.
 samples=[]
 for inst in sorted(INSTANCES):
  for variant,records,z in (("masked",m,mz),("unmasked",u,uz)):
   candidates=[(k,r) for k,r in sorted(records.items()) if k[0]==inst]
   selected=[]
   for label in [("train","good"),("test","good")]:
    subset=[r for k,r in candidates if (k[1],k[2])==label]
    if subset:selected.append(subset[0])
   defects={k[2]:r for k,r in candidates if k[1]=="test" and k[2]!="good"}
   if defects:selected.append(next(iter(defects.values())))
   for row in selected:
    try:dim=image_dimension(z,row);error=""
    except Exception as exc:dim=[];error=str(exc)
    samples.append({"variant":variant,"instance":inst,"path":row["path"],"width":dim[0] if dim else "",
                    "height":dim[1] if dim else "","error":error})
 with open(out/"SAMPLE_IMAGE_DIMENSIONS.csv","w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
 report={
  "masked_archive_sha256":sha256(maskp),
  "unmasked_archive_sha256":sha256(fullp),
  "masked_size":maskp.stat().st_size,"unmasked_size":fullp.stat().st_size,
  "masked_images":len(m),"unmasked_images":len(u),"shared_image_keys":len(both),
  "masked_only_keys":len(monly),"unmasked_only_keys":len(uonly),
  "masked_ground_truth_entries":len(masks),"unmasked_ground_truth_entries":len(umasks),
  "masked_extra_files_count":len(masked_other),"unmasked_extra_files_count":len(unmasked_other),
  "counts_masked":dict(Counter((r["split"],r["defect"]) for r in m.values())),
  "counts_unmasked":dict(Counter((r["split"],r["defect"]) for r in u.values())),
  "examples_masked_only":[m[k]["path"] for k in monly[:30]],
  "examples_unmasked_only":[u[k]["path"] for k in uonly[:30]],
  "sample_dimensions":samples,
  "matching_rule":"instance + split + defect + filename stem (not full pixel hash)",
  "same_pixels_proven":False,
  "model_inference_performed":False,
 }
 # Tuple dict keys cannot be JSON serialized, convert to readable keys.
 for field in ("counts_masked","counts_unmasked"):
  report[field]={f"{k[0]}/{k[1]}":v for k,v in report[field].items()}
 (out/"RAD_ARCHIVE_COMPARISON.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("ARCHIVE_COMPARISON",json.dumps({k:report[k] for k in ("masked_images","unmasked_images","shared_image_keys","masked_only_keys","unmasked_only_keys","masked_ground_truth_entries","unmasked_ground_truth_entries")},ensure_ascii=False))
 mz.close();uz.close()
if __name__=="__main__":main()
