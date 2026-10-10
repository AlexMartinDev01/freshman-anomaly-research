#!/usr/bin/env python3
"""Read-only RAD archive inspection. No anomaly-model evaluation.
Requirements: pip install pillow numpy huggingface_hub
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, random, zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from PIL import Image
import numpy as np

MAPPING = {
 "binderclip":"binderclip", "binderclip2":"binderclip",
 "bowl_upright":"bowl", "box":"box", "can":"can", "charger":"charger",
 "cup1_upright":"cup1", "cup2_upright":"cup2",
 "cup2_upright2":"cup2", "cup2_upright3":"cup2",
 "gluebottle":"gluebottle","gluebottle2":"gluebottle",
 "phonecase":"phonecase","phonecase2":"phonecase",
 "rubberduck":"rubberduck","spoon_upright":"spoon",
 "spraybottle2":"spraybottle","tennisball":"tennisball"}
EXPECTED_SHA="ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee"
IMG={".jpg",".jpeg",".bmp",".png",".tif",".tiff"}

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(4<<20),b""): h.update(b)
 return h.hexdigest()

def write_csv(path,rows,fields):
 with open(path,"w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def zip_parts(name):
 p=PurePosixPath(name.replace("\\","/")).parts
 for i,part in enumerate(p):
  if part in MAPPING:return part,p[i+1:]
 return None,()

def main():
 parser=argparse.ArgumentParser()
 sub=parser.add_subparsers(dest="mode",required=True)
 p=sub.add_parser("download");p.add_argument("--root",required=True)
 p=sub.add_parser("audit");p.add_argument("--archive",required=True);p.add_argument("--out",required=True)
 p.add_argument("--seed",type=int,default=20261010);p.add_argument("--draws",type=int,default=5)
 a=parser.parse_args()
 if a.mode=="download":
  from huggingface_hub import hf_hub_download
  root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
  name="Anomaly_refine_msk.zip";target=root/name
  if not target.exists():target=Path(hf_hub_download(repo_id="zhouk777/RAD",filename=name,repo_type="dataset",local_dir=str(root)))
  sha=digest(target)
  if sha!=EXPECTED_SHA:raise RuntimeError("RAD release hash mismatch: "+sha)
  print("VERIFIED_RAD_ARCHIVE",target,sha);return
 archive=Path(a.archive).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 sha=digest(archive)
 if archive.name=="Anomaly_refine_msk.zip" and sha!=EXPECTED_SHA:raise RuntimeError("Release hash mismatch: "+sha)
 rows=[];maskfiles={};unknown=[];invalid=[]
 with zipfile.ZipFile(archive) as z:
  bad=z.testzip()
  if bad:raise RuntimeError("ZIP CRC failed: "+bad)
  entries={i.filename:i for i in z.infolist() if not i.is_dir()}
  # Two-pass index: masks often occur AFTER anomaly images inside ZIP.
  for candidate in entries:
   inst,parts=zip_parts(candidate)
   if inst is not None and len(parts)>=3 and parts[0]=="ground_truth" and Path(parts[-1]).suffix.lower() in IMG:
    maskfiles[(inst,parts[1],parts[-1])]=candidate
  for path,info in entries.items():
   inst,parts=zip_parts(path)
   if inst is None:unknown.append(path);continue
   if len(parts)>=3 and parts[0]=="ground_truth" and Path(parts[-1]).suffix.lower() in IMG:
    maskfiles[(inst,parts[1],parts[-1])]=path;continue
   if len(parts)!=3 or parts[0] not in ("train","test") or Path(parts[-1]).suffix.lower() not in IMG:
    unknown.append(path);continue
   split,defect,basename=parts
   if split=="train" and defect!="good":unknown.append(path);continue
   rec=dict(path=path,category=MAPPING[inst],instance=inst,split=split,defect=defect,label=int(defect!="good"),
            width="",height="",image_sha256="",mask="",mask_status="",mask_nonzero="")
   try:
    blob=z.read(path);rec["image_sha256"]=hashlib.sha256(blob).hexdigest()
    with Image.open(io.BytesIO(blob)) as im:
     im.load();rec["width"],rec["height"]=im.size
   except Exception as e: invalid.append([path,repr(e)]);rows.append(rec);continue
   if rec["label"] and split=="test":
    stem=Path(basename).stem
    mp=next((maskfiles[k] for n in (f"{stem}_mask.png",f"{stem}.png")
            if (k:=(inst,defect,n)) in maskfiles),None)
    # Masks may precede or follow the image in the archive: collected in the initial scan.
    if mp is None:rec["mask_status"]="MISSING"
    else:
     rec["mask"]=mp
     try:
      with Image.open(io.BytesIO(z.read(mp))) as im:
       im.load();pixels=np.asarray(im.convert("L"));rec["mask_nonzero"]=int(np.count_nonzero(pixels))
       rec["mask_status"]=("SIZE_MISMATCH" if im.size!=(rec["width"],rec["height"])
                           else "ALL_ZERO" if rec["mask_nonzero"]==0 else "OK")
     except Exception as e:rec["mask_status"]="DECODE_ERROR";invalid.append([mp,repr(e)])
   rows.append(rec)
 rows.sort(key=lambda r:r["path"])
 fields=["path","category","instance","split","defect","label","width","height","image_sha256","mask","mask_status","mask_nonzero"]
 write_csv(out/"RAD_ACTUAL_DATA_MANIFEST.csv",rows,fields)
 counts=Counter((r["category"],r["instance"],r["split"],r["defect"]) for r in rows)
 write_csv(out/"RAD_COUNTS.csv",[dict(category=k[0],instance=k[1],split=k[2],defect=k[3],count=v)
           for k,v in sorted(counts.items())],["category","instance","split","defect","count"])
 pool=defaultdict(list);test=defaultdict(Counter)
 for r in rows:
  if r["split"]=="train" and r["label"]==0:pool[r["category"]].append(r)
  elif r["split"]=="test":test[r["category"]][r["label"]]+=1
 eligibility=[];supports=[]
 for cat in sorted(set(MAPPING.values())):
  choices=sorted(pool[cat],key=lambda r:r["path"])
  ok=len(choices)>=4 and test[cat][0]>0 and test[cat][1]>0
  eligibility.append(dict(category=cat,train_good=len(choices),test_good=test[cat][0],test_bad=test[cat][1],eligible=ok))
  if not ok:continue
  for draw in range(a.draws):
   rng=random.Random(f"RAD_v1/{a.seed}/{cat}/{draw}")
   groups=defaultdict(list)
   for r in choices:groups[r["instance"]].append(r)
   pools={k:rng.sample(v,len(v)) for k,v in sorted(groups.items())}
   selected=[]
   while len(selected)<4:
    for inst in sorted(pools):
     if pools[inst] and len(selected)<4:selected.append(pools[inst].pop())
   for i,r in enumerate(selected):
    supports.append(dict(category=cat,draw=draw,seed=a.seed,shot=i,instance=r["instance"],path=r["path"],image_sha256=r["image_sha256"]))
 write_csv(out/"RAD_ELIGIBILITY.csv",eligibility,["category","train_good","test_good","test_bad","eligible"])
 write_csv(out/"RAD_4SHOT_SUPPORT_FROZEN.csv",supports,["category","draw","seed","shot","instance","path","image_sha256"])
 hash_groups=defaultdict(list)
 for r in rows:hash_groups[r["image_sha256"]].append(r["path"])
 dup={h:p for h,p in hash_groups.items() if h and len(p)>1}
 summary=dict(archive=str(archive),sha256=sha,images=len(rows),categories=len({r["category"] for r in rows}),
              instances=len({r["instance"] for r in rows}),normal=sum(not r["label"] for r in rows),anomaly=sum(r["label"] for r in rows),
              mask_status=dict(Counter(r["mask_status"] for r in rows if r["label"] and r["split"]=="test")),
              duplicate_image_groups=dup,unknown_entries=unknown,invalid_images=invalid,
              support_draws=a.draws,eligible_categories=sum(int(r["eligible"]) for r in eligibility),
              detector_inference_performed=False)
 (out/"RAD_FILE_AUDIT.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
 print("RAD_AUDIT_COMPLETE",json.dumps({k:summary[k] for k in ["images","categories","instances","normal","anomaly","mask_status","eligible_categories","detector_inference_performed"]},ensure_ascii=False))
if __name__=="__main__":main()
