# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np, argparse

ap=argparse.ArgumentParser()
ap.add_argument("--input",default=r"results\model_v0\metrics\r6a_smoke\summary.csv")
a=ap.parse_args()
d=pd.read_csv(a.input)
d["hard"]=d.object.isin(["screw","macaroni2","pcb2"])
best=d.sort_values("patch_auc_probe").groupby("object").tail(1)
hard=best[best.hard]
mvt=best[best.object.isin(["screw","cable","bottle"])]
visa=best[best.object.isin(["macaroni2","pcb2","chewinggum"])]

hard_pass=(hard.patch_auc_probe.ge(.90)&hard.patch_auc_gain.ge(10)).sum()
print("=== R6-A SMOKE FROZEN READOUT ===")
print(best[["object","probe","patch_auc_probe","patch_auc_1nn","patch_auc_gain",
            "image_auc_probe","image_auc_1nn","image_auc_gain"]].round(4).to_string(index=False))
print()
print("Hard objects meeting >=.90 and >=+10 patch-AUC points:",hard_pass,"/ 3")
print("MVTec mean patch gain:",mvt.patch_auc_gain.mean())
print("VisA mean patch gain:",visa.patch_auc_gain.mean())
print("Smoke representation-sufficiency evidence:",
      (hard_pass>=2) and (mvt.patch_auc_gain.mean()>0) and (visa.patch_auc_gain.mean()>0))
print()
print("NOTE: 'best probe' is only a diagnostic summary between two preregistered linear probes;")
print("it is NOT a deployable method selection and must not be used to tune final algorithm.")
