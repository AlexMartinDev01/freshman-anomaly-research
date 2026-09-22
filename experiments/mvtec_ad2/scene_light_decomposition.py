# -*- coding: utf-8 -*-
"""
Scene vs Lighting decomposition (Task C).

test_public filenames are {scene:03d}_{lighting}.png with lighting in
{regular, overexposed, shift_1, shift_2, shift_3, underexposed}.

For each object (using every available shot/seed score CSV):
  * two-way ANOVA on GOOD image scores: fraction of variance explained by
    scene identity vs lighting condition
  * within-scene lighting range  Δlight = mean_scene(max_l s - min_l s)
  * between-scene spread        Δscene = std of scene means
  * defect effect size          Cohen's d between bad and good scores
  * per-lighting mean score shift vs regular

Question: is the dominant confounder "scene" (normal variability) or
"lighting" (distribution shift)?
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\work\freshman\experiments\mvtec_ad2")
from ad2_pipeline import AD2_OBJECTS, RESULTS_ROOT

OUT = os.path.join(RESULTS_ROOT, "metrics", "scene_light_decomposition.csv")


def parse_name(name):
    m = re.match(r"(\d{3})_(.+)\.png", name)
    return int(m.group(1)), m.group(2)


rows = []
light_rows = []
for csv_path in sorted(glob.glob(os.path.join(
        RESULTS_ROOT, "anomaly_scores", "*-shot_seed=*.csv"))):
    base = os.path.basename(csv_path)[:-4]
    obj = next(o for o in AD2_OBJECTS if base.startswith(o + "_"))
    shot = int(base.split("-shot")[0].split("_")[-1])
    seed = int(base.split("seed=")[1])
    df = pd.read_csv(csv_path)
    pub = df[df.split == "test_public"].copy()
    pub[["scene", "light"]] = pub["name"].apply(
        lambda n: pd.Series(parse_name(n)))
    good = pub[pub.type == "good"]
    bad = pub[pub.type == "bad"]
    if good.empty or bad.empty:
        continue

    # ---- two-way ANOVA (scene + lighting, no interaction) ----
    gm = good.groupby(["scene", "light"])["score"].mean().reset_index()
    # balanced design: 6 lightings x N scenes
    grand = gm["score"].mean()
    scene_means = gm.groupby("scene")["score"].mean()
    light_means = gm.groupby("light")["score"].mean()
    ss_total = ((gm["score"] - grand) ** 2).sum()
    n_light = gm["light"].nunique()
    n_scene = gm["scene"].nunique()
    ss_scene = n_light * ((scene_means - grand) ** 2).sum()
    ss_light = n_scene * ((light_means - grand) ** 2).sum()

    # ---- deltas ----
    rng = good.groupby("scene")["score"].agg(lambda s: s.max() - s.min())
    d_light = float(rng.mean())
    d_scene = float(scene_means.std())
    cohens_d = float((bad["score"].mean() - good["score"].mean())
                     / good["score"].std())

    rows.append({
        "object": obj, "shot": shot, "seed": seed,
        "ss_total": ss_total, "frac_scene": ss_scene / ss_total,
        "frac_light": ss_light / ss_total,
        "d_light": d_light, "d_scene": d_scene,
        "cohens_d_defect": cohens_d,
        "good_mean": good["score"].mean(), "good_std": good["score"].std(),
        "bad_mean": bad["score"].mean(),
    })

    # per-lighting shift vs regular
    reg = light_means.get("regular", np.nan)
    for light, m in light_means.items():
        light_rows.append({
            "object": obj, "shot": shot, "seed": seed, "light": light,
            "shift_vs_regular": float(m - reg),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.DataFrame(light_rows).to_csv(
    os.path.join(RESULTS_ROOT, "metrics", "scene_light_perlight.csv"), index=False)

print("=== Scene vs Lighting decomposition (1-shot, seed 0) ===")
sub = df[(df.shot == 1) & (df.seed == 0)]
show = sub[["object", "frac_scene", "frac_light", "d_scene", "d_light",
            "cohens_d_defect"]].copy()
show[["frac_scene", "frac_light"]] = (show[["frac_scene", "frac_light"]] * 100).round(1)
show[["d_scene", "d_light"]] = show[["d_scene", "d_light"]].round(3)
show["cohens_d_defect"] = show["cohens_d_defect"].round(2)
print(show.to_string(index=False))

print("\n=== per-lighting mean shift vs regular (1-shot seed0, all objects) ===")
pl = pd.DataFrame(light_rows)
pl = pl[(pl.shot == 1) & (pl.seed == 0)]
print(pl.pivot_table(index="light", values="shift_vs_regular",
                     aggfunc="mean").round(4).to_string())
print("\nsaved:", OUT)
