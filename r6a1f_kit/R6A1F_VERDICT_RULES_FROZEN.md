# R6-A1F verdict rules — FROZEN before the trace was run

The supplied kit ships a spec but no decision procedure. These rules were written
and encoded into `analyze_r6a1f.py` **before any R6-A1F number existed**. No
layer, radius, alpha, donor or threshold is chosen after seeing results.

## Fixed definitions (not results)

- Layer grid: 12 blocks, all traced. `depth_frac = (index+1)/12`.
- `early` = **block 2**. Fixed by arithmetic continuation of the project's
  pre-existing pin drop `mid=5` (50%), `midlate=8` (75%), `final=11` (100%) —
  equal 25%-of-depth steps, so early = 25% = block 2. It is a **label**, not a
  finding.
- `onset_layer` is a **result**, read off the dense curve. The two are never
  conflated.
- Radii {0,2,4,8,16}, alpha = 1% (the kit's config). Headline radii r=8, r=16.
- Primary drift functional: mean over far-field valid tokens. `drift_top01`
  (same top-1% functional as the image score) is carried alongside.

## Primary statistic — magnitude-controlled residual

The confound: a bad image's defect region differs from a normal donor **more**
than a good image's corresponding region does, so a larger drift is trivially
expected. Within each stratum `(object, layer, radius, donor)` — never pooled
across radii or layers, since both the mask area and the input→feature gain
change with depth and radius:

    Drift_good = a + b * delta        (fit on the matched-good series only)
    residual_b = Drift_bad(b) - (a + b * delta_b)

Covariate `delta` = mean |raw-donor| over the replaced pixels, denormalized.
Companion fit on `delta_token` = mean ||Δ patch_embed||. Guards: `fit_linear`
returns None if n<30 or sd(delta) < 5% of mean(delta) → emit `LOW_SPREAD` and
NaN, never a number. `EXTRAPOLATED` flagged if >20% of bad series exceed the
good maximum.

**Null.** Within-cluster label permutation: inside each
(bad_image, donor, radius) cluster of 4 series, permute which series is labelled
"bad". This preserves cluster structure, delta distribution, mask, donor and
sample sizes, destroying only the label. 2000 permutations, one-sided p.
Interval by cluster bootstrap over bad images, 2000 reps.

**Robustness.** Matched-subset: restrict to bad series whose delta lies inside
the good pool's central 90% and recompute drift_diff. If regression and matched
estimates disagree in sign, say so.

## Frozen verdict rules

- `onset_layer` = smallest `l*` with `resid(l*, r=16) > 0` **and** one-sided
  perm p < 0.05 **and** every `l' ∈ [l*, 11]` also has `resid > 0`. The
  "sustained to final" clause stops a noise spike being called onset.
  Otherwise `NONE_DETECTED`.
- `depth_trend`:
  - **HIERARCHICAL** if Spearman(block, resid at r=16) ρ ≥ 0.6 with p < 0.05
    **and** resid(11) > resid(2);
  - **INPUT_STAGE** if resid(2) ≥ 0.8 · resid(11);
  - else **MIXED**.
- Object verdict (final layer, both headline radii):
  - **PROPAGATION_SUPPORTED** — resid > 0 at r=8 **and** r=16, perm p < 0.05 at
    both, both donors same sign at both;
  - **UNDER_PROPAGATION** — resid < 0 at both with p < 0.05 (bad images move
    *less* than magnitude-matched goods);
  - **MAGNITUDE_EXPLAINED** — naive |drift_diff| > 0.01 somewhere at r∈{8,16}
    but the residual is not significant;
  - else **MIXED_INDETERMINATE**.
- Relation to anomaly collapse: Spearman across objects between
  `propagation_score` and the A1E `score_collapse`. **Power limit stated
  explicitly: with n=5 objects, |ρ|=1 gives p≈0.017 — directional evidence, not
  a correlation claim.**

## Boundaries

- The magnitude control equalizes the **amount** of input change; it cannot
  equalize the **kind** (`defect→normal` vs `normal→normal`). This is the honest
  ceiling of A1F and is stated in the report, not footnoted.
- A reverse-insertion arm (good image, bad-region pixels inserted) would invert
  that confound. It is NOT part of A1F and must be separately pre-registered.
- 5-object mechanism discovery. pcb2 is excluded by the kit's config; that is a
  subset of A1E's 6 and is stated.
- No detector design, no 27-object claim, no threshold tuning.
