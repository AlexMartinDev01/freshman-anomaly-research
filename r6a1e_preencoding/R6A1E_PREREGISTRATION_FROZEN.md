# R6-A1E — Pre-Encoding Counterfactual Defect Removal
## Frozen preregistration

### 0. Why this experiment exists
R6-A1D removed score-map/token locations only AFTER DINO encoding.
Therefore a residual far-field score cannot distinguish:

A. true signal physically present in far-field pixels, versus
B. defect information propagated into far-field tokens by ViT global self-attention.

R6-A1E moves the intervention BEFORE DINO encoding.

### 1. Scientific question
If the annotated defect + surrounding context is replaced by normal pixels BEFORE
the image enters DINO, does the anomaly information in FAR-FIELD tokens disappear?

This is a mechanism/oracle audit, NOT a deployable detector.

### 2. Frozen scope
Objects:
- MVTec: screw, cable, bottle
- VisA: macaroni2, pcb2, chewinggum

shot = 4
split = 0
backbone = dinov2_vits14
resolution = 448
layer = final
GT defect threshold = gt_frac > 0.10

Radii in PATCH units:
r = {0, 2, 4, 8, 16}

Image aggregation:
MeanTop-alpha on FAR-FIELD VALID patches only.

alpha:
{0.1%, 0.25%, 0.5%, 1%, 2%, 5%}

Primary alpha = 1%.
Other alphas are frozen sensitivity analyses, never best-alpha selection.

### 3. Parity Gate — mandatory before intervention
The experiment MUST STOP if re-extracting ORIGINAL images through the current
AnomalyDINO/DINOv2 pipeline does not reproduce the frozen feature cache.

For two fixed diagnostic images/object (first valid good + first valid bad):
- grid must match cache exactly;
- feature count/dimension must match;
- median positionwise cosine(cache, re-extracted) >= 0.9995;
- 1st-percentile positionwise cosine >= 0.995.

If parity fails:
NO counterfactual result is scientifically interpretable.
Do not loosen the threshold after seeing data.

### 4. Counterfactual construction
For each BAD image b:
1. Build GT patch mask G: gt_frac > 0.10.
2. Dilate by Chebyshev radius r.
3. Convert patch mask exactly to DINO input pixels:
   every masked patch corresponds to a full patch_size x patch_size block.
4. Prepare a NORMAL donor image with the EXACT SAME DINO preprocessing.
5. Replace the masked input tensor pixels of b by donor pixels at SAME coordinates.
6. Re-run DINO from the modified INPUT tensor.
7. Score ONLY tokens outside the replacement mask.

This is:
image -> normal-pixel replacement -> DINO -> feature -> 1NN -> far-field score

NOT:
image -> DINO -> delete score tokens.

### 5. Normal donors
Primary donor pool:
TRAIN-GOOD images not used in the 4-shot support bank.

Exactly 2 donors/object, deterministically chosen with seed 20260930.

No donor is allowed to be a support image.

If fewer than 2 resolvable non-support train-good donors exist:
STOP that object. Do not silently use support donors.

### 6. Matched good control
Replacement itself can introduce a donor/target seam or global attention perturbation.

Therefore for each BAD image and each radius:
- freeze 3 GOOD test comparators, selected BEFORE score computation;
- apply the SAME bad-derived spatial mask;
- use the SAME normal donor;
- perform the SAME pre-encoding replacement.

Thus every bad-vs-good comparison has identical:
- mask geometry;
- replaced coordinates;
- donor identity;
- preprocessing;
- scoring region.

The comparison is causal with respect to target image label, not replacement artifact.

### 7. Primary analysis set — survivor matched
To remove the R6-A1D attrition confound:

For each object, define the PRIMARY bad-image set before running DINO as:
bad images retaining >=10 valid patches at r=16.

The SAME bad images are used at r=0,2,4,8,16.

Secondary "all available per radius" results may be reported but cannot override primary.

### 8. Primary endpoints

#### E1. Matched far-field discrimination
For each bad-good matched pair:
raw_win = 1[raw_bad > raw_good] (+0.5 ties)
cf_win  = 1[counterfactual_bad > counterfactual_good] (+0.5 ties)

Aggregate:
PWR_raw = mean(raw_win)
PWR_cf  = mean(cf_win)

Causal drop:
Delta_PWR = PWR_cf - PWR_raw

Negative Delta_PWR means pre-encoding defect removal suppresses bad-vs-good far-field evidence.

#### E2. Far-field feature drift
On far-field tokens only:

drift = mean_p [1 - cos(f_raw(p), f_cf(p))]

For the matched good controls compute the same quantity.

Primary contrast:
Delta_drift = drift_bad - mean(drift_good_matched)

If Delta_drift > 0, the same normal replacement changes far-field representation
more strongly in bad images than in good images.

#### E3. Far-field 1NN-score difference-in-differences
For each target:
Delta_score = score_cf - score_raw

Primary:
DiD = Delta_score_bad - mean(Delta_score_good_matched)

Negative DiD means pre-encoding defect replacement selectively suppresses
far-field anomaly evidence in bad images.

#### E4. Distance-stratified propagation
Using distance from ORIGINAL GT defect:
rings = {1-2, 3-4, 5-8, 9-16, >16} patches.

Report bad-good difference in feature drift and 1NN-score shift per ring.
This is a mechanism profile, not a tuned model.

### 9. Primary radii for adjudication
Headline adjudication uses:
r = 8 and r = 16, alpha = 1%.

All other radii/alphas are frozen sensitivity checks.

### 10. Frozen object-level decisions

#### PROPAGATION-SUPPORTED
For an object with raw matched PWR >= 0.60 at r=8 or r=16:

At either r=8 or r=16, all must hold:
1. donor-mean PWR drops by >= 0.05 after pre-encoding replacement;
2. BOTH donors have the same drop sign (cf < raw);
3. donor-mean Delta_drift > 0;
4. paired per-bad Delta_drift Wilcoxon two-sided p < 0.05.

Interpretation:
part of the post-encoding "far-field signal" was carried into far-field tokens
by information originating in the replaced defect/context region.

#### FAR-FIELD-PERSISTENT
At BOTH r=8 and r=16:
1. raw matched PWR >= 0.60;
2. counterfactual matched PWR >= 0.60;
3. |PWR_cf - PWR_raw| < 0.03;
4. both donors preserve the same sign/pattern.

Interpretation:
the residual cannot be explained primarily by propagation from the replaced region.
Next audit must distinguish physical global/material shift, acquisition covariate,
unannotated anomaly, or dataset shortcut.

#### MIXED / INDETERMINATE
Anything else.

No threshold may be changed after results are visible.

### 11. Macaroni2 negative-control role
R6-A1D already showed far-field image discrimination ~chance after local excision.
Macaroni2 is therefore a localized-regime negative control.

It is NOT required to satisfy raw PWR >= 0.60.
A large "propagation PASS" claim cannot be based on macaroni2 alone.

### 12. Important boundaries
- Normal donor replacement is an oracle counterfactual intervention, not deployment.
- This experiment does not prove a dataset shortcut.
- A persistent far-field signal is not automatically legitimate anomaly evidence.
- A propagation effect does not prove attention is the only mechanism.
- Do not design the final detector from this 6-object discovery set.
- Do not expand to 27 objects until the mechanism branch is frozen.

### 13. Next decision
If propagation is supported in the persistent A1D objects:
-> study local-to-global representation mixing / defect-cue localization.

If far-field signal persists after pre-encoding replacement:
-> run global-shift / acquisition / unannotated-defect source audit.

If macaroni2 remains localized while others differ:
-> object-specific mechanistic regimes are strengthened.

Only verified nuisance regimes proceed to:
D vs N_H directional identifiability (R6-C0).
