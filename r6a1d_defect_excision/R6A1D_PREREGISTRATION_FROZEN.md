# R6-A1D — Defect-Excision / Far-Field Signal Audit (FROZEN)

## Scientific question
Are the far-field GT-clean extreme patches in bad images merely nuisance false positives,
or do they carry non-local bad-image-associated information?

This is a mechanism audit, NOT a deployable method.

## Scope
Same six R6-A smoke objects:
- MVTec: screw, cable, bottle
- VisA: macaroni2, pcb2, chewinggum
- shot=4, split=0
- final-layer frozen DINOv2 1NN cosine-distance map
- GT threshold: gt_frac > 0.10

No object may be added/removed after results are seen.

## Core intervention
For each BAD image, define GT defect mask G on the patch grid.
Dilate G with Chebyshev radius:

r = {0, 2, 4, 8, 16} patches.

All patches inside the dilated mask are REMOVED from image scoring.

Important:
For every bad-vs-good comparison, apply the EXACT SAME spatial excision mask
from the bad image to the corresponding good image.
Thus bad and good use identical valid patch coordinates and identical valid-patch count.

If grids differ within an object, STOP rather than resize/interpolate.

## Image score
On remaining valid patches:

MeanTop_alpha, alpha =
{0.1%, 0.25%, 0.5%, 1%, 2%, 5%}

No best-alpha selection.

## Primary diagnostic statistic
Because every bad image has its own excision mask, use pairwise AUC / Mann–Whitney win probability:

For each bad image b and each good image g:
- score b using b's excision mask
- score g using the SAME mask coordinates
- count b>g, ties=0.5

Aggregate across all bad-good pairs.

This yields a fair "far-field-only" image discrimination statistic.

## Negative control — shape/area matched random translations
For every defect-centered excision mask and every radius:
- generate 100 random rigid translations of the SAME binary mask
- translation must keep the mask fully inside the image
- exact shape and exact removed area are preserved
- apply the translated mask to both the bad image and all good comparators

This tests whether a drop from defect-centered excision is specifically caused by
removing defect-associated spatial regions rather than merely removing many patches.

Seed = 20260930.

## Outputs
For each object, radius, alpha:
- pairwise_auc_centered
- random_translation_mean
- random_translation_q05
- random_translation_q95
- centered_minus_random_mean
- mean removed fraction
- mean valid patch count
- number bad images / good images

Also save per-bad-image:
- defect patch count
- removed patch count
- valid count
- bad score
- mean good comparator score
- pairwise win rate

## Interpretation rules

### A. Defect-local/context signal
Supported if, as radius increases:
- centered far-field AUC approaches ~0.5,
- centered AUC falls substantially below shape/area-matched random-mask null,
- effect is monotonic or near-monotonic across radii.

Interpretation:
Most image-level signal is localized to the annotated defect + surrounding context.

### B. Persistent far-field label signal
Supported if, even at r=8 or r=16:
- centered far-field AUC remains materially >0.5,
- and is not substantially lower than random-mask null.

Interpretation:
Bad images contain non-local label-associated signal outside the annotated defect neighborhood.
Do NOT yet call it "useful anomaly context"; it could be:
- global material/structural change,
- acquisition covariate,
- unannotated defect,
- dataset shortcut.

### C. Object-specific regimes
If some objects go to chance and others remain discriminative:
do not force a universal mechanism.
Proceed with regime decomposition.

## Important boundaries
1. This experiment does NOT prove causality of a dataset shortcut.
2. Far-field signal is not automatically legitimate anomaly evidence.
3. No spatial fusion, no new detector, no adaptive alpha is designed here.
4. The six objects remain mechanism-discovery only.
5. No paper-level universal claim until a later 27-object confirmatory experiment.

## Next decision
- If far-field signal collapses: proceed to D-vs-N_H directional identifiability only where N_H is verified nuisance.
- If far-field signal persists: run a far-field source audit (global shift / acquisition / unannotated defect) BEFORE designing any detector.
- If mixed: split objects by pre-registered mechanistic regime and confirm on all 27 objects.
