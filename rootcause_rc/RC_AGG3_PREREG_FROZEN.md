# RC-AGG3 — Real tail-width dose intervention
Status: **FROZEN BEFORE RC-AGG3 TARGET RESULTS ARE COMPUTED**

## Purpose
Stage2 already supported aggregation dilution in macaroni2 using:
- near-perfect within-image 1NN patch ranking;
- severe image-AUROC collapse;
- defect extent much smaller than fixed top-1% width;
- GT extent-matched oracle gain;
- defect-excision specificity.

RC-AGG3 tests the mechanism with a direct **non-GT image-readout intervention**:
keep every patch 1NN score fixed and change only the top-tail width alpha.

This is not a proxy. `r6a1_tail_occupancy.csv` stores the actual image-level 1NN tail mean at each frozen alpha.

## Input
`results/model_v0/metrics/r6a1_tail_audit/r6a1_tail_occupancy.csv`

Use arm = `1nn` only.

Frozen alphas:
0.001, 0.0025, 0.005, 0.01, 0.02, 0.05.

Primary object: **macaroni2**.
Controls: screw, pcb2, bottle, cable, chewinggum.

## Primary intervention contrast
Macaroni2 has median defect extent approximately 0.1–0.2% of patches from the already-frozen Stage2 analysis. Therefore the primary narrower non-GT readout is fixed **before this run** as:

alpha = 0.001

Baseline:
alpha = 0.01

No best-alpha selection is allowed for the primary claim.

For each alpha:
- image AUROC from actual `tail_mean`;
- image AP;
- bad-image median tail score;
- good-image median and p95 tail score;
- bad-image median GT precision and GT recall (mechanism diagnostics only).

## Bootstrap
Statistical unit = image.

For the primary AUROC difference:
Delta = AUROC(alpha=.001) - AUROC(alpha=.01)

Use a stratified image bootstrap:
- independently resample bad and good images with replacement;
- 20,000 draws;
- seed 20260930;
- preserve paired alpha scores by image_id within each resample.

Report percentile 95% CI.

## Frozen verdict for macaroni2
`TAIL_WIDTH_DILUTION_CAUSALLY_SUPPORTED` iff ALL:

A. Delta AUROC >= +0.05 absolute;

B. bootstrap 95% CI lower bound for Delta > 0;

C. image AP(alpha=.001) - image AP(alpha=.01) >= +0.05;

D. median bad-image GT precision at alpha=.001 is at least 2x the median precision at alpha=.01;

E. at least one of the larger widths alpha in {0.02,0.05} has AUROC <= AUROC(.01), showing that widening the same readout does not rescue the failure.

Secondary alpha=.0025 is reported but cannot rescue a failed primary alpha=.001 gate.

## Cross-object specificity
For every control compute the same primary Delta.

Specificity is SUPPORTING, not required:
- macaroni2 should be in the top two positive Deltas across six objects;
- bottle should remain near ceiling and need no rescue.

## Interpretation
Passing AGG3 strengthens the causal mechanism:
the same patch score field becomes substantially more discriminative when only the non-GT tail width is reduced toward the sparse defect scale.

It does not solve the patch-stage failure in screw/pcb2 and does not substitute for COV2 or DIR.
