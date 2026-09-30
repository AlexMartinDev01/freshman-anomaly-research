# RC Stage 2 — Failure-locus causal decomposition

Status: **FROZEN BEFORE STAGE-2 TARGET STATISTICS ARE COMPUTED**

## Why this gate exists

RC Stage 1 rejected the pre-registered macaroni2 hypothesis that far-field hard-normal patches are systematically farther from the support than true defects. The sign was the opposite across d1/d2/d5/d10 and across mid/midlate/final.

Therefore the next question is not allowed to be "try another geometry metric". The failure locus must be localized first.

Primary object remains **macaroni2**, because independent earlier experiments establish:
- broad contextual propagation is absent;
- the object has severe image-level 1NN failure;
- final DINO retains strong supervised defect information.

The candidate now tested is:

> true defect patches are locally ranked correctly, but the fixed top-1% image readout is wider than the sparse defect extent, so true defect evidence is diluted by clean-tail evidence and cross-image tail variability.

This is an image-readout mechanism, not a representation mechanism.

---

# RC-LOC2 — macaroni2 aggregation-dilution root gate

## Inputs

1. `results/model_v0/metrics/r6a_smoke/per_image_patch_metrics.csv`
2. `results/model_v0/metrics/r6a_smoke/summary.csv`
3. `results/model_v0/metrics/r6a1b_selection_halo/r6a1b_oracle_ladder.csv`
4. `results/model_v0/metrics/r6a1d_defect_excision/object_radius_alpha_summary.csv`

Only existing frozen outputs are used. No R6-G result is read.

## A. Local scalar ranking is already strong

Use LogReg rows only to avoid duplicate copies of the identical 1NN per-image metrics.

On bad macaroni2 images with both classes present:
- compute median within-image `patch_auc_1nn`;
- compute fraction with `patch_auc_1nn >= 0.90`.

Pass A if:
- median >= 0.95; AND
- >= 80% of bad images are >= 0.90.

Interpretation: the scalar 1NN patch score itself is not globally broken inside the image.

## B. Yet image classification fails

Use the frozen R6-A summary.

Pass B if:
- `image_auc_1nn <= 0.75`.

This creates the required localization contrast: high within-image patch ranking but poor cross-image image score.

## C. Fixed tail is wider than the actual defect

For every bad macaroni2 image:
- `k_i = ceil(0.01 * n_eligible_i)`, exactly matching R6-A.
- extent ratio = `n_defect_i / k_i`.

Pass C if:
- median extent ratio <= 0.25; AND
- >= 90% of bad images satisfy `n_defect_i < k_i`.

This is a structural condition, not a performance statistic.

## D. Causal width intervention helps

Use A1B within its own pairwise-AUC evaluation only:
- L1 = fixed 1% tail;
- L2 = GT extent-matched tail.

Pass D if:
- L2 - L1 >= +5.0 AUC points for macaroni2.

No comparison to R6-A AUROC is made because the evaluation conventions differ.

## E. Removing the true defect region is specifically harmful

Use A1D at radius=0, alpha=0.01:
- centered defect-region excision pairwise AUC;
- shape/area-matched translated-mask random-null mean.

Pass E if BOTH:
- centered AUC <= 60;
- random-null mean - centered AUC >= +5 points.

This is the causal specificity check: removing the actual defect evidence must hurt more than removing an equally sized translated region.

## Frozen verdict

`AGGREGATION_DILUTION_ROOT_SUPPORTED` for macaroni2 iff A+B+C+D+E all pass.

If A fails: scalar patch geometry remains a candidate.
If A passes but C/D/E fail: fixed-width dilution is not established as the root.
If A+B+C pass but D/E fail: report structural sparsity only, not causal root.

## Controls

The same A/B/C descriptive quantities are reported for screw, pcb2, bottle, cable, chewinggum, but cannot rescue a failed macaroni2 gate.

---

# RC-ARCH2 — architecture-level readout bottleneck synthesis

This is a **classification of failure locus**, not a new statistical gate.

Using only frozen results:
- Patch-stage loss = large supervised probe AP minus 1NN patch AP.
- Aggregation-stage failure = LOC2 pass despite high local 1NN patch AUC.
- Propagation = prior A1F label only for interpretation, not a criterion.

The architecture-level statement "readout bottleneck" may be called **SUPPORTED** only if:
1. RC-LOC2 passes for macaroni2; AND
2. at least one other hard object has R6-A patch AP gap >= +0.30 (absolute AP) showing a patch-stage readout loss; AND
3. the frozen DINO supervised patch AUROC on those hard objects is >= 0.95.

This would establish two distinct failure locations downstream of the same frozen representation:
- scalar patch readout failure in at least one hard object;
- image aggregation failure in macaroni2.

It does NOT establish which high-dimensional feature component is lost at the patch stage. RC-DIR remains required for that final mechanistic question.
