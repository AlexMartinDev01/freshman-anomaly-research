# RC Root-Cause Program — Frozen preregistration

Status: **FROZEN BEFORE RC-GEO1 / RC-SUFF1 / RC-COV1 RESULTS ARE COMPUTED**
Base branch: `research-sync-20260930`
Experiment branch: `rc-rootcause-directional`

## Scientific target

The remaining competing explanations for few-shot failure are:

1. **Normal-support under-coverage**: hard normal patches are far from the sparse support because the support bank is too small.
2. **Scalar normality-geometry misalignment / readout insufficiency**: frozen DINO retains label-relevant high-dimensional structure, but a scalar nearest-normal distance conflates true defects with rare-but-valid normal deviations.
3. Contextual propagation is already known to be real, but A1E/A1F/X1 show it is object-dependent; R6-G is independently testing whether it explains baseline readout failure. RC experiments below do not reuse R6-G outputs.

No RC gate may be changed after its target result is inspected. If an implementation defect is found, preserve the failed output, mark it INVALIDATED, add an amendment, and rerun unchanged scientific criteria.

---

# RC-GEO1 — Same-image D vs N_H normality-geometry inversion

## Purpose

Test directly whether the frozen one-class geometry ranks **far-field GT-clean extreme patches (N_H)** as more abnormal than **true defect patches (D)** inside the same bad image.

This is not a proxy: `d1,d2,d5,d10` are actual distances to the frozen normal support bank from the corrected A1C audit.

## Inputs

- `results/model_v0/metrics/r6a1c_corrected/support_coverage_corrected.csv`
- `results/model_v0/metrics/r6a1c_corrected/layer_consistency_corrected.csv`
- `results/model_v0/metrics/r6a1c_corrected/neighbour_diversity_corrected.csv`
- `results/model_v0/metrics/r6a_smoke/summary.csv`

Primary object: **macaroni2**.
Reason fixed before this analysis: it is the strongest propagation-negative hard object (A1D far-field AUC -> chance; A1E/A1F broad propagation ~0) yet has severe 1NN image-readout failure.

Secondary hard objects: screw, pcb2.
Controls: bottle, chewinggum, cable.

## Unit and pairing

Statistical unit = one bad image.
Within each object/image, pair D and N_H from the same row family.

For k in {1,2,5,10}:
[
Delta_k(i)=d_k(N_H,i)-d_k(D,i).
]

Positive Delta means the scalar normality geometry assigns the clean hard-normal group greater novelty than the actual defect group.

For each layer in {mid, midlate, final}:
[
Delta_l(i)=d1_l(N_H,i)-d1_l(D,i).
]

No patch-level pseudo-replication.

## Statistics

Per object:
- median and mean paired Delta;
- one-sided paired Wilcoxon H1: Delta > 0;
- 20,000 image-cluster bootstrap draws for the median, seed 20260930;
- sign fraction P(Delta > 0).

No multiple-object pooling for the primary claim.

## Frozen primary gate: NORMALITY_GEOMETRY_MISALIGNMENT_SUPPORTED

For **macaroni2**, ALL must hold:

A. d1 median Delta > 0, one-sided Wilcoxon p < 0.01, and 95% image-bootstrap CI lower bound > 0.

B. At least 3 of {d1,d2,d5,d10} have median Delta > 0 and one-sided p < 0.05.

C. Final layer and at least one of {mid, midlate} have median Delta > 0 and one-sided p < 0.05.

D. The N_H nearest-neighbour provenance does not collapse to one support image:
   median distinct_support_images >= 2.0 AND median frac_from_one_image <= 0.80.

If A-C fail: this candidate is NOT established by GEO1.
If D fails but A-C pass: verdict = SUPPORT_IDENTITY_CONFOUNDED, not root-cause support.

Secondary objects are descriptive replication only and cannot rescue a failed macaroni2 primary gate.

## Interpretation boundary

Passing GEO1 proves a concrete geometric pathology in a propagation-negative hard case:
clean far-field extremes are placed farther from the normal support than true defect patches, robustly across neighbour order/depth.

It does NOT by itself prove why that geometry arose (support coverage vs representation/readout).

---

# RC-SUFF1 — Does frozen DINO contain label information not present in scalar 1NN image score?

## Purpose

Test whether the scalar 1NN image score is an insufficient statistic for label information already recoverable from frozen DINO.

This uses the real out-of-fold supervised probe scores from R6-A, not in-sample predictions.

## Inputs

- `results/model_v0/metrics/r6a_smoke/image_scores.csv`
- `results/model_v0/metrics/r6a_smoke/folds.csv`
- `results/model_v0/metrics/r6a_smoke/summary.csv`

Primary hard objects: **screw, macaroni2, pcb2**.
Controls: bottle, cable, chewinggum.
Probe families: LogReg and LinearSVM; neither may be selected post hoc.

## Models

Existing 5 frozen image folds are the outer evaluation folds.

For each object/probe/fold:

BASE:
[
y ~ z(score_{1NN})
]

EXTENDED:
[
y ~ z(score_{1NN}) + z(score_{probe,OOF})
]

The meta-classifier is fixed LogisticRegression(C=1, solver=liblinear, class_weight=balanced, max_iter=3000).

Standardization is fitted only on the four training folds.

Primary metric = held-out binary log loss.
Secondary = held-out AUROC and Brier score.

The probe score for an image is already generated by a patch classifier trained without that image's outer R6-A fold; therefore it is a diagnostic oracle feature, not a deployable unsupervised feature.

## Permutation negative control

Within object, preserve 1NN score and labels.
Permute the probe score **within the training/evaluation data as a whole before the outer meta-fit**, 5,000 deterministic permutations, seed 20260930.
For every permutation rerun all 5 outer folds.
Null statistic = total held-out log-loss improvement (BASE minus EXTENDED).

Permutation p = (1 + #null >= observed)/(5001).

## Frozen hard-object gate: SCALAR_READOUT_INSUFFICIENT

An object passes only if BOTH probe families satisfy:

A. total OOF log-loss improvement > 0;
B. permutation p < 0.01;
C. pooled OOF AUROC(EXTENDED) - AUROC(BASE) >= +0.05.

Root-cause-level support requires at least **2 of 3** hard objects to pass.
Easy controls are not required to improve and cannot substitute for hard objects.

## Interpretation boundary

Passing SUFF1 means the scalar 1NN score loses label information that remains recoverable from the same frozen representation under image-grouped label supervision.

It does NOT yet prove that the missing information is specifically residual direction; direct feature-level radial matching is reserved for RC-DIR.

---

# RC-COV1 — Existing real-support dose sensitivity (supporting only)

## Purpose

Assess whether simply increasing the number of real normal support images is already sufficient to explain failure.

## Input

`results/model_v0/metrics/gate17a_fusion.csv`

Use **alpha=0 only** (raw baseline), shots 1 and 4, all available pre-existing splits.

## Important design limitation

The repository's `draw_images()` seeds on shot count, therefore 1-shot and 4-shot banks are NOT nested. This prevents RC-COV1 from being a clean causal dose experiment. Accordingly it is explicitly **supporting evidence only**, never a root-cause gate.

Compute paired object x split Delta image-AUROC = AUC(4-shot)-AUC(1-shot), then object means and dataset-stratified summaries.

A dominant under-coverage pattern would require, descriptively:
- median object mean Delta >= +5 AUROC,
- >=12/15 objects positive,
- screw and cable each >= +5.

Failure of this descriptive pattern does not prove that all-shot coverage cannot help; it only argues against 1->4 support count as a dominant universal explanation.

A definitive nested 1/2/4/8/all support intervention with duplicate-bank control must be run later on raw caches if needed.

---

# Root-cause decision after RC-GEO1 + RC-SUFF1

If GEO1 passes AND SUFF1 root-cause support passes:

**NORMALITY_GEOMETRY_READOUT_BOTTLENECK — STRONGLY SUPPORTED**

Meaning:
- a propagation-negative hard object shows direct D vs N_H inversion in the actual one-class geometry;
- scalar 1NN score is not sufficient for label information retained by frozen DINO in >=2 hard objects.

This still stops short of saying **direction specifically** is the lost variable.

The next and final discriminating gate is RC-DIR:
strict radial matching of D and N_H on d1 (plus d2/d5 sensitivity), followed by direction-only vs magnitude-only classification and within-radius direction shuffle under image-grouped and defect-type-held-out evaluation.

If either GEO1 or SUFF1 fails, do not run RC-DIR as confirmation of this hypothesis; reopen the surviving competing mechanism instead.

---

# No result laundering

- R6-G is running independently and is not an input to these gates.
- Previously inspected AD2 shot curves are exploratory prior only and cannot be presented as new confirmatory evidence.
- Previously established A1C/A1F findings can motivate object choice but cannot substitute for the new gate statistics.
- No thresholds, primary objects, endpoints, or p-value directions may be changed after result generation.
