# R6-A Smoke — Frozen-DINO Supervised Upper Bound (FROZEN)

## Scientific question
Does the frozen final-layer DINO representation already contain a linearly
recoverable defect signal that the normal-only 1NN readout fails to expose?

This is a DIAGNOSTIC ORACLE / UPPER-BOUND experiment.
It is NOT a deployable anomaly-detection method.

## Scope
Exactly 6 representative objects:
MVTec: screw, cable, bottle
VisA: macaroni2, pcb2, chewinggum

Layer: final only
Normal-only baseline: 4-shot, split=0, exact original 1NN cosine distance

No object may be added or removed after seeing results.

## Patch labels
defect patch: gt_frac > 0.10
clean patch: gt_frac == 0
boundary: 0 < gt_frac <= 0.10 is EXCLUDED.

## Split
Primary grouping unit: scene metadata if it exists in the cache; otherwise image.
No patch-level random split.

Use 5-fold stratified grouped CV when feasible.
Fallback to 3-fold only if 5-fold cannot keep both good/bad image groups represented.

Fold assignments MUST be saved once and reused by both Logistic Regression and Linear SVM.

## Train sampling
For each training image:
- <=512 clean patches
- <=512 defect patches
Sampling seed is fixed.
Test fold uses ALL eligible patches.

## Probes
1. L2 Logistic Regression
2. Linear SVM
Features are standardized using TRAIN fold statistics only.

No MLP / nonlinear kernel / fine-tuning / data augmentation.

## Outputs
Patch:
- AUROC
- Average Precision

Image diagnostic:
- top1% mean of probe decision score/probability
- AUROC

Normal-only baseline on exactly same held-out patches:
- 1NN patch AUROC/AP
- image top1 AUROC

## Smoke interpretation (NOT final paper Gate)
For hard objects (screw, macaroni2, pcb2):
- strong evidence for representation sufficiency if a linear probe gives
  patch AUROC >= 0.90 AND exceeds 1NN patch AUROC by >=10 points on >=2/3 hard objects.

For easy controls (bottle, cable, chewinggum):
- probe should not catastrophically fail; otherwise the probe pipeline is suspect.

Dataset direction:
- MVTec and VisA must both show positive mean supervised-vs-1NN patch AUROC gain
  before expanding to all 27 objects.

## Stop rules
1. If grouped-CV replay/folds are invalid, STOP.
2. Do not tune GT threshold after result.
3. Do not tune regularization per object after result.
4. Do not use test-fold statistics in StandardScaler.
5. Do not interpret supervised performance as a deployable method.
6. Do not start layer-wise R6-B until R6-A smoke is reviewed.
