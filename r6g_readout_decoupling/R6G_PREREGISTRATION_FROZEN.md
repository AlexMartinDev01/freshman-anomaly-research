# R6-G — 12-Layer Normal-1NN vs Supervised-Oracle Readout Decoupling
## FINAL ROOT-CAUSE GATE — FROZEN BEFORE RUN

### 0. Why this is the last R6 experiment
R6-A1F established that encoder-mediated contextual propagation is real and
object-dependent, with:
- bottle: hierarchical broad propagation;
- chewinggum: hierarchical distance-decaying propagation;
- cable: short-range context only;
- macaroni2: localized / far-field-null control;
- screw: intervention-artifact control.

What remains unproven is the final causal bridge:

    Contextual propagation
        ?→
    normal-only readout failure

This experiment asks whether defect information remains linearly recoverable
while normal-only 1NN readout becomes increasingly deficient through depth.

Regardless of outcome, R6 ENDS after this gate.
No R6-H / A1G propagation-description experiment is permitted.

---

## 1. Frozen scope
Objects:
- bottle
- chewinggum
- cable
- macaroni2
- screw

shot = 4
split = 0
DINOv2 ViT-S/14
all 12 blocks: 0..11

The exact R6-A smoke image folds MUST be reused.
The exact 4-shot support IDs MUST match the R6-A smoke summary.

No fold regeneration is allowed.
No support redraw is allowed.

---

## 2. Patch labels
Same as R6-A:

defect:
    gt_frac > 0.10

clean:
    gt_frac == 0

boundary:
    0 < gt_frac <= 0.10
    EXCLUDED

No GT-threshold tuning.

---

## 3. Normal-only readout
For every layer l:

    d_l(q) = 1 - max_{r in R_normal,l} cosine(q, r)

The support bank is the SAME 4-shot train-normal image set at every layer.

No k tuning.
No PCA.
No prototype refinement.
No spatial fusion.

---

## 4. Supervised diagnostic oracle
Use the EXACT R6-A grouped image folds.

For every layer and fold:
- StandardScaler fit on training patches only
- max 512 clean + max 512 defect patches per training image
- deterministic sampling, same patch indices for every layer
- LogReg:
    C=1.0, balanced, liblinear, max_iter=3000
- LinearSVM:
    C=1.0, balanced, max_iter=10000

No layer-specific hyperparameter tuning.
No post-hoc regularization.

LogReg is PRIMARY.
LinearSVM is mandatory robustness.

The supervised probe is diagnostic/oracle only.

---

## 5. Primary metrics
All primary discrimination metrics are computed FOLD-WISE first and then
macro-averaged across folds. This avoids mixing score scales across independently
trained supervised folds.

Per layer:
1. patch AUROC
2. patch AP  **PRIMARY patch metric**
3. image AUROC using fixed top-1% mean
4. mean bad-image Precision@Top1%
5. mean bad-image Recall@Top1%

Patch AP is emphasized because R6-A1 showed that AUROC can look near-perfect
while a small number of extreme clean outliers destroy the image tail.

No best-layer selection is a scientific claim.

---

## 6. Readout-gap definitions
For probe p at layer l:

    Gap_AP(l,p)   = AP_probe(l,p)   - AP_1NN(l)
    Gap_AUC(l,p)  = AUC_probe(l,p)  - AUC_1NN(l)
    Gap_IMG(l,p)  = IMG_probe(l,p)  - IMG_1NN(l)
    Gap_Purity(l,p) = Precision@Top1_probe(l,p)
                      - Precision@Top1_1NN(l)

These are diagnostic gaps, not deployable gains.

---

## 7. Propagation alignment
Reuse the frozen R6-A1F magnitude-controlled r=16 curve:

    P(l) = resid_mean(object, radius=16, layer=l)

No new propagation statistic is invented.

For each object and each probe report:
- Spearman[P(l), Gap_AP(l)]
- Spearman[P(l), Gap_AUC(l)]
- Spearman[P(l), Gap_Purity(l)]

These 12 layers are correlated measurements.
The p-values are DESCRIPTIVE SUPPORT, not independent-sample proof.
Pattern logic below is primary.

---

## 8. Frozen cue-retention definitions
Reference early layer = block 2, fixed before this experiment.

For each probe:

    CueAUC_drop = AUC_probe(final) - AUC_probe(block2)
    CueAP_drop  = AP_probe(final)  - AP_probe(block2)

CUE_PRESERVED for a probe if BOTH:
- CueAUC_drop >= -0.03
- CueAP_drop  >= -0.10

CUE_FADED for a probe if BOTH:
- CueAUC_drop <= -0.05
- CueAP_drop  <= -0.15

Otherwise cue state = MIXED.

This does NOT claim block2 is optimal; it is the already-frozen 25%-depth
reference used throughout A1F.

---

## 9. Frozen readout-distortion definitions
For each probe:

GAP_COUPLED if ALL:
1. Spearman(P, Gap_AP) >= +0.60
2. final Gap_AP > block2 Gap_AP
3. final Gap_Purity > block2 Gap_Purity

GAP_DECOUPLED if ALL:
1. Spearman(P, Gap_AP) <= +0.30
2. final Gap_AP <= block2 Gap_AP + 0.02

Otherwise = MIXED.

The AP gap is primary. AUROC/image correlations are supporting diagnostics.

---

## 10. Frozen object verdict
Only objects with A1F verdict PROPAGATION_SUPPORTED are eligible for the
strong root-cause verdict.

### READOUT_DISTORTION_SUPPORTED
A1F = PROPAGATION_SUPPORTED
AND for BOTH LogReg and LinearSVM:
- CUE_PRESERVED
- GAP_COUPLED

Interpretation:
deep contextual propagation grows while label-informed defect information is
retained, but the normal-only extreme-tail readout increasingly fails to use it.

### CUE_FADING_SUPPORTED
A1F = PROPAGATION_SUPPORTED
AND for BOTH probes:
- CUE_FADED

Interpretation:
the representation itself loses supervised defect separability through depth.
Propagation may be associated, but the dominant evidence is cue fading.

### PROPAGATION_DECOUPLED_FROM_READOUT
A1F = PROPAGATION_SUPPORTED
AND for BOTH probes:
- CUE_PRESERVED
- GAP_DECOUPLED

Interpretation:
propagation is a real representation phenomenon but is not supported as the
cause of the normal-only readout failure.

### MIXED_INDETERMINATE
Anything else.

For cable/macaroni2/screw:
report the same curves as controls, but do not promote them to a universal
root-cause verdict.

---

## 11. Cross-object Q3
Across the 5 objects, correlations between propagation strength and readout
failure are EXPLORATORY ONLY because n=5.

They cannot override the within-object frozen verdict.

---

## 12. Mandatory pipeline gates
STOP if any fails:

1. R6-A smoke folds.csv / summary.csv missing
2. A1F dense_layer_curve.csv / object_verdict.csv missing
3. A1E resolved_test_paths.csv missing
4. DINO dense block11 does not reproduce final cache:
   median positionwise cosine >= 0.9995
   p01 >= 0.995
5. support IDs do not exactly match R6-A smoke summary
6. support image paths cannot be uniquely resolved
7. test image names do not exactly align with frozen R6-A folds
8. any layer has a different token grid from the frozen final cache

No threshold loosening after seeing data.

---

## 13. Interpretation boundaries
- Supervised probes use GT; they are NOT deployment methods.
- High supervised separability proves information is linearly recoverable under
  label information, not that normal-only inference can identify that direction.
- A propagation/readout association does not prove a specific attention head.
- All 12 blocks share network ancestry; layer-wise rho is a mechanistic trend,
  not 12 independent experimental units.
- Five objects are mechanism discovery, not benchmark-wide prevalence.
- R6 ends after this experiment.

---

## 14. R7 branch after this gate
If READOUT_DISTORTION_SUPPORTED:
    R7 = Context-Stable / Propagation-Aware Normality Modeling.

If CUE_FADING_SUPPORTED:
    R7 = defect-cue-preserving / intermediate-layer representation route.

If PROPAGATION_DECOUPLED:
    close propagation-as-root-cause;
    R7 = D-vs-N_H directional identifiability / normality geometry.

If mixed:
    use object-regime split, but do NOT restart open-ended R6 exploration.
