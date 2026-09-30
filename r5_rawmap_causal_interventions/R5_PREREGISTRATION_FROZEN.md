# R5 RAW-MAP CAUSAL INTERVENTIONS — FROZEN PROTOCOL

## Scientific question
Does the image-level failure arise because a one-dimensional extreme-tail/ratio
readout is structurally non-monotone to defect strength/extent, and does the raw
patch map contain independent spatial evidence discarded by histogram-only scores?

No final method is designed in R5. R5 is root-cause identification only.

## Scope
MAIN = all 27 MVTec+VisA objects, 4-shot, support splits 0/1/2 = 81 cells.
Object is the independent statistical block.
2-shot/8-shot split-2 cells are LOCKED replication and must not be inspected
until the MAIN verdict is frozen.

## Raw score
Frozen DINOv2 final-layer cosine NN distance, reconstructed with exactly the
same support draw and bank code as formal TSN.

## GT patch rule
defect patch: gt_frac > 0.10
clean patch: gt_frac == 0.0

## R5-A: Strength-only intervention
On each real BAD image:
- keep patch positions, defect extent, clean scores, and topology fixed
- robust clean scale s = 1.4826 * MAD(clean), fallback std
- add lambda*s to GT defect patches only
- lambda = [0, .25, .5, 1, 2, 4]

Readouts:
- exact formal top1% mean
- max
- q = log(max/top1%)

Primary causal tests:
A1. per-image Spearman(lambda, top1) should be >=0 (positive control)
A2. q is structurally pathological if object-block median rho_q < 0,
    >=70% objects have mean rho_q < 0, exact sign p<.05
A3. paired (rho_top1 - rho_q) >0 across objects, Wilcoxon one-sided p<.05
A4. q pathology must replicate in both datasets at the object level.

No tuning of lambda grid after observing results.

## R5-B: Extent-only intervention on REAL NORMAL backgrounds
For every GOOD test map:
- choose a fixed synthetic anomaly strength h = original_max + robust_clean_scale
- replace the m LOWEST-scoring patches by h (coordinatewise increase only)
- extent alpha = [.001, .0025, .005, .01, .02, .05, .10]
- m = max(1, round(alpha*N))

Thus strength is fixed and only the number of anomalous patches changes.
This is a controlled counterexample on real normal score fields.

Primary causal tests:
B1. top1 must be nondecreasing for 100% of images (implementation positive control)
B2. q is non-monotone/pathological if >=80% images have rho(alpha,q)<0 after
    the first implanted patch and every object has negative median rho_q,
    object sign p<.05
B3. max should become approximately constant after the first implant; this verifies
    that q decline is caused by denominator/top-tail catch-up, not weakening strength.

## R5-C: Spatial-information test
Only MAIN split=0 is used to limit repeated-map dependence.
For each original map, at alpha=[.005,.01,.02,.05]:
- take the top-alpha rank mask
- compute largest 4-connected component fraction among selected patches (LCC)
- compute selected-neighbor adjacency ratio

These statistics use RANK POSITIONS only; score histogram is unchanged.
Then perform 100 histogram-preserving random spatial permutations/map.

Primary tests:
C1. image AUROC(original spatial statistic) is compared with shuffle null.
C2. spatial evidence is promoted only if the same orientation beats shuffle q95
    in >=70% objects and both datasets show positive object-block effect.
C3. If sign/orientation flips across objects, "single scalar coherence" is rejected.
C4. q/top1 must be exactly invariant to spatial permutation (unit control).

## Statistical unit
Object, not cell or image.

Report:
- per-image intervention curves
- per-cell summaries
- per-object averages across the 3 support splits
- exact sign tests / Wilcoxon at object level
- MVTec and VisA separately
- no cell-level p-values used as headline evidence

## Stop rules
- If A fails: do not claim strength non-monotonicity as general mechanism.
- If B fails: do not claim unknown extent as structural source of q sign reversal.
- If C fails: do not add a spatial branch solely on this evidence.
- No new mechanism is promoted until MAIN is complete.
- LOCKED replication is run only after MAIN conclusion is written.
