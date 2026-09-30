# TSN Formal Failure — Actual Static Root-Cause Audit

This report is computed from the uploaded 243-cell formal result package.

## 1. Formal failure is object-level, not a cell-correlation illusion
- 27/27 object-level mean deltas are negative.
- Mean object delta: -33.982 AUROC points.
- Object-bootstrap 95% CI: [-38.531, -28.923].
- Wilcoxon signed-rank (alternative: delta < 0): p=7.451e-09.

Therefore the failure remains decisive when the independent block is the object, not the individual cell.

## 2. Shot count is not the main cause
Mean delta:
- k2: -33.707
- k4: -33.764
- k8: -34.475

Friedman shot test p=0.862. There is no evidence that increasing support count systematically rescues the method.

Support-q dispersion DOES increase with shot:
- k2 mean std=0.0568
- k4 mean std=0.0957
- k8 mean std=0.1186

Yet performance remains similarly poor. This dissociates support dispersion from the primary failure.

## 3. Split is also not the main cause
Friedman split test p=0.355.

## 4. The q_ref relationship flips by the pre-registered hard-quartile regime
The baseline bottom quartile threshold is 90.766 AUROC.

Hard quartile:
- rho(q_ref, delta)=-0.434, p=4.734e-04

Upper three quartiles:
- rho(q_ref, delta)=0.524, p=3.282e-14

The sign flips. Robust interaction regression gives a strong q_ref × hard-regime interaction.

Implication:
q_ref magnitude is not a universal "calibration quality" scalar. Its meaning is regime-conditioned.
A simple global center-correction story is insufficient.

## 5. Positive cells are exceptions in weak-baseline regimes
There are 8/243 nonnegative cells.
All are in screw / macaroni1 / macaroni2 and all have baseline AUROC <= 75.010.

This does NOT prove a valid router; it identifies the right positive controls for causal diagnosis.

## 6. TSN itself is usually mediocre, not merely losing to a ceiling baseline
- mean TSN AUROC: 58.987
- median: 58.040
- below chance (AUROC < 50): 74/243
- >=70 AUROC: 53/243

Therefore ceiling effects cannot explain the collapse by themselves.

## 7. What cannot be causally answered from this package
The uploaded formal package contains no per-test q values and no raw DINO train/test features.
Thus the following cannot honestly be run in this environment:
- oracle test-good center
- AUC(q) vs AUC(-q)
- matched-jackknife bank scoring
- shuffled bank-identity control
- patch-count subsampling

These require the original E:\work\freshman feature caches.

## 8. Next decisive test
Use `11_causal_selection_v2.csv` on the original machine and compute:
1. formal replay
2. q / -q
3. oracle full-k good center
4. matched support-only
5. matched oracle
6. wrong/shuffled pairing
7. same-size common-reference control

Only after those results should a replacement mechanism be proposed.
