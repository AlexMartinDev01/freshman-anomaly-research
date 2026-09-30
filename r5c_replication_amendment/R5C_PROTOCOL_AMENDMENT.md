# R5-C Locked Replication — Protocol Amendment

## Why this amendment exists
The original locked replication selection used split=2 for shot 2/8.
However, the pre-registered R5-C implementation intentionally computes spatial
shuffle only when split==0, so the submitted replication package contains no
R5-C result.

This is an execution-design mismatch, not an R5-C failure.

## Amendment made BEFORE seeing any R5-C replication result
To replicate spatial evidence across shot count while preserving the original
R5-C implementation and all frozen spatial parameters:

- objects: all 27
- shots: 2 and 8
- split: 0 only
- spatial alphas: unchanged [0.005, 0.01, 0.02, 0.05]
- shuffle reps: unchanged 100
- metrics: unchanged LCC and adjacency
- null comparison: unchanged shuffle q95
- no alpha selection
- object remains the headline independent unit

Total: 54 cells.

## Frozen replication criteria
For each shot separately:
1. adjacency original > shuffle q95 in >=70% objects at every pre-registered alpha.
2. MVTec and VisA both show positive mean adjacency effect at every alpha.
3. No post-hoc sign flip or alpha selection.
4. q/top1 histogram-only scores are not used to judge spatial replication.

LCC is secondary; failure of LCC alone does not invalidate adjacency if adjacency
meets its pre-registered criterion, because MAIN already showed adjacency was the
uniformly strongest spatial statistic.

## Interpretation
- PASS at both 2-shot and 8-shot: spatial signal is cross-shot replicated.
- PASS at one shot only: shot-dependent spatial signal; do not promote as general branch.
- FAIL both: R5-C MAIN does not replicate; spatial branch is closed.

Do not run R6-A until this amendment is complete if the goal is to freeze all R5 conclusions first.
