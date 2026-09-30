# R6-A1C Corrected Audit — Frozen Repair

This rerun exists because code review found three implementation defects in the
previous R6-A1C package:

1. kNN order statistic bug:
   np.partition(d, kmax-1)[:,:kmax] does not sort the first kmax values.
   Therefore d1/d2/d5 from the old coverage file are not valid kth-nearest distances.
   d10 was the only guaranteed order statistic.

2. Layer query bug:
   old layer-consistency pass indexed tr[layer] with TEST image/patch indices.
   Correct version queries te[layer].

3. Detectability bug:
   old matched nnd() used max(1-cosine), i.e. farthest-neighbour distance.
   Correct version uses min(1-cosine) = 1-max(cosine), i.e. true 1NN distance.

No thresholds, objects, shot, split, FAR radius, alpha, or support draw are changed.

Scope:
- same six R6-A smoke objects
- shot=4, split=0
- GT_THR=.10
- N_H = GT-clean, Chebyshev distance >4 from defect, inside final-layer top1% 1NN tail
- N_R_far matched from far GT-clean non-tail patches
- layers = mid, midlate, final
- k = 1,2,5,10

Corrected Audit 4 uses bank-size-matched true 1NN:
- null: each support image vs other 3
- test: average true-1NN distance over the four 3-image sub-banks

Good-tail control remains mandatory.

Interpretation boundary:
Even if N_H and good-tail remain indistinguishable by corrected true-1NN distance,
the permitted conclusion is ONLY:
    "normal-only 1NN-distance tail statistic does not distinguish them."
It is NOT evidence that every possible normal-only statistic is fundamentally unable
to distinguish them.

R6-C is entered only if the corrected result reproduces:
- N_H remains outside support coverage across k;
- behavior is not final-layer-only;
- corrected 1NN-distance statistic fails to separate N_H from good-image extreme tail.
