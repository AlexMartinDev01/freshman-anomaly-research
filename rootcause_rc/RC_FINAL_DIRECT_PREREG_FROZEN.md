# RC Final Direct Gates — Frozen before raw-cache execution

Status: **FROZEN BEFORE RC-COV2 / RC-DIR RAW-FEATURE RESULTS**
Execution target: the existing local project with the real R0 feature caches. These gates must NOT be replaced by summaries or proxies.

## Scientific state entering this gate

Stage 1 did not support a universal D-vs-N_H distance inversion in macaroni2. Stage 2 localized macaroni2 to a causal aggregation-dilution failure and supported an architecture-level downstream readout bottleneck.

Two questions remain before the deepest root cause is stated:

1. Is the patch-stage failure primarily caused by too little normal support?
2. If not, does scalar normal distance discard defect-sensitive **directional** information that remains in the frozen DINO vector?

These are answered directly below.

---

# RC-COV2 — Nested real-normal support intervention

## Objects

Primary hard: screw, macaroni2, pcb2.
Controls: bottle, cable, chewinggum.

Representation: R0 final DINO layer only.
Patch label: defect iff gt_frac > 0.10; clean iff gt_frac == 0; boundary excluded.
Image score: mean of top ceil(1%) eligible patch distances, exactly matching R6-A.

## Intervention

For each object and seed r in {0,1,2,3,4}:
- create ONE deterministic permutation of train/good image IDs;
- support sets are nested prefixes:
  S1 subset S2 subset S4 subset S8 subset S16 subset S32 subset S64 subset Sall.
- If fewer than k train images exist, that finite k is omitted.
- Sall uses every train/good image and is identical across seeds.

Nothing else changes: same query features, same cosine 1NN rule, same labels, same image aggregation.

This is a real support-coverage intervention, not a proxy.

## Metrics

For every object / seed / support size:
- patch AUROC;
- patch AP (primary patch endpoint);
- image AUROC (primary image endpoint);
- good-image top1% score median and p95;
- bad-image top1% score median.

## Cardinality negative control

For each object seed at S1:
- construct a bank by exact repetition of S1 patch vectors until its patch count is at least the S4 bank patch count;
- verify on three deterministic test images that duplicated-bank 1NN scores equal S1 scores with max absolute difference <= 1e-7.

Any failure invalidates COV2 implementation.

## Frozen under-coverage verdict

For each hard object:
- Delta_img = image_AUROC(Sall) - mean_seed image_AUROC(S4)
- Delta_AP  = patch_AP(Sall) - mean_seed patch_AP(S4)

Object-level COVERAGE_RESCUE if BOTH:
- Delta_img >= +0.10 absolute AUROC;
- Delta_AP >= +0.20 absolute AP;
- and Sall image AUROC >= 0.90.

`UNDERCOVERAGE_ROOT_SUPPORTED` only if at least 2 of 3 hard objects are COVERAGE_RESCUE.

`UNDERCOVERAGE_CONTRIBUTOR_ONLY` if root criterion fails but at least one hard object has Delta_img >= +0.05 or Delta_AP >= +0.10.

Otherwise `UNDERCOVERAGE_NOT_SUPPORTED`.

No monotonicity requirement is imposed; support can legitimately change defect distance as well as normal coverage. The decisive test is whether full real-normal coverage actually rescues the hard failures.

---

# RC-DIR — Radial-matched D vs N_H directional identifiability

## Primary object

**screw**.

Reason fixed before execution:
- severe patch AP gap: 1NN 0.159 vs supervised LogReg 0.816;
- Stage-1 leakage-free image matching passed scalar-readout insufficiency for both probes;
- unlike macaroni2, screw is a patch-stage failure candidate.

Replication: pcb2.
Controls: cable, bottle, chewinggum, macaroni2.

## Group construction

Exactly reproduce corrected A1C on final R0 features using the frozen 4-shot split-0 support:
- D = gt_frac > 0.10;
- clean = gt_frac == 0;
- far clean = Chebyshev distance > 4 from any D patch;
- d1 tail width k = round(1% of all image patches);
- N_H = far-clean patches that lie in the image's d1 top tail.

For every D and N_H patch compute exact cosine kNN distances d1,d2,d5,d10 and nearest-normal patch index.

All DINO query and bank features are L2-normalized.

Residual direction:
  u = normalize(q - n1)

For unit q and n1, ||q-n1|| is a monotonic function of d1; the matching below removes this radial magnitude.

## Primary matching

Within each bad image AND nearest support-image slot:
- standardized radial covariates = [d1,d2,d5,d10];
- standardization scale is the pooled candidate SD for that object, computed before matching;
- Hungarian one-to-one assignment minimizes Euclidean radial-covariate distance;
- retain a pair only if every absolute standardized radial difference <= 0.20.

Each retained pair contributes exactly one D and one N_H patch.

Primary matching is valid only if:
- >=50 retained pairs for screw;
- absolute SMD <=0.10 for each of d1,d2,d5,d10.

A relaxed same-image-only match may be reported as sensitivity but can never rescue a failed primary matching gate.

## Readout arms

Fixed classifiers: LogisticRegression(C=1, balanced, liblinear, max_iter=3000) and LinearSVM(C=1, balanced, max_iter=10000).

Five-fold GroupKFold by bad-image ID. StandardScaler is fit on training folds only.

Arms:
1. d1 only
2. radial4 = [d1,d2,d5,d10]
3. anchor-only = nearest normal feature n1
4. direction-only = u
5. raw query q

Primary metric = pooled OOF AUROC.
Secondary = pooled OOF AP.

No hyperparameter selection.

## Pair-swap negative control

For the primary screw matched pairs:
- independently swap the two direction vectors inside each D/N_H pair with probability 0.5;
- keep labels, radial values, images, folds, and anchors fixed;
- rerun direction-only LogReg with the exact same GroupKFold;
- 500 deterministic permutations, seed 20260930.

p = (1 + # null_AUC >= observed_AUC) / 501.

## Defect-type-held-out test

For screw only:
- defect type = directory component of the bad image name;
- Leave-One-Defect-Type-Out;
- a held-out type contributes only if it has >=10 matched pairs;
- train on every other type, test on the held-out type;
- aggregate OOF predictions over eligible types.

At least 3 eligible defect types are required.

## Frozen DIRECTIONAL_IDENTIFIABILITY_SUPPORTED verdict

ALL must hold for screw:

A. primary matching valid (>=50 pairs; all four radial SMD <=0.10);

B. direction-only OOF AUROC:
   - LogReg >= 0.75;
   - LinearSVM >= 0.70;

C. direction adds information beyond radial/local-density geometry:
   - LogReg direction AUC - radial4 AUC >= +0.15;

D. anchor semantic confound is insufficient:
   - anchor-only AUC <=0.65 OR direction AUC - anchor AUC >= +0.10;

E. pair-swap permutation p < 0.01;

F. defect-type-held-out:
   - >=3 eligible types;
   - pooled direction LogReg AUROC >=0.70.

If A fails: verdict MATCHING_INSUFFICIENT, not a mechanistic failure.
If A passes but B/C/E fail: directional hypothesis is rejected.
If D fails: ANCHOR_SEMANTIC_CONFOUNDED.
If F fails: direction is type-specific, not a general defect-sensitive direction.

pcb2 is replication only because VisA image names do not expose reliable defect subtypes in the same manner.

---

# Final root-cause decision

1. If COV2 root passes:
   normal-support under-coverage is a major root and DIR is interpreted as residual readout structure.

2. If COV2 root fails AND DIR passes:
   the patch-stage root is **scalar radial readout information loss**: the normal-only 1NN scalar throws away defect-sensitive directional structure retained by frozen DINO.

3. Combined with already frozen LOC2 macaroni2 result, if COV2 root fails and DIR passes, the architecture-level deepest statement is:

> Frozen DINO is not the principal information bottleneck. The dominant failure lies downstream in a two-stage normal-only readout: patch deviations are collapsed to radial novelty, discarding defect-sensitive direction for geometry-hard objects, and the resulting scalar map is collapsed again by fixed-width extreme-tail pooling, diluting sparse defects.

4. If DIR fails after valid matching, do not use the directional explanation. Retain the broader `DOWNSTREAM_READOUT_BOTTLENECK_SUPPORTED` conclusion and reopen the exact patch-stage variable.

No threshold may be changed after raw-cache execution.
