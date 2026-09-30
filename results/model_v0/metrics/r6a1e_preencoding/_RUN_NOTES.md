# R6-A1E run notes (returned with results)

## 1. Parity gate — PASS, exact

Gate required median_cos >= 0.9995 and p01_cos >= 0.995.
Measured on all 12 diagnostic images (first good + first bad, 6 objects):

    median_cos = 1.0        p01_cos = 1.0        min_cos = 1.0

Feature counts match the cache exactly (1024 / 1536 / 1312 / 1216), dim 384.
The counterfactual feature path is therefore in the SAME feature space as the
frozen cache. See `parity_report.csv`.

## 2. ONE KIT BUG FIXED — crash, not a criterion change

The shipped `r6a1e_preencoding_counterfactual.py` could not run:

    r6a1e_preencoding_counterfactual.py:178  raw_scores=[topmean(m,valid,alpha) for m in raw_target_maps]
    r6a1e_utils.py:100                       v=np.asarray(values)[np.asarray(valid,bool)]
    IndexError: axis 0 is 32 but boolean axis is 1024

Shape mismatch. `valid` (line 154) is deliberately flat (H*W,) and must stay flat:
line 174 `feature_drift(rf, cf, valid)` indexes a (H*W, D) array, and line 209
builds `idx` against the flat `dist`. The maps are 2-D (H,W) from lines 84/164.
Lines 204-205 already reshape explicitly; lines 178-179 were missed.

Fix (2 lines, content-preserving):

    -  raw_scores=[topmean(m,valid,alpha) for m in raw_target_maps]
    -  cf_scores=[topmean(m,valid,alpha) for m in cf_maps]
    +  raw_scores=[topmean(m.reshape(-1),valid,alpha) for m in raw_target_maps]
    +  cf_scores=[topmean(m.reshape(-1),valid,alpha) for m in cf_maps]

    original sha256  ee0a2407755b37d13f6135df9f47d506d89436a8a1baae5de447c435600d6b14
    patched  sha256  46b2ac6208e5ab070164ebe14148e7176c444e6b6c92d8aa3409ee910b24391b

`_kit_patch_ORIGINAL_counterfactual.py` = pristine copy. `_kit_patch.diff` = full diff.
The other 9 kit files are byte-identical to SHA256_MANIFEST.txt.
NO radius, alpha, GT threshold, donor count, pair count or decision threshold was touched.

## 3. Checklist item 4 — raw trajectory reproduces R6-A1D survivor-matched EXACTLY

`raw_full_good_baseline.csv` (alpha=1%) vs the independently recomputed
R6-A1D survivor-matched trajectory:

    object       r=0              r=16
    bottle      92.25 / 92.25    81.00 / 81.00
    cable       78.75 / 78.75    56.12 / 56.12
    screw       65.98 / 65.98    63.76 / 63.76
    pcb2        63.71 / 63.71    58.45 / 58.45
    macaroni2   57.90 / 57.90    48.40 / 48.40
    chewinggum  95.43 / 95.43    67.49 / 67.49

Survivor counts also match R6-A1D's r=16 n_bad_used exactly:
113 / 49 / 40 / 90 / 96 / 87.

## 4. Other pre-run audits

- resolved_test_paths.csv: no collisions (83/150/150/200/200/160 distinct files).
- donor_manifest.csv: 12/12 is_support=False, all train/good, grids match cache.
- pair_manifest.csv: exactly 3 matched goods per bad, all 6 objects.
- ring_profile / per_pair shapes validated against each other.

## 5. Frozen verdicts

    bottle       PROPAGATION_SUPPORTED
    chewinggum   PROPAGATION_SUPPORTED
    cable        MIXED_INDETERMINATE
    macaroni2    MIXED_INDETERMINATE
    pcb2         MIXED_INDETERMINATE
    screw        MIXED_INDETERMINATE

Stable across ALL SIX preregistered alphas (no best-alpha selection).
No NaN in any wilcoxon p-value (0/360).

## 6. Interpretation boundary (as preregistered)

The pre-encoding donor replacement is NOT a neutral intervention. Three
distinct profiles appear, and only the E2/E3 difference measures are immune
to the replacement offset:

    object       bad_shift vs good_shift        drift_diff(r=16)   verdict
    bottle       only bad moves (good ~0.000)   +0.0176            PROPAGATION
    chewinggum   bad moves more                 +0.0106            PROPAGATION
    screw        good moves more                -0.0608            MIXED
    cable        both ~equal                    +0.0001            MIXED
    macaroni2    both move together             +0.0011            MIXED
    pcb2         both move, bad more            +0.0161            MIXED (raw<0.60 at r=16)

bottle drift_diff > 0 in 100% of bad images (q05 = +0.0028) — unanimous.
chewinggum in 71.8%. Not driven by a few images.

These are 6 mechanism-discovery objects. No paper-level or 27-object claim.
