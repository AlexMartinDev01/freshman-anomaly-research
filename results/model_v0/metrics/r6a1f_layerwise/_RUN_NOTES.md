# R6-A1F run notes (returned with results)

## 0. The kit was a specification, not a runnable experiment

```
r6a1f_layer_trace.py  ->  raise SystemExit('请连接你的现有R6-A1E feature pipeline后运行')
r6a1f_precheck.py     ->  print('Precheck placeholder')
analyze_r6a1f.py      ->  print('Generate R6-A1F reports')
RUN_R6A1F.ps1         ->  no Set-Location; broken from any cwd
```
Kit SHA256 manifest verified ALL MATCH before any change. The three scripts were
implemented against the audited R6-A1E pipeline. The three reconstructed
originals are byte-exact against `SHA256_MANIFEST.txt` (verified by hash) and are
shipped as `_kit_patch_ORIGINAL_*.py`; `_kit_patch.diff` is the full diff
(+700 / -20). Two files added: `r6a1f_utils.py`, `R6A1F_VERDICT_RULES_FROZEN.md`.
No frozen parameter (radii / alpha / GT threshold / layer grid / donor / pair)
was changed.

## 1. Parity gate -- PASS

```
model: 12 blocks, patch 14, embed_dim 384, register_tokens 0
dense[11] == get_intermediate_layers(x)[0]   maxdiff = 0.00e+00   (bitwise, 10 images)
blocks 5/8/11 vs frozen cache:  min median_cos = 1.0 ,  min p01 = 0.99999982
N = 1024 / 1216 / 1024 / 1536 / 1024, asserted from FEATURE LENGTH
```
The last point matters: `phase_m1_rescue.load_cached` line 86 HARDCODES
`grids=(32,32)` for the stacked MVTec cache. It is correct for screw/cable/bottle
only by coincidence. The precheck asserts `N == H*W` from the feature array
itself and cross-checks `model.prepare_image`, so the hardcode cannot corrupt a
reshape silently.

## 2. Anchors -- A1F reproduces A1E exactly

**Cell-level (block 11), 90 bottle pairs:**
```
|A1F a1e_drift - A1E bad_feature_drift| : mean 4.7e-09  max 2.2e-08
correlation = 1.00000000
```

**Ring profile at radius 0, layer 11 -- 15/15 numbers match the A1E report:**
```
object       ring     A1E        A1F
bottle       1-2      0.0454     0.04540
bottle       >16      0.0138     0.01384
chewinggum   1-2      0.0722     0.07219
chewinggum   3-4      0.0119     0.01193
chewinggum   5-8      0.0060     0.00601
chewinggum   9-16     0.0041     0.00410
chewinggum   >16      0.0032     0.00320
cable        1-2      0.0297     0.02967
cable        3-4      0.0076     0.00756
cable        >16      0.0037     0.00365
```

## 3. Frozen verdicts

Rules frozen in `r6a1f_kit/R6A1F_VERDICT_RULES_FROZEN.md` BEFORE the trace ran.

```
object       onset_layer  depth_trend    rho      resid_r8  resid_r16  p_r16   verdict
bottle            3       HIERARCHICAL   0.986     0.0164    0.0183   0.0005  PROPAGATION_SUPPORTED
chewinggum        2       HIERARCHICAL   0.986     0.0075    0.0095   0.0005  PROPAGATION_SUPPORTED
cable       NONE_DETECTED INPUT_STAGE   -0.483    -0.0028   -0.0012   0.9940  MIXED_INDETERMINATE
macaroni2   NONE_DETECTED INPUT_STAGE   -0.944    -0.0007   -0.0029   1.0000  MIXED_INDETERMINATE
screw       NONE_DETECTED INPUT_STAGE   -0.874    -0.0021   -0.0380   1.0000  MAGNITUDE_EXPLAINED
```
0 NaN in 600 strata (resid_mean and perm_p).

## 4. Two honesty flags on my own frozen rule

**(a) `depth_trend` is not interpretable for the three non-propagating objects.**
The frozen `INPUT_STAGE` branch compares `resid(2) >= 0.8*resid(11)`, which was
written for a POSITIVE effect. On a near-zero or negative residual it is a
comparison outside its domain. The rule was NOT changed after seeing results;
instead `object_verdict.csv` carries `depth_trend_applicable` (True only when
resid(11) > 0), which is False for cable / macaroni2 / screw. Read those three
`INPUT_STAGE` labels as "no effect to trend", not as a mechanism claim.

**(b) The `UNDER_PROPAGATION` branch is not reachable as written.** It requires
`resid < 0 at both radii with p < 0.05`, but the permutation p as specified is
upper-tail only. A lower-tail p was added as a diagnostic (`perm_p_lower`,
`confound_control.csv`). It shows:
```
screw      r=16  resid -0.0380   p_lower 0.0005   <- significantly UNDER-propagating
macaroni2  r=16  resid -0.0029   p_lower 0.0015   (effect size ~0.003, negligible)
cable      r=16  resid -0.0012   p_lower 0.9595   not significant either tail
```
screw is significantly under-propagating at r=16 but not at r=8, so the frozen
"both radii" requirement correctly leaves it at MAGNITUDE_EXPLAINED.

## 5. What screw actually is (the four-way label working)

```
screw r=16:  naive drift_diff = -0.0608   (large)
             delta_pixel: bad 0.1007  good 0.1344     <- ratio 0.75, INVERTED
             magnitude-controlled resid = -0.0380, p_upper = 1.0000
```
The good controls received a LARGER input perturbation than the bad images. The
naive negative effect does not survive the magnitude control. This is A1E's
"good-side intervention artifact" made quantitative, and the verdict language
turns it into a finding rather than a defect.

## 6. Magnitude-control quality

```
object       overlap(bad delta inside good range)   extrap   matched-subset agrees?
bottle       0.80 / 0.86                            0.14-0.20   yes
chewinggum   0.95 / 0.92                            0.05-0.08   yes
cable        0.95 / 0.90                            0.05-0.10   yes
macaroni2    1.00 / 1.00                            0.00        yes
screw        0.88 / 0.75                            0.00        yes
```
bottle r=8 extrap = 0.20 sits exactly on the `EXTRAPOLATED` threshold (>0.2), so
it is not flagged but is marginal. Every object's matched-subset estimate agrees
in sign with the regression residual.

## 7. Q3 -- propagation vs anomaly collapse

```
bottle        prop +0.01735   collapse 11.25
chewinggum    prop +0.00851   collapse 27.93
cable         prop -0.00196   collapse 22.62
macaroni2     prop -0.00183   collapse  9.50
screw         prop -0.02003   collapse  2.22
Spearman rho = 0.500, p = 0.391   (n=5)
```
NOT supported. With n=5 this test has almost no power (|rho|=1 gives p~0.017),
and the observed ordering does not support "more propagation -> larger collapse".
Reported as directional evidence only.

## 8. Architecture of the effect (descriptive)

Both drift measures (cosine and relative-L2) agree, so the curves are not a
LayerNorm-magnitude artifact:
- **bottle**: ~0 in blocks 0-2, monotone accumulation through 3-10, then a
  5.8x cosine / 3.0x relL2 jump at block 11. `onset_layer = 3`.
- **chewinggum**: ~0 in blocks 0-2, smooth monotone accumulation, only 1.85x at
  block 11. `onset_layer = 2`. Distance-decaying: ring 1-2 to >16 falls 23x.
- **cable**: short-range only (ring 1-2 +0.0297, >16 +0.0037) and identically
  zero at r=16 where delta_pixel ratio is 1.02 -- a true null, not a weak test.
- **macaroni2**: near-field effect present but the smallest of all four
  (ring 1-2 = 0.0072 vs chewinggum 0.0722), far field ~0, all layers ~0 at r=16.

**Caveat:** the RADIUS profile at large r is contaminated by the replacement
seam (a 52-90% mask creates a large artificial discontinuity). e.g. chewinggum
radius profile rises r0 +0.00864 -> r16 +0.01063 while its ring profile decays
23x. The ring profile within a fixed small mask is the clean distance measure;
the radius profile at large r is not.

## 9. Boundaries

- The magnitude control equalises the AMOUNT of input change, not its KIND
  (`defect->normal` vs `normal->normal`). That confound is not removable by any
  statistic on these cells and is the honest ceiling of A1F.
- A reverse-insertion arm (good image, bad-region pixels inserted) would invert
  it; it is NOT part of A1F and must be separately pre-registered.
- 5 objects, mechanism discovery. pcb2 excluded by the kit's config (subset of
  A1E's 6). No detector design, no 27-object claim, no threshold tuning.
