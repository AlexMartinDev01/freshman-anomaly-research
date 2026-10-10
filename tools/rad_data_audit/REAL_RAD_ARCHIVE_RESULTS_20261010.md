# RAD actual two-archive audit — verified 2026-10-10

This report derives from actual downloads of BOTH original Hugging Face ZIP archives into ephemeral GitHub Actions runners; it is not inferred from paper tables.

## Actual released file counts

| Property | Masked ZIP | Unmasked ZIP |
|---|---:|---:|
| Archive size (bytes) | 210,859,648 | 5,851,950,481 |
| SHA256 | ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee | c2c3082b24d7e855359c0e4883adbbac4ba1a98f914f56f66064bbc83a4c6d50 |
| RGB image count | 4,047 | 4,047 |
| Normal train / normal test / anomalous test | 1,062 / 162 / 2,823 | 1,062 / 162 / 2,823 |
| Semantic categories / instance directories | 13 / 18 | 13 / 18 |
| ground_truth masks | 2,823 | 2,823 |

All 4,047 image logical keys (instance, split, defect, filename stem) match across the archives with NO exclusive keys.
All 2,823 paired pixel ground-truth masks have identical SHA256 bytes. The public description suggesting that only the small ZIP has ground-truth masks is incorrect for these exact release files.

Among 84 real, paired RGB image samples, 0 pairs were pixel-identical, and 95.50% of pixel locations differed on average. Masked images contain 95.513% near-white pixels, versus 0.0115% in unmasked images. Both have 1291x721 image dimensions in every sampled pair. This strongly indicates that the small version is foreground/background whitened RGB, NOT simply low resolution; the full-resolution nonmasked images preserve the real surrounding scene. Exact transformation for every image was not reconstructed.

## Real file integrity

- Masked ZIP hash and CRC checked, every RGB image decoded.
- All 2,823 anomalous test images match a nonzero, image-size-matched and decodable pixel mask.
- Zero missing masks; zero all-zero masks; zero dimension mismatches; zero invalid image decodes.
- No duplicate masked image SHA256 groups.
- All 13 classes meet 4-shot count requirements.
- Five pre-frozen 4-normal-image supports per SEMANTIC CATEGORY, 13x5x4=260 reference entries; checked no support/test image SHA overlap.
- No Frozen, Plain, Full, or Group-Robust target model inference took place.

## Why 5,848 / 4,287 / 4,047?

- arXiv:2410.00713 v7 abstract and narrative: 5,848.
- The SAME paper Table 1 and Table 2: 4,287 (1,224 normal + 3,063 anomalous).
- Both actual download archives: 4,047 (1,224 normal + 2,823 anomalous).
- Compared with the paper category table, 240 fewer anomalous images in the real release: binderclip -204, cup2 -26, spraybottle -10; remaining ten category totals agree. Some defect-type distributions may vary even when totals match.
- Operationally, exactly which ZIP to use and its contents are now known. The historical reason for the two published counts is not proven and should not be invented; seek author clarification only if explaining version history in a manuscript.

## Scientific implications

Choose BEFORE looking at new target model scores:
- Primary cohort: nonmasked real-background RGB, same official train/test splits.
- Secondary pre-declared background-suppression sensitivity cohort: masked RGB, same physical images and 2,823 ground-truth masks.
- Each target semantic category has exactly four normal references per draw (5 predetermined draws), not four per instance. All four detectors share them.
- The two variants are the SAME 13 semantic classes, NOT 26 independent test classes.
- Do not claim physical-object independence: multiple views belong to a single object.
- Primary hypothesis Full50 vs Plain50, secondary Plain50 vs Frozen and Full50 vs Group-Robust50. Pre-lock statistic, meaningful effect threshold, harms, mask resizing and data qualification.
- The source-supervised adaptation methods have a different training budget from RAD paper's normal-only baselines. Do not compare numbers as if supervision were equal.

## Reproducible logs

Real mask ZIP audit: https://github.com/AlexMartinDev01/freshman-anomaly-research/actions/runs/38046290783

Two ZIP image-key identity: https://github.com/AlexMartinDev01/freshman-anomaly-research/actions/runs/38046515190

84 sampled RGB-pixel comparisons and all 2,823 masks SHA256 comparison: https://github.com/AlexMartinDev01/freshman-anomaly-research/actions/runs/38046715701

Original sources: https://arxiv.org/pdf/2410.00713 ; https://github.com/kaichen-z/RAD ; https://huggingface.co/datasets/zhouk777/RAD

NO original images or masks were committed to the public repository. No new anomaly detection model run has been authorized or performed.
