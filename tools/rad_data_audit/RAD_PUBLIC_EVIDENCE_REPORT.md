# RAD source evidence and the exact verification boundary (2026-10-10)

## Verified facts from public sources

1. arXiv:2410.00713v7 abstract and Section 3.2 state **5,848** RGB images. Its Table 1 states **1,224 normal + 3,063 anomalous = 4,287**, and the 13 category totals in Table 2 also sum to **4,287**. The current project page likewise states 4,287. The cause of this manuscript-internal conflict **is not resolved**. Do not assign each number to one download archive without checking.
2. The official code defines **13 semantic classes across 18 instance folders**: `src/rad_benchmark/constants.py`.
3. The official dataset validator uses `train/good`, `test/good`, `test/<defect>`, `ground_truth/<defect>` and resolves masks as either `<stem>_mask.png` or `<stem>.png`.
4. HF publishes masked archive SHA256 **ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee**. Masked ZIP website size: 211MB; unmasked: ~5.85GB.
5. Public HF dataset metadata lists `license:other`; GitHub software MIT does **not** automatically authorize dataset redistribution. The researcher has the author's permission for non-commercial use, but should keep original assets out of public Git history.

Sources: https://arxiv.org/pdf/2410.00713 ; https://chang-xinhai.github.io/rad-website/ ; https://github.com/kaichen-z/RAD ; https://huggingface.co/datasets/zhouk777/RAD/blob/main/Anomaly_refine_msk.zip

## Verified operations / still blocked

- A reproducible dataset downloader and file auditor were committed to this isolated branch.
- Synthetic ZIP smoke tests of the corresponding auditor verified 18 instance folders, 13 semantic classes, mask matching, count logic, and five 4-shot reference draws (synthetic data only).
- Attempts to retrieve the 211MB release archive from the current ChatGPT container failed. Therefore **no RAD image data, real per-instance counts, real mask validity, or correspondence between the two releases have been verified here**.
- Do not run the four detectors yet; first execute the downloader and auditor in a machine with HF download access, inspect generated metadata, and lock the target evaluation cohort and statistical decision rule.
- Record the actual ZIP SHA, category- and instance-level train/test counts, image sizes, missing/empty masks, duplicate image hashes, support/query identity overlap.
- A total of four references per **semantic category** across all instances is a proposed 4-shot protocol, not the original full-normal-sample baseline. Different viewpoints of the same physical object are not independent manufactured items.

## Where to get the executable scripts

`tools/rad_data_audit/rad_audit.py`, `requirements.txt`, and `README.md` in this branch.
