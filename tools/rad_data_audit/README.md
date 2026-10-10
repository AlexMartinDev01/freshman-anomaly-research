# RAD data: verified upstream facts and reproducible file-level audit

This directory contains a read-only **dataset acquisition and audit tool**, not the RAD dataset itself. **Do not commit the original RAD archives or image files into a public GitHub repository.** The authors' software MIT license does not grant an independent image redistribution license. The dataset author has separately confirmed that non-commercial use is acceptable for this research; preserve that evidence privately.

## How to run on your Windows machine

From the repository root, with Python 3.10+:

```powershell
python -m pip install -r tools/rad_data_audit/requirements.txt
python tools/rad_data_audit/rad_audit.py download --root "E:\work\freshman\datasets\RAD_source"
python tools/rad_data_audit/rad_audit.py audit --archive "E:\work\freshman\datasets\RAD_source\Anomaly_refine_msk.zip" --out "E:\work\freshman\research_breakthrough\ASU_ALGORITHM_UNIFIED\RAD_DATA_AUDIT"
```

The `download` step validates the official archive SHA256 (public masked archive, 211MB): `ba00938286fd511c1c7cec78586435015fe637aa8de3198be583ceae2155fcee`. If HF updates the archive, **stop** rather than bypassing the mismatch.

The `audit` step inspects ZIP CRC, all 13 semantic categories and 18 physical-instance directories, actual split/defect image counts, dimensions, image hashes, masks (missing, all-zero, size mismatch), file duplication, and emits five deterministic 4-shot reference sets per *semantic category*, with an image budget of exactly four. Model weights are never loaded. No anomaly inference or model selection is performed.

Outputs (outside the Git repository): `RAD_ACTUAL_DATA_MANIFEST.csv`, `RAD_COUNTS.csv`, `RAD_ELIGIBILITY.csv`, `RAD_4SHOT_SUPPORT_FROZEN.csv`, `RAD_FILE_AUDIT.json`.

**After running:** inspect validation anomalies and official mask pairing before locking the confirmation protocol; use exactly the same support sets and query images across Frozen, Plain50, Full50, and Group-Robust50. Do not tune or rerank by RAD test labels.

## Already established, not an image-file audit result

- arXiv:2410.00713v7 abstract and Sec 3.2: 5,848 images. The same v7 Table 1 lists 1,224 normal + 3,063 anomalous = 4,287; Table 2 per-class rows also total 4,287. Underlying reason **unresolved**.
- Official source `src/rad_benchmark/constants.py` contains the 18-directory to 13-category mapping, reproduced in the script.
- Official source `src/rad_benchmark/dataset.py` uses `train/good`, `test/good`, `test/<defect>`, and `ground_truth/<defect>`, with masks named `<stem>_mask.png` or `<stem>.png`.
- 201MB/211MB 'masked' vs 5.85GB 'unmasked' archive identity/sample correspondence still requires actual byte-level verification; these scripts inspect **only the masked version**.
- RAD uses many viewpoints of the same physical objects. Distinct image files cannot be claimed as independent manufactured units. Pre-register the evaluation unit and avoid inflating 13 semantic classes into 18 independent tasks.

Sources:
- https://arxiv.org/pdf/2410.00713
- https://github.com/kaichen-z/RAD
- https://huggingface.co/datasets/zhouk777/RAD
- https://huggingface.co/datasets/zhouk777/RAD/blob/main/Anomaly_refine_msk.zip
