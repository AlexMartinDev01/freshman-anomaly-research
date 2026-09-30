# RC Final Direct Gate execution

This script requires the real local R0 caches referenced by `phase_m1_rescue.load_obj()`.
GitHub-hosted runners do not contain those caches, so do NOT substitute summary CSVs.

Frozen science:
- `RC_FINAL_DIRECT_PREREG_FROZEN.md`

Run only after the currently running R6-G job has finished, to avoid GPU/RAM contention.

From the project root on the research machine:

```powershell
cd E:\work\freshman
git fetch origin rc-rootcause-directional
git checkout rc-rootcause-directional
python rootcause_rc\run_rc_final_direct.py --mode all
```

Outputs:
`results\model_v0\metrics\rc_final_direct\`

Mandatory return files:
- RC_COV2_curve.csv
- RC_COV2_duplicate_control.csv
- RC_COV2_summary.csv
- RC_COV2_VERDICT.csv
- RC_DIR_summary.csv
- RC_DIR_VERDICT.csv
- RC_DIR_pairs_screw.csv
- RC_FINAL_VERDICT.csv

Do not alter thresholds, matching calipers, object lists, support seeds, or classifier hyperparameters after execution starts.
