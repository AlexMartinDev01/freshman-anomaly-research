$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
Write-Host "=== R5 MAIN: FROZEN RAW-MAP CAUSAL INTERVENTIONS ==="

Write-Host "`n[1/4] Unit tests"
python r5_rawmap_causal_interventions\r5_unit_tests.py

Write-Host "`n[2/4] Precheck"
python r5_rawmap_causal_interventions\r5_precheck.py

Write-Host "`n[3/4] Run MAIN 81 cells"
python r5_rawmap_causal_interventions\r5_run_rawmap_interventions.py `
  --selection r5_rawmap_causal_interventions\R5_MAIN_SELECTION.csv `
  --outdir results\model_v0\metrics\r5_rawmap `
  --shuffle-reps 100 `
  --resume

Write-Host "`n[4/4] Analyze with frozen gates"
python r5_rawmap_causal_interventions\r5_analyze_main.py `
  --indir results\model_v0\metrics\r5_rawmap `
  --selection r5_rawmap_causal_interventions\R5_MAIN_SELECTION.csv

Write-Host "`nMAIN COMPLETE. DO NOT RUN LOCKED REPLICATION YET."
