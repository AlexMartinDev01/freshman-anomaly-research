$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "=== R5-C LOCKED REPLICATION AMENDMENT ==="
Write-Host "All 27 objects | shots 2 and 8 | split 0 | spatial params unchanged"

python r5_rawmap_causal_interventions\r5_run_rawmap_interventions.py `
  --selection r5c_replication_amendment\R5C_REPLICATION_SELECTION.csv `
  --outdir results\model_v0\metrics\r5c_replication `
  --shuffle-reps 100 `
  --resume

Write-Host "`nDONE."
Write-Host "Upload the whole folder:"
Write-Host "results\model_v0\metrics\r5c_replication\"
