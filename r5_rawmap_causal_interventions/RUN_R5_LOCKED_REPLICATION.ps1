$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
Write-Host "LOCKED REPLICATION — run only after MAIN verdict is frozen."
python r5_rawmap_causal_interventions\r5_run_rawmap_interventions.py `
  --selection r5_rawmap_causal_interventions\R5_LOCKED_REPLICATION_SELECTION.csv `
  --outdir results\model_v0\metrics\r5_rawmap_replication `
  --shuffle-reps 100 `
  --resume
