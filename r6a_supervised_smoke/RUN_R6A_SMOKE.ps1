$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "=== R6-A SUPERVISED UPPER-BOUND SMOKE ==="
python r6a_supervised_smoke\r6a_precheck.py

python r6a_supervised_smoke\r6a_smoke.py `
  --selection r6a_supervised_smoke\R6A_SMOKE_SELECTION.csv `
  --outdir results\model_v0\metrics\r6a_smoke

python r6a_supervised_smoke\r6a_analyze.py `
  --input results\model_v0\metrics\r6a_smoke\summary.csv

Write-Host "`nDONE. Upload the whole folder:"
Write-Host "results\model_v0\metrics\r6a_smoke\"
