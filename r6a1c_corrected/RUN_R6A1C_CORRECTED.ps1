$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
Write-Host "=== R6-A1C CORRECTED AUDIT ==="
python r6a1c_corrected\r6a1c_corrected.py `
  --selection r6a_supervised_smoke\R6A_SMOKE_SELECTION.csv `
  --outdir results\model_v0\metrics\r6a1c_corrected

python r6a1c_corrected\analyze_corrected.py `
  --indir results\model_v0\metrics\r6a1c_corrected

Write-Host "`nUpload the full folder:"
Write-Host "results\model_v0\metrics\r6a1c_corrected\"
