$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "=== R6-A1D DEFECT-EXCISION / FAR-FIELD SIGNAL AUDIT ==="

python r6a1d_defect_excision\r6a1d_precheck.py

python r6a1d_defect_excision\r6a1d_defect_excision.py `
  --selection r6a1d_defect_excision\R6A1D_SELECTION.csv `
  --outdir results\model_v0\metrics\r6a1d_defect_excision

python r6a1d_defect_excision\analyze_r6a1d.py `
  --indir results\model_v0\metrics\r6a1d_defect_excision

Write-Host "`nDONE. Upload the full folder:"
Write-Host "results\model_v0\metrics\r6a1d_defect_excision\"
