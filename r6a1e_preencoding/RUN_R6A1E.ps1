$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "=================================================="
Write-Host "R6-A1E Pre-Encoding Counterfactual Defect Removal"
Write-Host "=================================================="

Write-Host "`n[1/3] Mandatory parity + path/donor/pair precheck"
python r6a1e_preencoding\r6a1e_precheck.py

Write-Host "`n[2/3] Counterfactual main experiment"
python r6a1e_preencoding\r6a1e_preencoding_counterfactual.py

Write-Host "`n[3/3] Frozen analysis"
python r6a1e_preencoding\analyze_r6a1e.py `
  --indir results\model_v0\metrics\r6a1e_preencoding

Write-Host "`nDONE."
Write-Host "Upload the entire directory:"
Write-Host "E:\work\freshman\results\model_v0\metrics\r6a1e_preencoding\"
