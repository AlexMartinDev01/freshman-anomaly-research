$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "========================================"
Write-Host "TSN ROOT-CAUSE CAUSAL AUDIT V2"
Write-Host "========================================"

New-Item -ItemType Directory -Force results\model_v0\metrics\tsn_failure_audit_v2 | Out-Null

Write-Host "`n[1/3] PRECHECK"
python tsn_rootcause_local_kit_v2\precheck_rootcause.py

Write-Host "`n[2/3] CAUSAL AUDIT"
python tsn_rootcause_local_kit_v2\tsn_failure_causal_audit.py `
  --formal results\model_v0\metrics\tsn_formal_v2.csv `
  --selection tsn_rootcause_local_kit_v2\11_causal_selection_v2.csv `
  --outdir results\model_v0\metrics\tsn_failure_audit_v2 `
  --shuffle-reps 200 `
  --resume

Write-Host "`n[3/3] ANALYZE"
python tsn_rootcause_local_kit_v2\analyze_causal_audit.py `
  --input results\model_v0\metrics\tsn_failure_audit_v2\causal_audit_cells.csv

Write-Host "`nDONE."
Write-Host "Please send these three files back:"
Write-Host "  results\model_v0\metrics\tsn_failure_audit_v2\causal_audit_cells.csv"
Write-Host "  results\model_v0\metrics\tsn_failure_audit_v2\causal_audit_per_image.csv"
Write-Host "  results\model_v0\metrics\tsn_failure_audit_v2\causal_audit_bank_level.csv"
