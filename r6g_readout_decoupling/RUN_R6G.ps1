$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman

Write-Host "===================================================="
Write-Host "R6-G FINAL ROOT-CAUSE GATE"
Write-Host "12-layer Normal-1NN vs Supervised-Oracle Decoupling"
Write-Host "===================================================="

Write-Host "`n[1/4] Frozen dependency / parity / fold / support precheck"
python r6g_readout_decoupling\r6g_precheck.py

Write-Host "`n[2/4] 12-layer experiment (two dense DINO passes per object)"
python r6g_readout_decoupling\r6g_run.py

Write-Host "`n[3/4] Frozen gate analysis"
python r6g_readout_decoupling\analyze_r6g.py

Write-Host "`n[4/4] Human-readable report"
python r6g_readout_decoupling\make_report.py

Write-Host "`nDONE. R6 root-cause stage is complete after this gate."
Write-Host "Upload the full folder:"
Write-Host "E:\work\freshman\results\model_v0\metrics\r6g_readout_decoupling\"
