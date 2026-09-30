$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
New-Item -ItemType Directory -Force results\model_v0\metrics | Out-Null

# IMPORTANT: run ONE GPU process only. Do not parallelize cells on an 8 GB Windows GPU.
python tsn_formal_runner\tsn_formal_confirm_v2.py `
  --shots 2,4,8 `
  --splits 3 `
  --out results\model_v0\metrics\tsn_formal_v2.csv `
  --resume 2>&1 | Tee-Object -FilePath results\model_v0\metrics\tsn_formal_v2.log

python tsn_formal_runner\check_tsn_results.py results\model_v0\metrics\tsn_formal_v2.csv
