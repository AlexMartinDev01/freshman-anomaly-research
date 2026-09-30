$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
New-Item -ItemType Directory -Force results\model_v0\metrics | Out-Null

# One MVTec hard-ish/control pair. Then add one VisA object from the preflight list if desired.
python tsn_formal_runner\tsn_formal_confirm_v2.py `
  --objects bottle,cable `
  --shots 2 `
  --splits 1 `
  --out results\model_v0\metrics\tsn_smoke.csv `
  --smoke

python tsn_formal_runner\check_tsn_results.py results\model_v0\metrics\tsn_smoke.csv
