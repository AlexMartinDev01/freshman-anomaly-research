$ErrorActionPreference = "Stop"
Set-Location E:\work\freshman
Write-Host "=== Git state ==="
git rev-parse HEAD
git status --short
Write-Host "=== GPU ==="
nvidia-smi
Write-Host "=== Python ==="
python --version
python -c "import torch, sklearn, numpy, pandas; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python tsn_formal_runner\test_tsn_math.py
python tsn_formal_runner\tsn_preflight.py
