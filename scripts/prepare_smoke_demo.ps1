$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
  python -m ml.scripts.build_dataset --config configs/train.smoke.yaml
  python -m ml.scripts.train --config configs/train.smoke.yaml --model tab_transformer
  python -m ml.scripts.package_model --config configs/train.smoke.yaml --model tab_transformer
}
finally {
  Pop-Location
}
