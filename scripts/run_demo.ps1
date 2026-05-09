$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SmokeModel = Join-Path $ProjectRoot "artifacts_smoke\current\model.pt"

if (-not (Test-Path $SmokeModel)) {
  Write-Host "No packaged smoke model found. Preparing smoke demo artifacts..." -ForegroundColor Cyan
  & (Join-Path $PSScriptRoot "prepare_smoke_demo.ps1")
}

Write-Host "Start backend in one terminal:" -ForegroundColor Green
Write-Host "powershell -ExecutionPolicy Bypass -File $PSScriptRoot\start_backend.ps1"
Write-Host ""
Write-Host "Start frontend in another terminal:" -ForegroundColor Green
Write-Host "powershell -ExecutionPolicy Bypass -File $PSScriptRoot\start_frontend.ps1"
