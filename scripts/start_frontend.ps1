$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "apps\frontend"

if (-not $env:VITE_API_BASE) {
  $env:VITE_API_BASE = "http://127.0.0.1:8000"
}
if (-not $env:VITE_WS_BASE) {
  $env:VITE_WS_BASE = "ws://127.0.0.1:8000"
}

Push-Location $FrontendRoot
try {
  if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    npm install
  }
  npx vite --host 127.0.0.1 --port 5173
}
finally {
  Pop-Location
}
