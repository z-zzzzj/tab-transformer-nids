$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendRoot = Join-Path $ProjectRoot "apps\backend"
$PythonPath = @(
  (Join-Path $ProjectRoot "src"),
  $ProjectRoot,
  $BackendRoot
) -join ";"

if (-not $env:PYTHONPATH) {
  $env:PYTHONPATH = $PythonPath
} else {
  $env:PYTHONPATH = "$PythonPath;$env:PYTHONPATH"
}

$ArtifactDir = Join-Path $ProjectRoot "artifacts\current"
if (-not (Test-Path $ArtifactDir)) {
  $SmokeArtifactDir = Join-Path $ProjectRoot "artifacts_smoke\current"
  if (Test-Path $SmokeArtifactDir) {
    $ArtifactDir = $SmokeArtifactDir
  }
}
$env:NIDS_ARTIFACT_DIR = $ArtifactDir

$ReplayPickle = Join-Path $ProjectRoot "ml\data\processed\test.pkl.gz"
if (-not (Test-Path $ReplayPickle)) {
  $SmokeReplayPickle = Join-Path $ProjectRoot "ml\data\processed_smoke\test.pkl.gz"
  if (Test-Path $SmokeReplayPickle) {
    $ReplayPickle = $SmokeReplayPickle
  }
}
if (Test-Path $ReplayPickle) {
  $env:NIDS_REPLAY_PICKLE = $ReplayPickle
}

Push-Location $BackendRoot
try {
  python -m daphne -b 127.0.0.1 -p 8000 backend_project.asgi:application
}
finally {
  Pop-Location
}
