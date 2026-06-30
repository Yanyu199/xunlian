param(
  [string]$ProjectDir = "H:\codexdata\xunlian",
  [string]$PythonExe = "H:\codexdata\envs\tem\python.exe",
  [string]$CodexDataDir = "H:\codexdata",
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

$env:TEM_CODEXDATA_DIR = $CodexDataDir
$env:TEM_OUTPUT_DIR = Join-Path $ProjectDir "output"
$env:TEM_DATA_DIR = Join-Path $ProjectDir "data"
$env:TEM_TRAINING_JOBS_DIR = Join-Path $CodexDataDir "training_jobs"
$env:TEM_DEVICE = "auto"

Set-Location $ProjectDir

Write-Host "Starting TEM backend..."
Write-Host "API: http://127.0.0.1:$Port/docs"
Write-Host "LAN: http://<server-ip>:$Port/docs"

& $PythonExe -m uvicorn app:app --host $HostAddress --port $Port
