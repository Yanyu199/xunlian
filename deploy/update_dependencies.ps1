param(
  [string]$ProjectDir = "H:\codexdata\xunlian",
  [string]$PythonExe = "H:\codexdata\envs\tem\python.exe"
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectDir

Write-Host "Updating backend dependencies..."
& $PythonExe -m pip install -r requirements.txt --cache-dir H:\codexdata\pip-cache

Write-Host ""
Write-Host "Checking backend syntax..."
& $PythonExe -m compileall app.py core

Write-Host ""
Write-Host "Updating frontend dependencies and building..."
Set-Location (Join-Path $ProjectDir "frontend")
npm install
npm run build

Write-Host ""
Write-Host "Update check completed."
