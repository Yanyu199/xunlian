param(
  [string]$FrontendDir = "H:\codexdata\xunlian\frontend"
)

$ErrorActionPreference = "Stop"

Set-Location $FrontendDir

if (!(Test-Path "node_modules")) {
  Write-Host "node_modules not found. Running npm install..."
  npm install
}

Write-Host "Starting TEM frontend..."
Write-Host "URL: http://127.0.0.1:5173"
npm run dev
