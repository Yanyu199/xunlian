param(
  [string]$ProjectDir = "H:\codexdata\xunlian",
  [string]$PythonExe = "H:\codexdata\envs\tem\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "== TEM deployment environment check =="
Write-Host "ProjectDir: $ProjectDir"
Write-Host "PythonExe : $PythonExe"

if (!(Test-Path $ProjectDir)) {
  throw "Project directory not found: $ProjectDir"
}

if (!(Test-Path $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

Write-Host ""
Write-Host "Python:"
& $PythonExe --version

Write-Host ""
Write-Host "Python executable:"
& $PythonExe -c "import sys; print(sys.executable)"

Write-Host ""
Write-Host "Backend dependencies:"
& $PythonExe -c "import fastapi, uvicorn, numpy, torch; print('fastapi ok'); print('uvicorn ok'); print('numpy ok'); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host ""
Write-Host "NVIDIA driver:"
try {
  nvidia-smi
} catch {
  Write-Warning "nvidia-smi is unavailable. CPU mode can still run, but GPU training will not be available."
}

Write-Host ""
Write-Host "Node:"
node --version
npm --version

Write-Host ""
Write-Host "Check completed."
