@echo off
chcp 65001 > nul
setlocal

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if defined TEM_PYTHON_EXE (
  set "PYTHON_EXE=%TEM_PYTHON_EXE%"
) else if exist "H:\codexdata\envs\tem\python.exe" (
  set "PYTHON_EXE=H:\codexdata\envs\tem\python.exe"
) else (
  set "PYTHON_EXE=python"
)

if defined TEM_CODEXDATA_DIR (
  set "CODEXDATA_DIR=%TEM_CODEXDATA_DIR%"
) else (
  for %%I in ("%PROJECT_DIR%\..") do set "CODEXDATA_DIR=%%~fI"
)

set "TEM_CODEXDATA_DIR=%CODEXDATA_DIR%"
set "TEM_OUTPUT_DIR=%PROJECT_DIR%\output"
set "TEM_DATA_DIR=%PROJECT_DIR%\data"
set "TEM_TRAINING_JOBS_DIR=%CODEXDATA_DIR%\training_jobs"
set "TEM_DEVICE=auto"
set "TEM_FORWARD_BACKEND=auto"

set "HOST=0.0.0.0"
set "PORT=8000"

echo ========================================================
echo     孔中瞬变电磁工程反演系统 - 后端启动
echo ========================================================
echo 项目目录: %PROJECT_DIR%
echo Python  : %PYTHON_EXE%
echo 输出目录: %TEM_OUTPUT_DIR%
echo 任务目录: %TEM_TRAINING_JOBS_DIR%
echo.

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
  echo 无法进入项目目录。
  pause
  exit /b 1
)

"%PYTHON_EXE%" -c "import fastapi, uvicorn, numpy, torch; print('依赖检查通过'); print('CUDA 可用:', torch.cuda.is_available())"
if errorlevel 1 (
  echo.
  echo 后端依赖检查失败。请确认 Python 环境已安装 requirements.txt 中的依赖。
  pause
  exit /b 1
)

echo.
echo 后端 API: http://127.0.0.1:%PORT%/docs
echo 局域网访问: http://服务器IP:%PORT%/docs
echo 按 Ctrl+C 可停止后端。
echo.

"%PYTHON_EXE%" -m uvicorn app:app --host %HOST% --port %PORT%

echo.
echo 后端已停止。
pause
