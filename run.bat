@echo off
setlocal

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "PYTHON_EXE=H:\codexdata\envs\tem\python.exe"
set "CODEXDATA_DIR=H:\codexdata"
set "HOST_ADDRESS=0.0.0.0"
set "PORT=8001"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found:
  echo         %PYTHON_EXE%
  echo.
  echo Please create/install the backend environment first.
  exit /b 1
)

set "TEM_CODEXDATA_DIR=%CODEXDATA_DIR%"
set "TEM_OUTPUT_DIR=%PROJECT_DIR%\output"
set "TEM_DATA_DIR=%PROJECT_DIR%\data"
set "TEM_TRAINING_JOBS_DIR=%CODEXDATA_DIR%\training_jobs"
set "TEM_DEVICE=auto"

cd /d "%PROJECT_DIR%"

echo Starting TEM backend...
echo Project: %PROJECT_DIR%
echo API:     http://127.0.0.1:%PORT%/docs
echo LAN:     http://^<server-ip^>:%PORT%/docs
echo.

"%PYTHON_EXE%" -m uvicorn app:app --host %HOST_ADDRESS% --port %PORT%

endlocal
