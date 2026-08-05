@echo off
setlocal

REM Start the local OneTrainer autoresearch dashboard.
cd /d "%~dp0"

set "AUTORESEARCH_DIR=%~dp0training_presets\autoresearch"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not defined AUTORESEARCH_HOST set "AUTORESEARCH_HOST=127.0.0.1"
if not defined AUTORESEARCH_PORT set "AUTORESEARCH_PORT=8765"
set "AUTORESEARCH_URL=http://%AUTORESEARCH_HOST%:%AUTORESEARCH_PORT%/"

if not exist "%PYTHON_EXE%" (
    echo Error: Python executable not found at "%PYTHON_EXE%".
    echo Run install.bat first.
    goto :end
)

if not exist "%AUTORESEARCH_DIR%\dashboard.py" (
    echo Error: dashboard.py not found at "%AUTORESEARCH_DIR%\dashboard.py".
    goto :end
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%AUTORESEARCH_URL%' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >NUL 2>NUL
if not errorlevel 1 (
    echo OneTrainer autoresearch dashboard is already running:
    echo %AUTORESEARCH_URL%
    start "" "%AUTORESEARCH_URL%"
    goto :end
)

echo Starting OneTrainer autoresearch dashboard:
echo %AUTORESEARCH_URL%
echo.
echo Close this window to stop the dashboard.
echo.

powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%AUTORESEARCH_URL%'" >NUL 2>NUL
cd /d "%AUTORESEARCH_DIR%"
"%PYTHON_EXE%" dashboard.py --host "%AUTORESEARCH_HOST%" --port "%AUTORESEARCH_PORT%"
if errorlevel 1 (
    echo.
    echo Error: dashboard server exited with code %ERRORLEVEL%.
)

:end
pause
