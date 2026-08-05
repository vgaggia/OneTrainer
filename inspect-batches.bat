@echo off
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" "scripts\inspect_batch_order.py"
) else (
    python "scripts\inspect_batch_order.py"
)

if errorlevel 1 pause
