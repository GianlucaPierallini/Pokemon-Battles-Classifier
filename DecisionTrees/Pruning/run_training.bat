@echo off
setlocal
set "HERE=%~dp0"
set "PYTHON=%HERE%..\..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: Python environment missing: "%PYTHON%"
    echo From Challenge1 project root: py -3.11 -m venv .venv
    exit /b 1
)
"%PYTHON%" "%HERE%run_pruning.py" %*
exit /b %ERRORLEVEL%
