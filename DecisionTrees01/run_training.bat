@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Missing project venv: "%PYTHON%"
    echo From project root: py -3.11 -m venv .venv
    echo Then: .\.venv\Scripts\python.exe -m pip install -r .\DecisionTree\requirements.txt
    exit /b 1
)
"%PYTHON%" "%SCRIPT_DIR%run_decision_tree.py" %*
exit /b %ERRORLEVEL%
