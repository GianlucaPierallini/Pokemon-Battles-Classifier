@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found. Run: py -3.11 -m venv .venv
  exit /b 1
)
".venv\Scripts\python.exe" "run_knn.py" --data-dir "data" --feature-set summary --backend auto --folds 5 --k-max 51 --scaling numeric %*
exit /b %errorlevel%
