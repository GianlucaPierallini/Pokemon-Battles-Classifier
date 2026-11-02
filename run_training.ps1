# Run from project root: .\DecisionTree\run_training.ps1 --backend cpu --trials 24
# If execution policy blocks .ps1, use .\DecisionTree\run_training.bat instead.
$py = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'run_decision_tree.py'
if (-not (Test-Path $py)) {
  Write-Error 'Missing project virtual environment. Create it with: py -3.11 -m venv .venv'
  exit 1
}
& $py $script @args
exit $LASTEXITCODE
