$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Here '..\..\.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Error "Python environment not found: $Python. Create it with: py -3.11 -m venv .venv"
    exit 1
}
& $Python (Join-Path $Here 'run_pruning.py') @args
exit $LASTEXITCODE
