param(
    [string]$DataDir = ".\data",
    [string]$FeatureSet = "summary",
    [string]$Backend = "auto",
    [int]$Folds = 5,
    [int]$KMax = 51,
    [string]$Scaling = "numeric",
    [string]$OutputDir = ".\results"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Manca .venv. Prima esegui: py -3.11 -m venv .venv" -ForegroundColor Red
    exit 1
}
& $Python .\run_knn.py --data-dir $DataDir --output-dir $OutputDir `
    --feature-set $FeatureSet --backend $Backend --folds $Folds --k-max $KMax --scaling $Scaling
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
