param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Runs)
$ErrorActionPreference = 'Stop'
$ViewerPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ViewerPython)) { throw 'Run setup.ps1 first.' }
& $ViewerPython -B -m fempost workbench @Runs
exit $LASTEXITCODE
