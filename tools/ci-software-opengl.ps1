# Hosted-runner-only graphics setup. Never changes a user's graphics driver.
param([string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'))
$ErrorActionPreference = 'Stop'
if ($env:GITHUB_ACTIONS -ne 'true') { throw 'This setup is only for disposable GitHub Actions runners.' }
$MesaFolder = Join-Path $env:RUNNER_TEMP 'femview-mesa-26.2.4'
New-Item -ItemType Directory -Force -Path $MesaFolder | Out-Null
$Archive = Join-Path $MesaFolder 'mesa.7z'
Invoke-WebRequest -Uri 'https://github.com/pal1000/mesa-dist-win/releases/download/26.2.4/mesa3d-26.2.4-release-msvc.7z' -OutFile $Archive
$Expected = '351fc8c8b695878ffb3eaa044b3ead08672a48b1a045e3c3e3975811df0f6695'
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) { throw 'Mesa archive hash mismatch' }
& 7z x $Archive "-o$MesaFolder" -y | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Mesa extraction failed' }
$InterpreterFolders = & $Python -c 'import sys; from pathlib import Path; print(Path(sys.executable).parent); print(Path(sys._base_executable).parent)'
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the isolated interpreter' }
foreach ($Folder in ($InterpreterFolders | Sort-Object -Unique)) {
    foreach ($Library in @('opengl32.dll', 'libgallium_wgl.dll')) {
        Copy-Item -LiteralPath (Join-Path $MesaFolder "x64/$Library") -Destination (Join-Path $Folder $Library)
    }
    Copy-Item -LiteralPath (Join-Path $MesaFolder 'x64/opengl32.dll') -Destination (Join-Path $Folder 'opengl32sw.dll')
}
'GALLIUM_DRIVER=llvmpipe' | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
'QT_OPENGL=software' | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
