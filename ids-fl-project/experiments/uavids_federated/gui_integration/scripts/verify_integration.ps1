param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$HandoffRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $HandoffRoot
try {
    & $Python .\verify_integration.py
    if ($LASTEXITCODE -ne 0) { throw "GUI integration verification failed." }
}
finally {
    Pop-Location
}
