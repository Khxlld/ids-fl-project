param([switch]$SkipBuild)

$ErrorActionPreference = "Stop"
$run = Join-Path $PSScriptRoot "run_secure_demo.ps1"
if ($SkipBuild) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $run -SkipBuild -AttackTests
} else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $run -AttackTests
}
if ($LASTEXITCODE -ne 0) { throw "Controlled Phase 5 attack exercise failed." }
