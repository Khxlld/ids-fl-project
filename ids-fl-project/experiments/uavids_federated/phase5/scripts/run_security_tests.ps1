param([switch]$SkipBuild)

$ErrorActionPreference = "Stop"
$Phase5Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $Phase5Root "docker-compose.yml"
$command = Get-Command docker -ErrorAction SilentlyContinue
$Docker = if ($command) { $command.Source } else { Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe" }
$env:Path = (Split-Path -Parent $Docker) + ";" + $env:Path

if (-not $SkipBuild) {
    & $Docker compose -f $ComposeFile build --pull
    if ($LASTEXITCODE -ne 0) { throw "Secure image build failed." }
}
& $Docker run --rm uavids-phase5-secure-demo:local `
    python -m pytest -q -p no:cacheprovider /app/tests /app/phase4_tests /app/phase5_tests
if ($LASTEXITCODE -ne 0) { throw "Containerized Phase 3/4/5 tests failed." }
