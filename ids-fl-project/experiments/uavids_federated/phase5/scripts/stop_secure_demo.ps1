param([switch]$RemoveRuntime)

$ErrorActionPreference = "Stop"
$Phase5Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $Phase5Root "docker-compose.yml"
$command = Get-Command docker -ErrorAction SilentlyContinue
$Docker = if ($command) { $command.Source } else { Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe" }
$env:Path = (Split-Path -Parent $Docker) + ";" + $env:Path
& $Docker compose -f $ComposeFile down --remove-orphans
if ($LASTEXITCODE -ne 0) { throw "Unable to stop the secure Compose deployment." }
if ($RemoveRuntime) {
    $Runtime = (Join-Path $Phase5Root "runtime")
    if ((Test-Path -LiteralPath $Runtime) -and ((Resolve-Path $Runtime).Path -eq (Join-Path $Phase5Root "runtime"))) {
        Remove-Item -LiteralPath $Runtime -Recurse -Force
    }
}
