param([switch]$RemoveRuntime)

$ErrorActionPreference = "Stop"
$Phase4Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $Phase4Root "docker-compose.yml"
$command = Get-Command docker -ErrorAction SilentlyContinue
if ($command) {
    $Docker = $command.Source
} else {
    $Docker = "C:\Users\kingn\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe"
}
if (-not (Test-Path -LiteralPath $Docker)) { throw "Docker CLI was not found." }

& $Docker compose -f $ComposeFile down --remove-orphans

if ($RemoveRuntime) {
    $RuntimeDir = Join-Path $Phase4Root "runtime"
    $ResolvedPhase4 = (Resolve-Path -LiteralPath $Phase4Root).Path
    if (Test-Path -LiteralPath $RuntimeDir) {
        $ResolvedRuntime = (Resolve-Path -LiteralPath $RuntimeDir).Path
        if (-not $ResolvedRuntime.StartsWith($ResolvedPhase4 + [IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to remove a runtime directory outside the Phase 4 workspace."
        }
        Remove-Item -LiteralPath $ResolvedRuntime -Recurse -Force
        Write-Host "Removed generated runtime evidence: $ResolvedRuntime"
    }
}
