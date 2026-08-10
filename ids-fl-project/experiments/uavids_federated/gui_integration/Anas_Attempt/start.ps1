param(
    [int]$Port = $(if ($env:GUI_PORT) { [int]$env:GUI_PORT } else { 3000 }),
    [string]$ApiBase = $(if ($env:GUI_API_BASE) { $env:GUI_API_BASE } else { "http://127.0.0.1:8090/api/gui/v1" }),
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$server = Join-Path $PSScriptRoot "serve.py"
$arguments = @($server, "--port", $Port, "--api-base", $ApiBase)
if (-not $NoBrowser) { $arguments += "--open" }

$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($launcher) {
    & $launcher.Source -3 @arguments
    exit $LASTEXITCODE
}

$python = Get-Command python.exe -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3 was not found. Install Python 3.10 or newer, then run this command again."
}

& $python.Source @arguments
exit $LASTEXITCODE
