param(
    [int]$Port = $(if ($env:GUI_PORT) { [int]$env:GUI_PORT } else { 3000 }),
    [string]$ApiBase = $(if ($env:GUI_API_BASE) { $env:GUI_API_BASE } else { "http://127.0.0.1:8090/api/gui/v1" }),
    [int]$RequestTimeoutMs = $(if ($env:GUI_REQUEST_TIMEOUT_MS) { [int]$env:GUI_REQUEST_TIMEOUT_MS } else { 4500 }),
    [int]$ConnectTimeoutMs = $(if ($env:GUI_CONNECT_TIMEOUT_MS) { [int]$env:GUI_CONNECT_TIMEOUT_MS } else { 20000 }),
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$server = Join-Path $PSScriptRoot "serve.py"
$arguments = @(
    $server,
    "--port", $Port,
    "--api-base", $ApiBase,
    "--request-timeout-ms", $RequestTimeoutMs,
    "--connect-timeout-ms", $ConnectTimeoutMs
)
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
