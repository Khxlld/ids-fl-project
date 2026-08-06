param(
    [string]$FrontendOrigin = "http://localhost:3000",
    [string]$FederatedBackend = "",
    [int]$Port = 8090,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$HandoffRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Arguments = @(
    "-m", "gui_integration.backend",
    "--host", "127.0.0.1",
    "--port", $Port,
    "--allowed-origins", $FrontendOrigin
)
if ($FederatedBackend) {
    $Arguments += @("--federated-backend", $FederatedBackend)
}
Push-Location (Split-Path -Parent $HandoffRoot)
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "GUI backend exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
