param(
    [int]$Port = 3000,
    [int]$AdapterPort = 8090,
    [string]$AdapterPython = "",
    [string]$FederatedBackend = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ExperimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DashboardOrigin = "http://127.0.0.1:$Port"
$ApiBase = "http://127.0.0.1:$AdapterPort/api/gui/v1"

if (-not $AdapterPython) {
    $AdapterPython = Join-Path $ExperimentRoot ".venv-gui\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $AdapterPython -PathType Leaf)) {
    throw @"
The GUI adapter environment was not found at:
$AdapterPython

Create it once from ${ExperimentRoot}:
  py -3.12 -m venv .venv-gui
  .\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
"@
}

$AdapterVersion = (& $AdapterPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run the configured adapter Python: $AdapterPython"
}
if (@("3.11", "3.12") -notcontains $AdapterVersion) {
    throw @"
The GUI adapter is using Python $AdapterVersion, but this repository pins torch==2.5.1.
Use Python 3.12 (recommended on this machine) or Python 3.11.

Recreate the project-level environment from ${ExperimentRoot}:
  py -3.12 -m venv .venv-gui
  .\.venv-gui\Scripts\python.exe -m pip install -r .\gui_integration\requirements.txt
"@
}

$AdapterArguments = @(
    (Join-Path $PSScriptRoot "live_adapter.py"),
    "--host", "127.0.0.1",
    "--port", $AdapterPort,
    "--allowed-origins", $DashboardOrigin
)
if ($FederatedBackend) {
    $AdapterArguments += @("--federated-backend", $FederatedBackend)
}

Write-Host "Starting the frozen-model GUI adapter..." -ForegroundColor Cyan
Write-Host "Loading the verified model and 24 demo flows; this normally takes a few seconds." -ForegroundColor DarkGray
$AdapterProcess = Start-Process `
    -FilePath $AdapterPython `
    -ArgumentList $AdapterArguments `
    -WorkingDirectory $ExperimentRoot `
    -WindowStyle Hidden `
    -PassThru

try {
    $StartedWaiting = Get-Date
    $NextProgress = $StartedWaiting.AddSeconds(5)
    $Deadline = (Get-Date).AddSeconds(60)
    $Ready = $false
    while ((Get-Date) -lt $Deadline) {
        if ($AdapterProcess.HasExited) {
            throw "The adapter exited before becoming ready. Run the documented backend command in a visible terminal to inspect its error."
        }
        try {
            $Health = Invoke-RestMethod -Uri "$ApiBase/health" -TimeoutSec 3
            if ($Health.ok -eq $true -and $Health.model_available -eq $true) {
                $Ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 750
        }
        if ((Get-Date) -ge $NextProgress) {
            $Elapsed = [math]::Floor(((Get-Date) - $StartedWaiting).TotalSeconds)
            Write-Host "Still loading safely... ${Elapsed}s" -ForegroundColor DarkGray
            $NextProgress = (Get-Date).AddSeconds(5)
        }
    }
    if (-not $Ready) {
        throw "The adapter did not finish loading the frozen model within 60 seconds."
    }

    Write-Host "Adapter and frozen model ready." -ForegroundColor Green
    Write-Host "Dashboard: $DashboardOrigin" -ForegroundColor Green
    Write-Host "Leave this PowerShell window open while presenting; waiting here is normal." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the dashboard and adapter." -ForegroundColor DarkGray
    $DashboardArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "start.ps1"),
        "-Port", $Port,
        "-ApiBase", $ApiBase
    )
    if ($NoBrowser) { $DashboardArguments += "-NoBrowser" }
    & powershell.exe @DashboardArguments
}
finally {
    if ($AdapterProcess -and -not $AdapterProcess.HasExited) {
        Stop-Process -Id $AdapterProcess.Id -ErrorAction SilentlyContinue
    }
}
