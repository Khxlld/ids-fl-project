param(
    [switch]$SkipBuild,
    [switch]$AttackTests
)

$ErrorActionPreference = "Stop"
$Phase5Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExperimentRoot = (Resolve-Path (Join-Path $Phase5Root "..")).Path
$ComposeFile = Join-Path $Phase5Root "docker-compose.yml"
$RuntimeDir = Join-Path $Phase5Root "runtime"
$Image = "uavids-phase5-secure-demo:local"

function Find-Docker {
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    throw "Docker CLI was not found. Start Docker Desktop and add its resources/bin directory to PATH."
}

function Find-Python {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { return $launcher.Source }
    throw "Python 3.11 was not found for host-side verification."
}

$Docker = Find-Docker
$Python = Find-Python
$env:Path = (Split-Path -Parent $Docker) + ";" + $env:Path
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

& $Docker compose -f $ComposeFile down --remove-orphans
if ($LASTEXITCODE -ne 0) { throw "Unable to clean the previous secure Compose deployment." }
if (-not $SkipBuild) {
    & $Docker compose -f $ComposeFile build --pull
    if ($LASTEXITCODE -ne 0) { throw "Secure image build failed." }
}

& $Docker run --rm `
    --mount "type=bind,source=$RuntimeDir,target=/runtime" `
    $Image python -m phase5_app.provision `
    --output /runtime/keys `
    --config /app/phase5_config/security_config.json
if ($LASTEXITCODE -ne 0) { throw "Phase 5 demo identity provisioning failed." }

$env:SECURITY_ATTACK_TESTS = if ($AttackTests) { "1" } else { "0" }
$HostRunTimer = [Diagnostics.Stopwatch]::StartNew()
& $Docker compose -f $ComposeFile up -d --remove-orphans
if ($LASTEXITCODE -ne 0) { throw "Secure Compose startup failed." }

$HealthDeadline = (Get-Date).AddMinutes(3)
do {
    try {
        $Status = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/status" -TimeoutSec 3
        break
    } catch {
        if ((Get-Date) -gt $HealthDeadline) { throw "Secure control center did not become healthy within three minutes." }
        Start-Sleep -Milliseconds 500
    }
} while ($true)
$ApiReadySeconds = $HostRunTimer.Elapsed.TotalSeconds

$StatsPath = Join-Path $RuntimeDir "host_container_stats.jsonl"
Set-Content -LiteralPath $StatsPath -Value "" -Encoding UTF8
$LastEventSequence = 0
do {
    $Status = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/status" -TimeoutSec 5
    $Events = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/events?after_seq=$LastEventSequence" -TimeoutSec 5
    foreach ($Event in $Events.events) {
        Write-Host ("[{0}] r{1} {2} {3}" -f $Event.elapsed_ms, $Event.round, $Event.source, $Event.event_type)
        $LastEventSequence = [Math]::Max($LastEventSequence, [int]$Event.seq)
    }
    $RawStats = & $Docker stats --no-stream --format "{{json .}}"
    foreach ($Line in $RawStats) {
        if (-not [string]::IsNullOrWhiteSpace($Line)) {
            $Sample = $Line | ConvertFrom-Json
            $Sample | Add-Member -NotePropertyName sampled_utc -NotePropertyValue ([DateTime]::UtcNow.ToString("o"))
            Add-Content -LiteralPath $StatsPath -Value ($Sample | ConvertTo-Json -Compress) -Encoding UTF8
        }
    }
    if ($Status.state -notin @("completed", "failed")) { Start-Sleep -Milliseconds 500 }
} while ($Status.state -notin @("completed", "failed"))
$HostRunTimer.Stop()

$HostTiming = [ordered]@{
    compose_up_to_api_ready_seconds = [Math]::Round($ApiReadySeconds, 6)
    compose_up_to_terminal_state_seconds = [Math]::Round($HostRunTimer.Elapsed.TotalSeconds, 6)
    terminal_state = $Status.state
    attack_tests_enabled = [bool]$AttackTests
}
$HostTiming | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeDir "host_run_timing.json") -Encoding UTF8

if ($Status.state -eq "failed") {
    & $Docker compose -f $ComposeFile logs --no-color
    throw "Secure demo failed: $($Status.failure)"
}

$ClientSummaryDeadline = (Get-Date).AddSeconds(30)
do {
    $Events = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/events?after_seq=0" -TimeoutSec 5
    $ClientSummaryCount = @($Events.events | Where-Object { $_.event_type -eq "client_security_summary" }).Count
    if ($ClientSummaryCount -eq 5) { break }
    if ((Get-Date) -gt $ClientSummaryDeadline) { throw "Timed out waiting for five client security summaries." }
    Start-Sleep -Milliseconds 250
} while ($true)

$CompletedRunDir = Join-Path (Join-Path $RuntimeDir "runs") ([string]$Status.run_id)
Copy-Item -LiteralPath (Join-Path $RuntimeDir "host_run_timing.json") -Destination (Join-Path $CompletedRunDir "host_run_timing.json") -Force
Copy-Item -LiteralPath $StatsPath -Destination (Join-Path $CompletedRunDir "host_container_stats.jsonl") -Force

$VerificationPath = Join-Path $RuntimeDir "latest_verification.json"
& $Python (Join-Path $Phase5Root "verify_secure_demo.py") | Tee-Object -FilePath $VerificationPath
if ($LASTEXITCODE -ne 0) { throw "Secure aggregation verification failed." }
Write-Host "Secure demo completed and verified. Status: http://127.0.0.1:8080/api/v1/status"
Write-Host "Runtime evidence: $RuntimeDir"
