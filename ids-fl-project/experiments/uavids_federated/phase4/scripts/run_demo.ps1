param(
    [switch]$SkipBuild,
    [switch]$SkipFailureExercise
)

$ErrorActionPreference = "Stop"
$Phase4Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExperimentRoot = (Resolve-Path (Join-Path $Phase4Root "..")).Path
$ComposeFile = Join-Path $Phase4Root "docker-compose.yml"
$RuntimeDir = Join-Path $Phase4Root "runtime"

function Find-Docker {
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallback = "C:\Users\kingn\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe"
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    throw "Docker CLI was not found. Start Docker Desktop and add its resources/bin directory to PATH."
}

function Find-Python {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallback = "C:\Users\kingn\AppData\Local\Programs\Python\Python311\python.exe"
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    throw "Python 3.11 was not found for host-side verification."
}

$Docker = Find-Docker
$Python = Find-Python
$env:Path = (Split-Path -Parent $Docker) + ";" + $env:Path
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

& $Docker compose -f $ComposeFile down --remove-orphans
if (-not $SkipBuild) {
    & $Docker compose -f $ComposeFile build --pull
}
$HostRunTimer = [Diagnostics.Stopwatch]::StartNew()
& $Docker compose -f $ComposeFile up -d --remove-orphans

$HealthDeadline = (Get-Date).AddMinutes(3)
do {
    try {
        $Status = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/status" -TimeoutSec 3
        break
    } catch {
        if ((Get-Date) -gt $HealthDeadline) { throw "Control center did not become healthy within three minutes." }
        Start-Sleep -Milliseconds 500
    }
} while ($true)
$ApiReadySeconds = $HostRunTimer.Elapsed.TotalSeconds

if (-not $SkipFailureExercise) {
    $FailureOutput = Join-Path $RuntimeDir "failure_path_stdout.json"
    $FailureError = Join-Path $RuntimeDir "failure_path_stderr.txt"
    $FailureProcess = Start-Process -FilePath $Python `
        -ArgumentList (Join-Path $Phase4Root "exercise_failure_path.py") `
        -RedirectStandardOutput $FailureOutput `
        -RedirectStandardError $FailureError `
        -WindowStyle Hidden `
        -PassThru
}

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
}
$HostTiming | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeDir "host_run_timing.json") -Encoding UTF8

if ($FailureProcess) {
    $FailureProcess.WaitForExit()
    $FailureProcess.Refresh()
    $FailureResult = Get-Content -Raw -LiteralPath $FailureOutput | ConvertFrom-Json
    if ($FailureResult.failure_path_verified -ne $true) {
        throw "The incompatible-update failure exercise did not complete successfully. See $FailureError"
    }
}

if ($Status.state -eq "failed") {
    & $Docker compose -f $ComposeFile logs --no-color
    throw "Demo failed: $($Status.failure)"
}

$VerificationPath = Join-Path $RuntimeDir "latest_verification.json"
& $Python (Join-Path $Phase4Root "verify_live_demo.py") | Tee-Object -FilePath $VerificationPath
Write-Host "Demo completed and verified. Status: http://127.0.0.1:8080/api/v1/status"
Write-Host "Runtime evidence: $RuntimeDir"
