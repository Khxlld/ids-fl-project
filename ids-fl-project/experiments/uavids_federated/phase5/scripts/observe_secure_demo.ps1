$ErrorActionPreference = "Stop"
$LastSequence = 0
do {
    $Status = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/status" -TimeoutSec 5
    $Response = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/events?after_seq=$LastSequence" -TimeoutSec 5
    foreach ($Event in $Response.events) {
        Write-Host ("[{0}] {1} r{2} {3} ({4})" -f $Event.timestamp_utc, $Event.source, $Event.round, $Event.event_type, $Event.severity)
        $LastSequence = [Math]::Max($LastSequence, [int]$Event.seq)
    }
    if ($Status.state -notin @("completed", "failed")) { Start-Sleep -Milliseconds 500 }
} while ($Status.state -notin @("completed", "failed"))
$Status | ConvertTo-Json -Depth 8
