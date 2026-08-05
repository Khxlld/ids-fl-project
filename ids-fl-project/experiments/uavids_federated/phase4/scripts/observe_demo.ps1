param([int]$AfterSequence = 0)

$ErrorActionPreference = "Stop"
$Status = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/status" -TimeoutSec 5
$Events = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/events?after_seq=$AfterSequence" -TimeoutSec 5
$Status | ConvertTo-Json -Depth 8
$Events.events | Select-Object seq, elapsed_ms, round, source, event_type, severity | Format-Table -AutoSize
