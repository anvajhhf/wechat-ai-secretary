param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$Repair
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$homeName = if ($Profile -eq "owner") { "hermes-home" } else { "hermes-home-partner" }
$hermesHome = [System.IO.Path]::GetFullPath((Join-Path $root "runtime\$homeName"))
$runtimePath = Join-Path $hermesHome "gateway_state.json"
$controlDir = Join-Path $root "runtime\control"
$disabledMarker = Join-Path $controlDir "$Profile.disabled"
$maintenanceMarker = Join-Path $controlDir "$Profile.maintenance"
$healthStatePath = Join-Path $controlDir "$Profile.health.json"
$logDir = Join-Path $root "runtime\logs\$Profile"
$logPath = Join-Path $logDir "health.log"
$taskName = "WechatAISecretary-$Profile"

foreach ($path in @($controlDir, $logDir)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Write-HealthEvent {
    param(
        [ValidateSet("healthy", "degraded", "recover", "disabled", "error")]
        [string]$Event,
        [string]$Detail = ""
    )
    $safeDetail = ($Detail -replace '[^A-Za-z0-9_.-]', '')
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value "$timestamp|$Event|$safeDetail"
}

function Read-HealthState {
    if (-not (Test-Path -LiteralPath $healthStatePath -PathType Leaf)) {
        return [pscustomobject]@{
            last_state = ""
            connection_failures = 0
            suppress_until = ""
        }
    }
    try {
        $state = Get-Content -LiteralPath $healthStatePath -Raw -Encoding UTF8 |
            ConvertFrom-Json -ErrorAction Stop
        return [pscustomobject]@{
            last_state = [string]$state.last_state
            connection_failures = [int]$state.connection_failures
            suppress_until = [string]$state.suppress_until
        }
    } catch {
        return [pscustomobject]@{
            last_state = ""
            connection_failures = 0
            suppress_until = ""
        }
    }
}

function Save-HealthState {
    param(
        [string]$LastState,
        [int]$ConnectionFailures = 0,
        [datetime]$SuppressUntil = [datetime]::MinValue
    )
    $payload = [ordered]@{
        last_state = $LastState
        connection_failures = $ConnectionFailures
        suppress_until = if ($SuppressUntil -eq [datetime]::MinValue) {
            ""
        } else {
            $SuppressUntil.ToUniversalTime().ToString("o")
        }
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath $healthStatePath -Encoding UTF8 -Value $payload
}

function Get-ActualGatewayState {
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        return [pscustomobject]@{ state = "down"; reason = "state-missing" }
    }
    try {
        $runtime = Get-Content -LiteralPath $runtimePath -Raw -Encoding UTF8 |
            ConvertFrom-Json -ErrorAction Stop
    } catch {
        return [pscustomobject]@{ state = "down"; reason = "state-invalid" }
    }

    $pidValue = 0
    if (-not [int]::TryParse([string]$runtime.pid, [ref]$pidValue) -or $pidValue -le 0) {
        return [pscustomobject]@{ state = "down"; reason = "pid-invalid" }
    }
    $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return [pscustomobject]@{ state = "down"; reason = "pid-gone" }
    }

    $recordedStart = 0L
    if (-not [long]::TryParse([string]$runtime.start_time, [ref]$recordedStart)) {
        return [pscustomobject]@{ state = "down"; reason = "start-invalid" }
    }
    try {
        $startedAt = [DateTimeOffset]$process.StartTime.ToUniversalTime()
        $actualStart = [long][Math]::Floor($startedAt.ToUnixTimeMilliseconds() / 10.0)
        if ([Math]::Abs($actualStart - $recordedStart) -gt 1) {
            return [pscustomobject]@{ state = "down"; reason = "pid-reused" }
        }
    } catch {
        return [pscustomobject]@{ state = "down"; reason = "start-unavailable" }
    }

    try {
        $recordedHome = [System.IO.Path]::GetFullPath([string]$runtime.hermes_home)
        if (-not $recordedHome.Equals($hermesHome, [StringComparison]::OrdinalIgnoreCase)) {
            return [pscustomobject]@{ state = "down"; reason = "profile-mismatch" }
        }
    } catch {
        return [pscustomobject]@{ state = "down"; reason = "profile-invalid" }
    }

    $gatewayState = [string]$runtime.gateway_state
    if ($gatewayState -notin @("running", "draining", "starting")) {
        return [pscustomobject]@{ state = "down"; reason = "runtime-not-live" }
    }

    $weixinState = ""
    if ($null -ne $runtime.platforms) {
        $weixin = $runtime.platforms.PSObject.Properties["weixin"]
        if ($null -ne $weixin -and $null -ne $weixin.Value) {
            $weixinState = [string]$weixin.Value.state
        }
    }
    if ($gatewayState -eq "running" -and $weixinState -eq "connected") {
        return [pscustomobject]@{ state = "healthy"; reason = "connected" }
    }
    return [pscustomobject]@{ state = "degraded"; reason = "weixin-not-connected" }
}

function Wait-ScheduledTaskNotRunning {
    param([string]$Name, [int]$TimeoutSeconds = 15)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $current = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
        if ($null -eq $current -or $current.State -ne "Running") {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "main-task-stop-timeout"
}

function Repair-Gateway {
    Import-Module ScheduledTasks -ErrorAction Stop
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        throw "main-task-missing"
    }
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        Wait-ScheduledTaskNotRunning -Name $taskName
    }
    Start-ScheduledTask -TaskName $taskName

    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Seconds 1
        $actual = Get-ActualGatewayState
        if ($actual.state -eq "healthy") {
            return $true
        }
    } while ((Get-Date) -lt $deadline)
    return $false
}

$previous = Read-HealthState
if (Test-Path -LiteralPath $disabledMarker -PathType Leaf) {
    if ($previous.last_state -ne "disabled") {
        Write-HealthEvent -Event "disabled"
    }
    Save-HealthState -LastState "disabled"
    Write-Host "$Profile 网关已按用户要求停用。"
    exit 0
}
if (Test-Path -LiteralPath $maintenanceMarker -PathType Leaf) {
    Write-Host "$Profile 网关正在维护，本次健康检查未干预。"
    exit 0
}

$now = (Get-Date).ToUniversalTime()
$suppressUntil = [datetime]::MinValue
if ($previous.suppress_until) {
    [void][datetime]::TryParse($previous.suppress_until, [ref]$suppressUntil)
}
$actual = Get-ActualGatewayState
if ($actual.state -eq "healthy") {
    if ($previous.last_state -ne "healthy") {
        Write-HealthEvent -Event "healthy"
    }
    Save-HealthState -LastState "healthy"
    Write-Host "$Profile 网关健康：进程存活，微信已连接。"
    exit 0
}

if ($suppressUntil -gt $now) {
    Write-Host "$Profile 网关正在启动宽限期内，暂不重复恢复。"
    exit 0
}

$connectionFailures = if ($actual.state -eq "degraded") {
    $previous.connection_failures + 1
} else {
    0
}
if ($previous.last_state -ne $actual.state) {
    Write-HealthEvent -Event "degraded" -Detail $actual.reason
}

$shouldRepair = $actual.state -eq "down" -or $connectionFailures -ge 5
if (-not $Repair -or -not $shouldRepair) {
    Save-HealthState -LastState $actual.state `
        -ConnectionFailures $connectionFailures
    if ($actual.state -eq "degraded") {
        Write-Host "$Profile 网关进程存活，但微信连接尚未恢复。"
        exit 3
    }
    Write-Host "$Profile 网关实际进程未运行。"
    exit 2
}

try {
    Write-HealthEvent -Event "recover" -Detail $actual.reason
    Save-HealthState -LastState "recovering" `
        -SuppressUntil $now.AddMinutes(2)
    $recovered = Repair-Gateway
    if (-not $recovered) {
        Write-HealthEvent -Event "error" -Detail "recovery-timeout"
        Write-Host "$Profile 网关已触发恢复，但尚未达到健康状态。"
        exit 4
    }
    Write-HealthEvent -Event "healthy" -Detail "recovered"
    Save-HealthState -LastState "healthy"
    Write-Host "$Profile 网关已自动恢复：进程存活，微信已连接。"
    exit 0
} catch {
    $detail = $_.Exception.Message -replace '[^A-Za-z0-9_.-]', ''
    Write-HealthEvent -Event "error" -Detail $detail
    Write-Host "$Profile 网关自动恢复失败；未显示内部信息。"
    exit 5
}
