param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$Apply,
    [switch]$StartNow,
    [switch]$Remove
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$taskName = "WechatAISecretary-$Profile"
$healthTaskName = "WechatAISecretaryHealth-$Profile"
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$controlDir = Join-Path $root "runtime\control"
$disabledMarker = Join-Path $controlDir "$Profile.disabled"
$backgroundScript = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "background-gateway.ps1")
)
$healthScript = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "gateway-health.ps1")
)
$hiddenLauncher = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "run-hidden.vbs")
)
if (
    -not (Test-Path -LiteralPath $backgroundScript -PathType Leaf) -or
    -not (Test-Path -LiteralPath $healthScript -PathType Leaf) -or
    -not (Test-Path -LiteralPath $hiddenLauncher -PathType Leaf)
) {
    throw "后台启动、健康检查或静默启动脚本不存在。"
}

if (-not $Apply) {
    $operation = if ($Remove) { "移除" } else { "安装" }
    Write-Host "当前仅预览：将为当前 Windows 用户$operation $taskName。"
    Write-Host "启动方式：用户登录时运行隐藏前台网关，并每分钟核对真实进程与微信连接；不安装 Windows 服务。"
    exit 0
}

Import-Module ScheduledTasks -ErrorAction Stop
. (Join-Path $PSScriptRoot "common.ps1")

function Wait-ScheduledTaskNotRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [int]$TimeoutSeconds = 15
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $current = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
        if ($null -eq $current -or $current.State -ne "Running") {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "$Name 未能在限定时间内停止；为避免重启竞态，本次未继续安装。"
}

function Stop-ProfileGateway {
    Initialize-SecretaryProcess -Profile $Profile
    $python = Get-SecretaryPython
    $env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
    $stopOutput = & $python -m wechat_secretary.profile_gateway_stop 2>&1
    $stopExit = $LASTEXITCODE
    $null = $stopOutput
    if ($stopExit -ne 0) {
        throw "$Profile 旧网关未能安全停止；为避免两个实例并存，本次未继续安装。"
    }
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$existingHealth = Get-ScheduledTask -TaskName $healthTaskName -ErrorAction SilentlyContinue
if ($Remove) {
    if (-not (Test-Path -LiteralPath $controlDir -PathType Container)) {
        New-Item -ItemType Directory -Path $controlDir | Out-Null
    }
    Set-Content -LiteralPath $disabledMarker -Encoding UTF8 -Value "disabled"
    $removed = $false
    foreach ($name in @($healthTaskName, $taskName)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -ne $task) {
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            Wait-ScheduledTaskNotRunning -Name $name
            $removed = $true
        }
    }
    Stop-ProfileGateway
    foreach ($name in @($healthTaskName, $taskName)) {
        if ($null -ne (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
        }
    }
    if ($removed) {
        Write-Host "$taskName 及其健康看护已移除。"
    } else {
        Write-Host "$taskName 不存在，无需移除。"
    }
    exit 0
}

if ($null -ne $existingHealth) {
    Disable-ScheduledTask -TaskName $healthTaskName -ErrorAction SilentlyContinue | Out-Null
    if ($existingHealth.State -eq "Running") {
        Stop-ScheduledTask -TaskName $healthTaskName -ErrorAction SilentlyContinue
        Wait-ScheduledTaskNotRunning -Name $healthTaskName
    }
}
if ($null -ne $existing -and $existing.State -eq "Running") {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Wait-ScheduledTaskNotRunning -Name $taskName
}
Stop-ProfileGateway
if (Test-Path -LiteralPath $disabledMarker -PathType Leaf) {
    Remove-Item -LiteralPath $disabledMarker -Force
}

$engine = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $engine) {
    $engine = (Get-Command powershell.exe -ErrorAction Stop).Source
}
$wscript = (Get-Command wscript.exe -ErrorAction Stop).Source
$arguments = (
    '"' + $hiddenLauncher + '" "' + $engine + '" "' +
    $backgroundScript + '" ' + $Profile + ' gateway'
)
$healthArguments = (
    '"' + $hiddenLauncher + '" "' + $engine + '" "' +
    $healthScript + '" ' + $Profile + ' health'
)
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $wscript -Argument $arguments `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$healthAction = New-ScheduledTaskAction -Execute $wscript -Argument $healthArguments `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$healthTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -Hidden `
    -MultipleInstances IgnoreNew
$healthSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -Hidden `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "微信 AI 个人秘书 $Profile 档案（当前用户登录时启动）" `
    -Force | Out-Null
Register-ScheduledTask -TaskName $healthTaskName -Action $healthAction `
    -Trigger $healthTrigger -Principal $principal -Settings $healthSettings `
    -Description "微信 AI 个人秘书 $Profile 档案真实健康看护（不记录消息内容）" `
    -Force | Out-Null
if ($StartNow) {
    Start-ScheduledTask -TaskName $taskName
    $healthDeadline = (Get-Date).AddSeconds(30)
    $healthy = $false
    do {
        $healthOutput = & $engine -NoLogo -NoProfile -NonInteractive `
            -WindowStyle Hidden -ExecutionPolicy Bypass `
            -File $healthScript -Profile $Profile 2>&1
        if ($LASTEXITCODE -eq 0) {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $healthDeadline)
    if (-not $healthy) {
        throw "$taskName 已启动，但真实网关或微信连接未在 30 秒内就绪。"
    }
}
Write-Host "$taskName 已安装。后台启动、登录自启与真实健康看护已启用。"
