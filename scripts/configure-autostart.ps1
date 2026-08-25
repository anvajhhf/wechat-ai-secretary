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
$backgroundScript = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "background-gateway.ps1")
)
if (-not (Test-Path -LiteralPath $backgroundScript -PathType Leaf)) {
    throw "后台启动脚本不存在。"
}

if (-not $Apply) {
    $operation = if ($Remove) { "移除" } else { "安装" }
    Write-Host "当前仅预览：将为当前 Windows 用户$operation $taskName。"
    Write-Host "启动方式：用户登录时运行隐藏前台网关；不安装 Windows 服务。"
    exit 0
}

Import-Module ScheduledTasks -ErrorAction Stop
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($Remove) {
    if ($null -ne $existing) {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "$taskName 已移除。"
    } else {
        Write-Host "$taskName 不存在，无需移除。"
    }
    exit 0
}

$engine = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $engine) {
    $engine = (Get-Command powershell.exe -ErrorAction Stop).Source
}
$arguments = (
    '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden ' +
    '-ExecutionPolicy Bypass -File "' + $backgroundScript + '" -Profile ' + $Profile
)
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $engine -Argument $arguments `
    -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "微信 AI 个人秘书 $Profile 档案（当前用户登录时启动）" `
    -Force | Out-Null
if ($StartNow) {
    Start-ScheduledTask -TaskName $taskName
}
Write-Host "$taskName 已安装。后台启动与开机登录自启已启用。"
