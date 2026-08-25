param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfirmWechatReplies,
    [switch]$ConfirmRealWrites,
    [switch]$ConfirmTaskCompletion,
    [switch]$ConfirmReminders,
    [switch]$Detached
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
if (-not $ConfirmWechatReplies) {
    throw "未启动：微信自动回复需要显式参数 -ConfirmWechatReplies。"
}
if (-not $ConfirmRealWrites) {
    $env:SECRETARY_DRY_RUN = "true"
} else {
    $env:SECRETARY_DIDA_CREATES_APPROVED = "1"
}
if ($ConfirmTaskCompletion -and -not $ConfirmRealWrites) {
    throw "未启动：允许完成滴答任务前，必须先显式启用真实写入。"
}
if ($ConfirmTaskCompletion) {
    $env:SECRETARY_DIDA_COMPLETIONS_APPROVED = "1"
}
if (-not $ConfirmReminders) {
    $env:SECRETARY_REMINDERS_ENABLED = "false"
}

$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
& $python -m wechat_secretary doctor --strict
if ($LASTEXITCODE -ne 0) {
    throw "安全检查未通过，网关未启动。"
}
# 严格检查已确认模型完整；运行阶段禁止 Hugging Face 自动联网或补下载。
$env:HF_HUB_OFFLINE = "1"

if ($Detached) {
    $env:HERMES_GATEWAY_DETACHED = "1"
    $launchOutput = & $python -m wechat_secretary.detached_gateway 2>&1
    $launchExit = $LASTEXITCODE
    $pidValue = 0
    $lastLine = if (@($launchOutput).Count -gt 0) {
        [string]@($launchOutput)[-1]
    } else {
        ""
    }
    if ($launchExit -ne 0 -or -not [int]::TryParse($lastLine.Trim(), [ref]$pidValue)) {
        throw "后台网关启动失败；未显示内部信息或凭证。"
    }
    Write-Host "Hermes $Profile 档案已在当前用户下隐藏启动；未安装系统服务。"
    exit 0
}

Set-Location -LiteralPath $script:SecretaryRoot
Write-Host "Hermes $Profile 档案将以前台方式运行；按 Ctrl+C 停止。未安装系统服务。"
$profileMarker = "HERMES_HOME=$env:HERMES_HOME"
& $python -m wechat_secretary.hermes_gateway_entry `
    hermes_cli.main $profileMarker gateway run
exit $LASTEXITCODE
