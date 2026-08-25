param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfirmWechatReplies,
    [switch]$ConfirmRealWrites,
    [switch]$ConfirmTaskCompletion,
    [switch]$ConfirmReminders
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

$hermes = Get-HermesCommand
Set-Location -LiteralPath $script:SecretaryRoot
Write-Host "Hermes $Profile 档案将以前台方式运行；按 Ctrl+C 停止。未安装系统服务。"
& $hermes gateway run
exit $LASTEXITCODE
