param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfirmStop
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
if (-not $ConfirmStop) {
    Write-Host "前台运行时请在启动窗口按 Ctrl+C。若确需发送停止命令，请加 -ConfirmStop。"
    exit 0
}
$controlDir = Join-Path $script:SecretaryRoot "runtime\control"
if (-not (Test-Path -LiteralPath $controlDir -PathType Container)) {
    New-Item -ItemType Directory -Path $controlDir | Out-Null
}
$disabledMarker = Join-Path $controlDir "$Profile.disabled"
Set-Content -LiteralPath $disabledMarker -Encoding UTF8 -Value "disabled"
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
& $python -m wechat_secretary.profile_gateway_stop
$gatewayExit = $LASTEXITCODE
$taskName = "WechatAISecretary-$Profile"
$healthTaskName = "WechatAISecretaryHealth-$Profile"
try {
    Import-Module ScheduledTasks -ErrorAction Stop
    foreach ($name in @($healthTaskName, $taskName)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -ne $task) {
            if ($name -eq $healthTaskName) {
                Disable-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue |
                    Out-Null
            }
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        }
    }
} catch {
    Write-Host "网关已停止，但无法读取当前用户的后台任务状态。"
}
exit $gatewayExit
