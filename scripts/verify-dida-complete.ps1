param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfirmCompleteTest
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
if (-not $ConfirmCompleteTest) {
    throw "未执行：完成专用核验任务需要显式参数 -ConfirmCompleteTest。"
}
$env:SECRETARY_DIDA_COMPLETION_TEST_APPROVED = "1"
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "将只完成已记录的 $Profile 专用核验任务并立即回读；不会操作其他任务或自动重试。"
& $python -m wechat_secretary verify-dida-complete --confirm-complete-test
exit $LASTEXITCODE
