param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfirmCreateTest
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
if (-not $ConfirmCreateTest) {
    throw "未执行：创建专用核验任务需要显式参数 -ConfirmCreateTest。"
}
$env:SECRETARY_DIDA_CREATE_TEST_APPROVED = "1"
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "将只创建一条 $Profile 专用核验任务并立即回读；不会完成、删除或自动重试。"
& $python -m wechat_secretary verify-dida-create --confirm-create-test
exit $LASTEXITCODE
