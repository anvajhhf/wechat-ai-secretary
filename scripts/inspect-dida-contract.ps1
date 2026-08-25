param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "正在只读核对 $Profile 的专用测试任务；不会显示任务标识或修改滴答任务。"
& $python -m wechat_secretary inspect-dida-contract
exit $LASTEXITCODE
