param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "正在只读发现 $Profile 档案的滴答工具参数；不会调用任何任务工具。"
& $python -m wechat_secretary inspect-dida-schema
exit $LASTEXITCODE
