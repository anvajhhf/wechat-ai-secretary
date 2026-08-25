param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "正在为 $Profile 档案只读检查滴答分类；不会启动聊天，也不会创建、完成或删除任务。"
& $python -m wechat_secretary inspect-dida
exit $LASTEXITCODE
