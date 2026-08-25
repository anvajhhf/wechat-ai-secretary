param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [Parameter(Mandatory = $true)]
    [string]$Title
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
Write-Host "只读查询 $Profile 滴答中的精确标题；不会创建、完成或修改任务。"
& $python -m wechat_secretary inspect-dida-task --title $Title
exit $LASTEXITCODE
