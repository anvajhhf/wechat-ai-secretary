param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
& $python -m wechat_secretary dry-run
exit $LASTEXITCODE
