param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$Strict
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
$arguments = @("-m", "wechat_secretary", "doctor")
if ($Strict) {
    $arguments += "--strict"
}
& $python @arguments
exit $LASTEXITCODE
