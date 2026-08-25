param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$python = Get-SecretaryPython
$env:PYTHONPATH = Join-Path $script:SecretaryRoot "src"
& $python -m wechat_secretary doctor

$hermesPath = Join-Path $script:SecretaryRoot ".venv\Scripts\hermes.exe"
if (Test-Path -LiteralPath $hermesPath -PathType Leaf) {
    $gatewayStatus = & $hermesPath gateway status 2>&1
    $gatewayStatus | Where-Object {
        $_ -notmatch 'gateway install' -and $_ -notmatch 'Scheduled Task'
    }
} else {
    Write-Host "Hermes：尚未安装到项目 .venv"
}
