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
$hermes = Get-HermesCommand
& $hermes gateway stop
exit $LASTEXITCODE
