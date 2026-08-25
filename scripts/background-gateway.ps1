param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$startScript = Join-Path $PSScriptRoot "start.ps1"

# This wrapper is invoked only by the per-user Scheduled Task installed after
# all four runtime gates have been explicitly approved.
& $startScript -Profile $Profile `
    -ConfirmWechatReplies `
    -ConfirmRealWrites `
    -ConfirmTaskCompletion `
    -ConfirmReminders
exit $LASTEXITCODE
