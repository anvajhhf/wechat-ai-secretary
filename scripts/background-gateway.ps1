param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$startScript = Join-Path $PSScriptRoot "start.ps1"
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$controlDir = Join-Path $root "runtime\control"
$disabledMarker = Join-Path $controlDir "$Profile.disabled"
$logDir = Join-Path $root "runtime\logs\$Profile"
$logPath = Join-Path $logDir "supervisor.log"

foreach ($path in @($controlDir, $logDir)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Write-SupervisorEvent {
    param(
        [ValidateSet("start", "exit", "exception", "disabled")]
        [string]$Event,
        [string]$Detail = ""
    )
    if (Test-Path -LiteralPath $logPath -PathType Leaf) {
        $length = (Get-Item -LiteralPath $logPath).Length
        if ($length -ge 1MB) {
            for ($index = 4; $index -ge 1; $index--) {
                $source = Join-Path $logDir "supervisor.$index.log"
                $destination = Join-Path $logDir "supervisor.$($index + 1).log"
                if (Test-Path -LiteralPath $source -PathType Leaf) {
                    Move-Item -LiteralPath $source -Destination $destination -Force
                }
            }
            Move-Item -LiteralPath $logPath `
                -Destination (Join-Path $logDir "supervisor.1.log") -Force
        }
    }
    $safeDetail = ($Detail -replace '[^A-Za-z0-9_.-]', '')
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Encoding UTF8 `
        -Value "$timestamp|$Event|$safeDetail"
}

# This wrapper is invoked only by the per-user Scheduled Task installed after
# all four runtime gates have been explicitly approved.
$delaySeconds = 5
while (-not (Test-Path -LiteralPath $disabledMarker -PathType Leaf)) {
    Write-SupervisorEvent -Event "start"
    $exitCode = 1
    try {
        & $startScript -Profile $Profile `
            -ConfirmWechatReplies `
            -ConfirmRealWrites `
            -ConfirmTaskCompletion `
            -ConfirmReminders
        $exitCode = $LASTEXITCODE
        Write-SupervisorEvent -Event "exit" -Detail ([string]$exitCode)
    } catch {
        Write-SupervisorEvent -Event "exception" `
            -Detail $_.Exception.GetType().Name
    }
    if (Test-Path -LiteralPath $disabledMarker -PathType Leaf) {
        break
    }
    Start-Sleep -Seconds $delaySeconds
    $delaySeconds = [Math]::Min($delaySeconds * 2, 300)
}
Write-SupervisorEvent -Event "disabled"
exit 0
