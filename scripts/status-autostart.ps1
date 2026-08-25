param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module ScheduledTasks -ErrorAction Stop
$taskName = "WechatAISecretary-$Profile"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Host "$taskName：未安装"
    exit 1
}
$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Host "$taskName：$($task.State)"
$resultText = switch ([int]$info.LastTaskResult) {
    0 { "成功" }
    267009 { "正在运行（正常）" }
    default { [string]$info.LastTaskResult }
}
Write-Host "最近结果：$resultText"
Write-Host "最近启动：$($info.LastRunTime)"
$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$supervisorLog = Join-Path $root "runtime\logs\$Profile\supervisor.log"
if (Test-Path -LiteralPath $supervisorLog -PathType Leaf) {
    $lastEvent = Get-Content -LiteralPath $supervisorLog -Tail 1
    if ($lastEvent -match '^[^|]+\|(start|exit|exception|disabled)\|([A-Za-z0-9_.-]*)$') {
        $eventLabel = @{
            start = "网关已启动"
            exit = "网关进程已退出"
            exception = "监督进程捕获异常"
            disabled = "后台已停用"
        }[$Matches[1]]
        Write-Host "监督状态：$eventLabel"
    }
}
