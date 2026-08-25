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
