param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner"
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$hermes = Get-HermesCommand
Set-Location -LiteralPath $script:SecretaryRoot

$settings = [ordered]@{
    "stt.provider" = "local"
    "stt.language" = "zh"
    "stt.local.model" = "small"
    "stt.local.language" = "zh"
    "stt.local.device" = "cpu"
    "stt.local.compute_type" = "int8"
    "stt.local.vad" = "true"
}

foreach ($entry in $settings.GetEnumerator()) {
    & $hermes config set $entry.Key $entry.Value
    if ($LASTEXITCODE -ne 0) {
        throw "无法为 $Profile 设置 $($entry.Key)。"
    }
}

Write-Host "$Profile 的本地语音配置已写入项目专用 Hermes 档案。"
Write-Host "图片继续由项目插件路由到 deepseek-v4-flash-vision-exp。"
