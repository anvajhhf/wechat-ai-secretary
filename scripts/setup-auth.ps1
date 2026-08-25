param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$ConfigureModel,
    [switch]$ConfigureWeixin,
    [switch]$AuthorizeDida
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$hermes = Get-HermesCommand
Set-Location -LiteralPath $script:SecretaryRoot

if (-not ($ConfigureModel -or $ConfigureWeixin -or $AuthorizeDida)) {
    Write-Host "未执行授权。请显式选择 -ConfigureModel、-ConfigureWeixin 或 -AuthorizeDida。"
    exit 0
}
if ($ConfigureModel) {
    Write-Host "即将为 $Profile 档案进入 Hermes 模型向导；请在本机输入 DeepSeek Key，不要粘贴到聊天。"
    & $hermes model
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($ConfigureWeixin) {
    Write-Host "即将为 $Profile 档案进入 Weixin/iLink 扫码配置；只启用私聊和该使用者的 allowlist。"
    & $hermes gateway setup
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($AuthorizeDida) {
    Write-Host "即将为 $Profile 档案打开滴答清单 OAuth；请登录该使用者自己的滴答账号，授权后仍保持 Dry Run。"
    & $hermes mcp login dida365
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
