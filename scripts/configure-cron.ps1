param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [string]$WeixinTarget = "",
    [switch]$UseStoredWeixinRoute,
    [switch]$Apply
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$root = $script:SecretaryRoot
$python = Get-SecretaryPython
if (-not $WeixinTarget -and $UseStoredWeixinRoute) {
    $stateDb = Join-Path $root "runtime\state\$Profile\secretary.sqlite3"
    if (-not (Test-Path -LiteralPath $stateDb -PathType Leaf)) {
        throw "没有找到 $Profile 的本地状态库，无法安全取得微信投递路由。"
    }
    $routeReader = @'
import sqlite3
import sys

path = sys.argv[1]
connection = sqlite3.connect(f"file:{path.replace(chr(92), '/')}?mode=ro", uri=True)
try:
    row = connection.execute(
        "SELECT chat_id FROM reminders "
        "WHERE platform = 'weixin' AND chat_id != '' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
finally:
    connection.close()
if not row or not str(row[0]).strip():
    raise SystemExit(3)
sys.stdout.write(str(row[0]).strip())
'@
    $WeixinTarget = (& $python -c $routeReader $stateDb)
    if ($LASTEXITCODE -ne 0 -or -not $WeixinTarget) {
        throw "$Profile 尚无已验证的本地微信路由；请先从该档案发送一条 allowlist 私聊。"
    }
}
if (-not $WeixinTarget) {
    throw "必须提供 -WeixinTarget，或使用 -UseStoredWeixinRoute 从该档案的本地状态安全读取。"
}
$WeixinTarget = ([string]$WeixinTarget).Trim()
if ($WeixinTarget.Length -gt 500 -or $WeixinTarget -match '\s') {
    throw "微信投递目标格式无效。"
}
$hermes = Get-HermesCommand
$delivery = "weixin:$WeixinTarget"
$morningPrompt = "只调用 secretary_morning_digest，并原样返回结果；不要调用其他工具。"
$eveningPrompt = "只调用 secretary_evening_review，并原样返回结果；不要调用其他工具。"
$morningName = "微信秘书-$Profile-08时今日重点"
$eveningName = "微信秘书-$Profile-22时晚间复盘"

Write-Host "准备为 $Profile 档案创建（Asia/Shanghai）：08:00 今日重点；22:00 晚间复盘。"
Write-Host "目标：weixin:[本地参数已提供，值不回显]"
if (-not $Apply) {
    Write-Host "当前仅预览。确认后重新运行并加 -Apply。"
    exit 0
}

$existing = (& $hermes cron list --all 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 $Profile 的现有 Cron；未创建任何新任务。"
}

if ($existing.Contains($morningName)) {
    Write-Host "$Profile 的 08:00 Cron 已存在，本次未重复创建。"
} else {
    & $hermes cron create "0 8 * * *" $morningPrompt `
        --name $morningName --deliver $delivery
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($existing.Contains($eveningName)) {
    Write-Host "$Profile 的 22:00 Cron 已存在，本次未重复创建。"
} else {
    & $hermes cron create "0 22 * * *" $eveningPrompt `
        --name $eveningName --deliver $delivery
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "$Profile 的 08:00/22:00 Cron 已配置；微信目标未回显。"
