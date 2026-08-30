param(
    [switch]$SkipHermes
)

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile owner

$root = $script:SecretaryRoot
$venv = Join-Path $root ".venv"
$runtime = Join-Path $root "runtime"
$hermesSource = Join-Path $runtime "hermes-agent"
$hermesCompatibilityPatches = @(
    (Join-Path $root "patches\hermes-dida-oauth-issuer.patch"),
    (Join-Path $root "patches\hermes-exact-tool-approval.patch"),
    (Join-Path $root "patches\hermes-weixin-compact-multiline.patch"),
    (Join-Path $root "patches\hermes-gateway-ready-hook.patch"),
    (Join-Path $root "patches\hermes-weixin-secretary-ingress.patch")
)
$hermesHome = Join-Path $runtime "hermes-home"
$partnerHermesHome = Join-Path $runtime "hermes-home-partner"
$pipCache = Join-Path $runtime "pip-cache"
$localTempRoot = Join-Path $runtime "temp"
$localTemp = Join-Path $localTempRoot ("install-" + [Guid]::NewGuid().ToString("N"))

foreach ($path in @($runtime, $hermesHome, $partnerHermesHome, $pipCache, $localTempRoot, $localTemp)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}
$env:PIP_CACHE_DIR = $pipCache
$env:TEMP = $localTemp
$env:TMP = $localTemp

if (-not (Test-Path -LiteralPath $venv -PathType Container)) {
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "创建项目虚拟环境失败。"
    }
}

$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    & $python -m ensurepip --upgrade --default-pip
    if ($LASTEXITCODE -ne 0) {
        throw "项目虚拟环境缺少 pip，修复失败。"
    }
}
& $python -m pip install --disable-pip-version-check -e "${root}[media]"
if ($LASTEXITCODE -ne 0) {
    throw "安装微信秘书本地包失败。"
}

if (-not $SkipHermes) {
    if (-not (Test-Path -LiteralPath $hermesSource -PathType Container)) {
        & git clone --depth 1 --branch v2026.8.19 `
            https://github.com/NousResearch/hermes-agent.git $hermesSource
        if ($LASTEXITCODE -ne 0) {
            throw "下载 Hermes Agent 官方源码失败。"
        }
    }
    $origin = (& git -C $hermesSource remote get-url origin).Trim()
    if ($origin -notmatch '^https://github\.com/NousResearch/hermes-agent(?:\.git)?$') {
        throw "runtime/hermes-agent 不是预期的官方仓库，已停止。"
    }
    $tag = (& git -C $hermesSource describe --tags --exact-match).Trim()
    if ($tag -ne "v2026.8.19") {
        throw "Hermes 源码版本不是锁定的 v2026.8.19，已停止。"
    }
    foreach ($compatibilityPatch in $hermesCompatibilityPatches) {
        & git -C $hermesSource apply --reverse --check $compatibilityPatch *> $null
        if ($LASTEXITCODE -ne 0) {
            & git -C $hermesSource apply --check $compatibilityPatch
            if ($LASTEXITCODE -ne 0) {
                throw "Hermes 兼容补丁与锁定的源码不匹配，已停止。"
            }
            & git -C $hermesSource apply $compatibilityPatch
            if ($LASTEXITCODE -ne 0) {
                throw "应用 Hermes 兼容补丁失败。"
            }
        }
    }
    & $python -m pip install --disable-pip-version-check -e "${hermesSource}[messaging,mcp]"
    if ($LASTEXITCODE -ne 0) {
        throw "安装 Hermes 项目依赖失败。"
    }
}

Write-Host "本地安装完成。未修改系统 PATH、未创建自启动、未启动后台服务。"
