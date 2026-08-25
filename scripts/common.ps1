Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:SecretaryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path -Path $PSScriptRoot -ChildPath "..")
)

function Import-SecretaryEnvFile {
    param(
        [string]$EnvPath,
        [string[]]$AllowedNames = @()
    )
    if (-not (Test-Path -LiteralPath $EnvPath -PathType Leaf)) {
        return
    }
    foreach ($line in Get-Content -LiteralPath $EnvPath -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -notmatch '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            throw "Invalid .env line. Fix the local file; no value was printed."
        }
        $name = $Matches[1]
        if ($AllowedNames.Count -gt 0 -and $name -notin $AllowedNames) {
            continue
        }
        $value = $Matches[2].Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Import-SecretaryEnv {
    $projectEnv = Join-Path $script:SecretaryRoot ".env"
    $hermesEnv = Join-Path $env:HERMES_HOME ".env"
    # 根目录 .env 只允许放两套档案可共用的 DeepSeek 设置，避免微信凭证串档。
    Import-SecretaryEnvFile $projectEnv @("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL")
    Import-SecretaryEnvFile $hermesEnv
}

function Initialize-SecretaryProcess {
    param(
        [ValidateSet("owner", "partner")]
        [string]$Profile = "owner"
    )

    # 防止在同一个 PowerShell 窗口先后操作两套档案时沿用上一人的进程变量。
    foreach ($name in [Environment]::GetEnvironmentVariables("Process").Keys) {
        if (
            ([string]$name).StartsWith("WEIXIN_", [StringComparison]::OrdinalIgnoreCase) -or
            $name -in @(
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_BASE_URL",
                "HERMES_GATEWAY_DETACHED",
                "SECRETARY_DRY_RUN",
                "SECRETARY_DIDA_WRITES_APPROVED",
                "SECRETARY_DIDA_CREATES_APPROVED",
                "SECRETARY_DIDA_COMPLETIONS_APPROVED",
                "SECRETARY_DIDA_CREATE_TEST_APPROVED",
                "SECRETARY_DIDA_COMPLETION_TEST_APPROVED",
                "SECRETARY_REMINDERS_ENABLED",
                "SECRETARY_VAULT_PATH",
                "SECRETARY_PRIVATE_INBOX_PATH"
            )
        ) {
            [Environment]::SetEnvironmentVariable([string]$name, $null, "Process")
        }
    }

    if ($Profile -eq "owner") {
        $homeName = "hermes-home"
        $configName = "secretary.toml"
    } else {
        $homeName = "hermes-home-partner"
        $configName = "secretary.partner.toml"
    }

    $env:HERMES_HOME = Join-Path $script:SecretaryRoot "runtime\$homeName"
    $sharedModelHome = Join-Path $script:SecretaryRoot "runtime\models"
    if (-not (Test-Path -LiteralPath $sharedModelHome -PathType Container)) {
        New-Item -ItemType Directory -Path $sharedModelHome | Out-Null
    }
    # 两套档案只共享只读模型权重；消息、媒体缓存和状态仍分别位于各自 HERMES_HOME。
    $env:HF_HOME = Join-Path $sharedModelHome "huggingface"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    # Windows 普通用户缓存不依赖软链接；禁用 Xet 可避免部分网络环境在大文件阶段挂起。
    $env:HF_HUB_DISABLE_XET = "1"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:SECRETARY_MODEL_HOME = $sharedModelHome
    $env:HERMES_ENABLE_PROJECT_PLUGINS = "true"
    $env:SECRETARY_CONFIG = Join-Path $script:SecretaryRoot "config\$configName"
    $env:SECRETARY_PROFILE = $Profile
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    Import-SecretaryEnv
}

function Get-SecretaryPython {
    $candidate = Join-Path $script:SecretaryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        return $candidate
    }
    return "python"
}

function Get-HermesCommand {
    $candidate = Join-Path $script:SecretaryRoot ".venv\Scripts\hermes.exe"
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Hermes 尚未安装到项目 .venv；请先运行 scripts\install-local.ps1。"
    }
    return $candidate
}
