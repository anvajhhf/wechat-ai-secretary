param(
    [ValidateSet("owner", "partner")]
    [string]$Profile = "owner",
    [switch]$Apply,
    [switch]$SelfTest,
    [string]$VerifyArchive = "",
    [ValidateRange(1, 30)]
    [int]$Keep = 7
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")
Initialize-SecretaryProcess -Profile $Profile
$root = $script:SecretaryRoot
$backupRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $root "runtime\backups\$Profile")
)
$entropy = [Text.Encoding]::UTF8.GetBytes("wechat-ai-secretary|$Profile|backup-v1")

try {
    Add-Type -AssemblyName System.IO.Compression -ErrorAction Stop
    Add-Type -AssemblyName System.Security.Cryptography.ProtectedData -ErrorAction SilentlyContinue
} catch {
    throw "当前 PowerShell 无法加载加密备份组件。"
}

function Assert-PathWithinProject {
    param([string]$Path)
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $prefix = $root.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "备份工具拒绝访问项目目录以外的路径。"
    }
    return $resolved
}

function Test-BackupPayload {
    param([byte[]]$Payload)
    $stream = [IO.MemoryStream]::new($Payload, $false)
    try {
        $archive = [IO.Compression.ZipArchive]::new(
            $stream,
            [IO.Compression.ZipArchiveMode]::Read,
            $false
        )
        try {
            $names = @($archive.Entries | ForEach-Object { $_.FullName })
            $requiredConfigName = if ($Profile -eq "owner") {
                "secretary.toml"
            } else {
                "secretary.partner.toml"
            }
            $requiredConfig = "config/$requiredConfigName"
            $requiredState = "runtime/state/$Profile/secretary.sqlite3"
            if ($requiredConfig -notin $names -or $requiredState -notin $names) {
                throw "加密包缺少必要的配置或状态快照。"
            }
            foreach ($entry in $archive.Entries) {
                if (
                    [IO.Path]::IsPathRooted($entry.FullName) -or
                    $entry.FullName -match '(^|/)\.\.(/|$)'
                ) {
                    throw "加密包包含不安全路径。"
                }
                $reader = $entry.Open()
                try {
                    $buffer = New-Object byte[] 65536
                    while ($reader.Read($buffer, 0, $buffer.Length) -gt 0) { }
                } finally {
                    $reader.Dispose()
                }
            }
        } finally {
            $archive.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

if ($SelfTest) {
    $configName = if ($Profile -eq "owner") { "secretary.toml" } else { "secretary.partner.toml" }
    $stream = [IO.MemoryStream]::new()
    try {
        $archive = [IO.Compression.ZipArchive]::new(
            $stream,
            [IO.Compression.ZipArchiveMode]::Create,
            $true
        )
        try {
            foreach ($name in @(
                "config/$configName",
                "runtime/state/$Profile/secretary.sqlite3"
            )) {
                $entry = $archive.CreateEntry($name)
                $entryStream = $entry.Open()
                try {
                    $bytes = [Text.Encoding]::UTF8.GetBytes("synthetic-self-test")
                    $entryStream.Write($bytes, 0, $bytes.Length)
                } finally {
                    $entryStream.Dispose()
                }
            }
        } finally {
            $archive.Dispose()
        }
        $plain = $stream.ToArray()
    } finally {
        $stream.Dispose()
    }
    $encrypted = [Security.Cryptography.ProtectedData]::Protect(
        $plain,
        $entropy,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    $opened = [Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted,
        $entropy,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    try {
        Test-BackupPayload -Payload $opened
        Write-Host "$Profile 加密备份自检通过；只使用了合成数据。"
    } finally {
        [Array]::Clear($plain, 0, $plain.Length)
        [Array]::Clear($encrypted, 0, $encrypted.Length)
        [Array]::Clear($opened, 0, $opened.Length)
    }
    exit 0
}

if ($VerifyArchive) {
    if (-not (Test-Path -LiteralPath $backupRoot -PathType Container)) {
        throw "当前档案还没有本地加密备份目录。"
    }
    $candidate = [System.IO.Path]::GetFullPath($VerifyArchive)
    $backupPrefix = $backupRoot.TrimEnd('\') + '\'
    if (
        -not $candidate.StartsWith($backupPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetExtension($candidate) -ne ".wasbak"
    ) {
        throw "只允许验证当前档案备份目录内的 .wasbak 文件。"
    }
    $encrypted = [IO.File]::ReadAllBytes($candidate)
    $plain = [Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted,
        $entropy,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    try {
        Test-BackupPayload -Payload $plain
        Write-Host "$Profile 加密备份验证通过；未解压、未显示任何内容。"
    } finally {
        [Array]::Clear($plain, 0, $plain.Length)
        [Array]::Clear($encrypted, 0, $encrypted.Length)
    }
    exit 0
}

$configName = if ($Profile -eq "owner") { "secretary.toml" } else { "secretary.partner.toml" }
$homeName = if ($Profile -eq "owner") { "hermes-home" } else { "hermes-home-partner" }
$configPath = Assert-PathWithinProject (Join-Path $root "config\$configName")
$stateDir = Assert-PathWithinProject (Join-Path $root "runtime\state\$Profile")
$stateDb = Assert-PathWithinProject (Join-Path $stateDir "secretary.sqlite3")
$hermesHome = Assert-PathWithinProject (Join-Path $root "runtime\$homeName")

foreach ($path in @($configPath, $stateDb, $hermesHome)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "备份源不完整；未创建备份。"
    }
    $item = Get-Item -LiteralPath $path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "备份源包含重解析点；已拒绝。"
    }
}

Write-Host "将为 $Profile 备份项目内的独立配置、SQLite 一致性快照和 Hermes 授权状态。"
Write-Host "不会读取项目外的 Vault，不会把明文凭证写入临时文件。"
if (-not $Apply) {
    Write-Host "当前仅预览。确认后加 -Apply；备份使用当前 Windows 用户的 DPAPI 加密。"
    exit 0
}

$homeIncludes = @(
    ".env",
    "auth.json",
    "channel_directory.json",
    "config.yaml",
    "gateway_state.json",
    "SOUL.md",
    "state.db",
    "kanban.db",
    "cron",
    "mcp-tokens",
    "pairing",
    "state",
    "weixin"
)
$files = @()
foreach ($relativeSource in $homeIncludes) {
    $source = Join-Path $hermesHome $relativeSource
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        $files += Get-Item -LiteralPath $source -Force
    } elseif (Test-Path -LiteralPath $source -PathType Container) {
        $files += Get-ChildItem -LiteralPath $source -Recurse -File -Force -ErrorAction Stop
    }
}
foreach ($file in $files) {
    $resolved = Assert-PathWithinProject $file.FullName
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Hermes 状态中包含重解析点；已拒绝备份。"
    }
}
$totalBytes = ($files | Measure-Object Length -Sum).Sum + (Get-Item $configPath).Length
if ($totalBytes -gt 128MB) {
    throw "备份源超过 128 MiB 安全上限；未创建备份。"
}

$python = Get-SecretaryPython
$snapshotScript = Join-Path $PSScriptRoot "snapshot-sqlite.py"
$snapshotBase64 = (& $python $snapshotScript $stateDb 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $snapshotBase64) {
    throw "无法生成 SQLite 一致性快照；未创建备份。"
}
$stateBytes = [Convert]::FromBase64String($snapshotBase64)
$zipStream = [IO.MemoryStream]::new()
try {
    $archive = [IO.Compression.ZipArchive]::new(
        $zipStream,
        [IO.Compression.ZipArchiveMode]::Create,
        $true
    )
    try {
        function Add-BytesToArchive {
            param([string]$Name, [byte[]]$Bytes)
            $entry = $archive.CreateEntry($Name, [IO.Compression.CompressionLevel]::Optimal)
            $stream = $entry.Open()
            try {
                $stream.Write($Bytes, 0, $Bytes.Length)
            } finally {
                $stream.Dispose()
            }
        }
        function Add-FileToArchive {
            param([string]$Name, [string]$Path)
            $entry = $archive.CreateEntry($Name, [IO.Compression.CompressionLevel]::Optimal)
            $output = $entry.Open()
            $input = [IO.File]::Open(
                $Path,
                [IO.FileMode]::Open,
                [IO.FileAccess]::Read,
                [IO.FileShare]::ReadWrite
            )
            try {
                $input.CopyTo($output)
            } finally {
                $input.Dispose()
                $output.Dispose()
            }
        }

        Add-FileToArchive -Name "config/$configName" -Path $configPath
        Add-BytesToArchive -Name "runtime/state/$Profile/secretary.sqlite3" `
            -Bytes $stateBytes
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($hermesHome.Length).TrimStart('\')
            $entryName = "runtime/$homeName/" + ($relative -replace '\\', '/')
            Add-FileToArchive -Name $entryName -Path $file.FullName
        }
    } finally {
        $archive.Dispose()
    }
    $plain = $zipStream.ToArray()
} finally {
    $zipStream.Dispose()
    [Array]::Clear($stateBytes, 0, $stateBytes.Length)
    $snapshotBase64 = ""
}

$encrypted = $null
try {
    Test-BackupPayload -Payload $plain
    $encrypted = [Security.Cryptography.ProtectedData]::Protect(
        $plain,
        $entropy,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    if (-not (Test-Path -LiteralPath $backupRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $backupRoot | Out-Null
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $destination = Join-Path $backupRoot "$Profile-$stamp.wasbak"
    $temporary = "$destination.tmp"
    [IO.File]::WriteAllBytes($temporary, $encrypted)
    Move-Item -LiteralPath $temporary -Destination $destination
    $old = @(
        Get-ChildItem -LiteralPath $backupRoot -Filter "$Profile-*.wasbak" -File |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -Skip $Keep
    )
    foreach ($item in $old) {
        $resolved = [IO.Path]::GetFullPath($item.FullName)
        if ($resolved.StartsWith($backupRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolved -Force
        }
    }
    Write-Host "$Profile 加密备份已创建并在内存中验证；内容和凭证均未显示。"
} finally {
    [Array]::Clear($plain, 0, $plain.Length)
    if ($null -ne $encrypted) {
        [Array]::Clear($encrypted, 0, $encrypted.Length)
    }
}
