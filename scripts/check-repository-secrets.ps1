Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $root
$files = @(git ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 跟踪文件列表。"
}

$rules = [ordered]@{
    "私钥头" = '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
    "GitHub 访问令牌" = '(?i)(github_pat_|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]+'
    "疑似 sk 密钥" = '(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}'
    "疑似内嵌凭证" = '(?im)(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)[ \t]*[:=][ \t]*["'']?[A-Za-z0-9._-]{16,}'
    "URL 内嵌凭证" = 'https?://[^\s/:]+:[^\s/@]+@'
}
$hits = @()
foreach ($relative in $files) {
    if (($relative -replace '\\', '/') -eq "scripts/check-repository-secrets.ps1") {
        continue
    }
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        continue
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -gt 5MB) {
        continue
    }
    try {
        $content = Get-Content -LiteralPath $path -Raw -Encoding UTF8 -ErrorAction Stop
    } catch {
        continue
    }
    foreach ($rule in $rules.GetEnumerator()) {
        if ($content -match $rule.Value) {
            $hits += "$relative｜$($rule.Key)"
        }
    }
}

if ($hits.Count -gt 0) {
    Write-Host "发现疑似凭证；只显示文件名和规则，不显示内容："
    $hits | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
    exit 1
}
Write-Host "仓库凭证扫描通过；未显示任何文件内容。"
