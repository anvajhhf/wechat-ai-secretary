Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $repositoryRoot

$contentRules = [ordered]@{
    "私钥头" = '-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----'
    "GitHub 访问令牌" = '(?i)(?:github_pat_|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]+'
    "疑似 sk 密钥" = '(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}'
    "AWS 访问密钥" = '(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])'
    "疑似 JWT" = '(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}'
    "疑似内嵌凭证" = '(?im)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)[ \t]*[:=][ \t]*["'']?[A-Za-z0-9+/._=-]{16,}'
    "Authorization 凭证" = '(?im)authorization[ \t]*[:=][ \t]*["'']?(?:bearer|basic)[ \t]+[A-Za-z0-9+/._=-]{12,}'
    "URL 内嵌凭证" = 'https?://[^\s/:]+:[^\s/@]+@'
    "微信账号标识" = '(?i)(?:wxid_[A-Za-z0-9_-]{6,}|o9cq[A-Za-z0-9_-]{10,}|[A-Za-z0-9._-]+@im\.(?:wechat|bot))'
    "本机用户目录" = '(?i)[A-Z]:\\Users\\[^\\\r\n]+\\'
}

$sensitivePathRules = [ordered]@{
    "环境变量文件" = '(?i)(?:^|/)\.env(?:\..+)?$'
    "本地运行目录" = '(?i)^runtime/'
    "真实档案配置" = '(?i)^config/secretary(?:\.(?!example\.toml$)[^/]+)?\.toml$'
    "凭证或证书文件" = '(?i)\.(?:pem|key|p12|pfx)$'
    "数据库或备份" = '(?i)\.(?:db|sqlite|sqlite3|wasbak)$'
    "日志文件" = '(?i)\.log$'
    "模型权重" = '(?i)\.(?:onnx|bin|ckpt|pt|pth|gguf|safetensors)$'
    "真实媒体" = '(?i)\.(?:wav|mp3|m4a|aac|flac|wma|silk|amr|opus|ogg|jpg|jpeg|png|gif|webp|heic|mp4|mov)$'
}

$hits = [System.Collections.Generic.List[string]]::new()

function Test-SensitivePath {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Scope
    )
    $normalized = $RelativePath -replace '\\', '/'
    if ($normalized -eq ".env.example") {
        return
    }
    foreach ($rule in $sensitivePathRules.GetEnumerator()) {
        if ($normalized -match $rule.Value) {
            $hits.Add("$Scope$normalized｜$($rule.Key)")
        }
    }
}

function Test-ContentRules {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Content,
        [Parameter(Mandatory = $true)][string]$Label
    )
    foreach ($rule in $contentRules.GetEnumerator()) {
        if ($Content -match $rule.Value) {
            $hits.Add("$Label｜$($rule.Key)")
        }
    }
}

$currentFiles = @(git ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 文件列表。"
}

foreach ($relativePath in $currentFiles) {
    $normalized = $relativePath -replace '\\', '/'
    Test-SensitivePath -RelativePath $normalized -Scope "当前:"
    if ($normalized -eq "scripts/check-repository-secrets.ps1") {
        continue
    }
    $fullPath = Join-Path $repositoryRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        continue
    }
    $item = Get-Item -LiteralPath $fullPath
    if ($item.Length -gt 5MB) {
        continue
    }
    try {
        $content = Get-Content -LiteralPath $fullPath -Raw -Encoding UTF8 -ErrorAction Stop
    } catch {
        continue
    }
    Test-ContentRules -Content $content -Label "当前:$normalized"
}

$historyPaths = @(git log --all --format= --name-only)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 历史路径。"
}
foreach ($historyPath in $historyPaths) {
    if (-not [string]::IsNullOrWhiteSpace($historyPath)) {
        Test-SensitivePath -RelativePath $historyPath -Scope "历史:"
    }
}

$objectPaths = @{}
$objectLines = @(git rev-list --objects --all)
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 历史对象。"
}
foreach ($line in $objectLines) {
    if ($line -match '^([0-9a-f]{40,64}) (.+)$') {
        $objectId = $Matches[1]
        $objectPath = $Matches[2] -replace '\\', '/'
        if (-not $objectPaths.ContainsKey($objectId)) {
            $objectPaths[$objectId] = $objectPath
        }
    }
}

foreach ($entry in $objectPaths.GetEnumerator()) {
    $objectId = [string]$entry.Key
    $objectPath = [string]$entry.Value
    if ($objectPath -eq "scripts/check-repository-secrets.ps1") {
        continue
    }
    $objectType = (& git cat-file -t $objectId 2>$null)
    if ($LASTEXITCODE -ne 0 -or $objectType -ne "blob") {
        continue
    }
    $objectSizeText = (& git cat-file -s $objectId 2>$null)
    if ($LASTEXITCODE -ne 0) {
        continue
    }
    $objectSize = 0L
    if (-not [long]::TryParse([string]$objectSizeText, [ref]$objectSize)) {
        continue
    }
    if ($objectSize -gt 5MB) {
        continue
    }
    try {
        $historyContent = (& git cat-file blob $objectId 2>$null) -join "`n"
    } catch {
        continue
    }
    if ($LASTEXITCODE -ne 0) {
        continue
    }
    Test-ContentRules -Content $historyContent -Label "历史:$objectPath"
}

$commitMessages = (& git log --all --format="%B") -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "无法读取 Git 历史提交说明。"
}
Test-ContentRules -Content $commitMessages -Label "历史:提交说明"

$uniqueHits = @($hits | Sort-Object -Unique)
if ($uniqueHits.Count -gt 0) {
    Write-Host "发现疑似敏感内容；只显示路径和规则，不显示内容："
    $uniqueHits | ForEach-Object { Write-Host $_ }
    exit 1
}
Write-Host "仓库当前文件及可达历史扫描通过；未显示任何文件内容。"
