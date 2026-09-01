param(
    [ValidateSet("sensevoice", "paraformer", "both")]
    [string]$Backend = "both",
    [ValidateSet("huggingface.co", "hf-mirror.com")]
    [string]$ModelHost = "huggingface.co"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$asrProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$asrModelRoot = Join-Path $asrProjectRoot "runtime\models\speech"
$asrPython = Join-Path $asrProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $asrPython -PathType Leaf)) { throw "Prepare the project Python environment first." }
$asrSources = @(
    @{ Name="sensevoice"; Repo="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"; Revision="2365baeacb507f821a0c8120fcee3d484dba7a07"; Bytes=239233841; Hash="c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"; TokensBytes=315894; TokensBlob="2cfc92fc2ff26aaa690b7c01fd96b41109413881"; TokensHash="f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc" },
    @{ Name="paraformer"; Repo="csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28"; Revision="fe3e2bbfa0a0d3789b653c4b6cf3f87a5dbf2b94"; Bytes=223385835; Hash="9ada9127ca5b82320385ac12340eb8b05dee64fd45cf8cf593ec693826ec2fd7"; TokensBytes=75756; TokensBlob="57bc045ddda0434ed4440c38e14287c595b258d9"; TokensHash="59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6" }
)

function Get-AsrPathItem {
    param([string]$Path)
    try { return Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch [System.Management.Automation.ItemNotFoundException] { return $null }
}

function Assert-AsrPath {
    param([string]$Path)
    $candidate = [System.IO.Path]::GetFullPath($Path)
    if (-not $candidate.StartsWith($asrProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "ASR path left the project." }
    while ($candidate -ne $asrProjectRoot) {
        # Get the link object even if its destination is missing; Test-Path
        # alone may report false for a dangling link.
        $item = Get-AsrPathItem $candidate
        if ($null -ne $item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw "ASR model path contains a reparse point." }
        $candidate = [System.IO.Path]::GetDirectoryName($candidate)
        if ([string]::IsNullOrEmpty($candidate)) { throw "ASR path left the project." }
    }
}

function Test-AsrArtifact {
    param([string]$Path, [long]$ExpectedBytes, [string]$ExpectedHash, [bool]$GitBlob)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -or $item.Length -ne $ExpectedBytes) { return $false }
    if (-not $GitBlob) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $ExpectedHash }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $header = [System.Text.Encoding]::ASCII.GetBytes("blob " + $bytes.Length + [char]0)
    $hasher = [System.Security.Cryptography.SHA1]::Create()
    try {
        $hash = $hasher.ComputeHash($header + $bytes)
        return ([System.BitConverter]::ToString($hash).Replace("-", "").ToLowerInvariant() -eq $ExpectedHash)
    } finally { $hasher.Dispose() }
}

foreach ($source in $asrSources) {
    if ($Backend -ne "both" -and $Backend -ne $source.Name) { continue }
    $modelDir = Join-Path $asrModelRoot $source.Name
    # Reject redirected existing ancestors before creating any directory.
    Assert-AsrPath $modelDir
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    Assert-AsrPath $modelDir
    foreach ($file in @("model.int8.onnx", "tokens.txt")) {
        $tokens = $file -eq "tokens.txt"
        $expectedBytes = if ($tokens) { $source.TokensBytes } else { $source.Bytes }
        $expectedHash = if ($tokens) { $source.TokensBlob } else { $source.Hash }
        $sha256 = if ($tokens) { $source.TokensHash } else { $source.Hash }
        $target = Join-Path $modelDir $file
        Assert-AsrPath $target
        if (Test-AsrArtifact $target $expectedBytes $expectedHash $tokens) {
            Write-Host ($source.Name + "/" + $file + ": verified existing file")
            continue
        }
        if ($null -ne (Get-AsrPathItem $target)) { throw "Existing ASR artifact failed verification; refusing to overwrite it." }
        $partial = Join-Path $modelDir ($file + ".partial")
        Assert-AsrPath $partial
        $partialItem = Get-AsrPathItem $partial
        if ($null -ne $partialItem) {
            if ($partialItem.PSIsContainer -or ($partialItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -or $partialItem.Length -gt $expectedBytes) { throw "Unexpected partial download; refusing to replace it." }
        }
        $url = "https://" + $ModelHost + "/" + $source.Repo + "/resolve/" + $source.Revision + "/" + $file
        Write-Host ($source.Name + "/" + $file + ": downloading " + $expectedBytes + " bytes")
        # One bounded downloader enforces HTTPS and destination allowlists for
        # both weights and tokens. The optional mirror is never trusted over
        # the pinned source hash. Interrupted .partial files stay resumable.
        $downloadArgs = @((Join-Path $PSScriptRoot "download-asr-artifact.py"), "--url", $url, "--target", $target, "--size", [string]$expectedBytes, "--sha256", $sha256)
        & $asrPython @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "ASR preparation failed; partial data retained for inspection/resume." }
        Assert-AsrPath $target
        if (-not (Test-AsrArtifact $target $expectedBytes $expectedHash $tokens)) { throw "Downloaded ASR artifact failed verification; runtime will not use it." }
        Write-Host ($source.Name + "/" + $file + ": verified SHA256 " + (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant())
    }
}
Write-Host "ASR models prepared. No gateway/configuration changes or audio uploads were made."
