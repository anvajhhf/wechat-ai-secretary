# Offline developer acceptance fixtures. Never calls a microphone or speaker,
# downloads a voice, or accesses a user's cached recordings.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$asrProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$cases = Get-Content -LiteralPath (Join-Path $asrProjectRoot "tests\fixtures\asr-control-phrases.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$fixtureRoot = Join-Path $asrProjectRoot ("runtime\asr-evaluation\controls-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $fixtureRoot | Out-Null
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice("Microsoft Huihui Desktop")
    foreach ($speechCase in $cases) {
        if ($speechCase.id -notmatch '^[a-z0-9-]+$') { throw "Invalid control fixture name." }
        $synth.SetOutputToWaveFile((Join-Path $fixtureRoot ($speechCase.id + ".wav")))
        $synth.Speak($speechCase.text)
        $synth.SetOutputToNull()
    }
} finally { $synth.Dispose() }
[pscustomobject]@{directory=$fixtureRoot;cases=$cases.Count;voice="Microsoft Huihui Desktop";network_used=$false} | ConvertTo-Json -Compress
