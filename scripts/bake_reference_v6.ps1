param(
    [string]$Baker = 'out/lut_v6_run/baker.exe',
    [string]$Config = 'crates/sky-atmosphere-lut/configs/reference-v6-candidate.json',
    [string]$OutputDir = 'out/lut_reference_v6',
    [string]$RgbDir = 'out/lut_reference_v6_rgb',
    [string]$RunDir = 'out/lut_v6_run'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$bakerPath = (Resolve-Path -LiteralPath $Baker).Path
$configPath = (Resolve-Path -LiteralPath $Config).Path
$null = New-Item -ItemType Directory -Path $RunDir -Force
$runPath = (Resolve-Path -LiteralPath $RunDir).Path
$logPath = Join-Path $runPath 'bake.log'
$statePath = Join-Path $runPath 'status.json'
$started = [DateTime]::UtcNow.ToString('o')
$binaryHash = (Get-FileHash -LiteralPath $bakerPath -Algorithm SHA256).Hash
$configHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
$stage = 'starting'

function Write-RunState([string]$status, [string]$detail) {
    $state = [ordered]@{
        status = $status; stage = $script:stage; detail = $detail
        pid = $PID; started_utc = $script:started; updated_utc = [DateTime]::UtcNow.ToString('o')
        source = $OutputDir; rgb = $RgbDir; config = $script:configPath
        baker_sha256 = $script:binaryHash; config_sha256 = $script:configHash
        accepted_reference = $false
    }
    $temporary = $script:statePath + '.tmp'
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $script:statePath -Force
}

function Invoke-Baker([string]$label, [string[]]$bakerArguments) {
    $script:stage = $label
    Write-RunState 'running' ''
    & $script:bakerPath @bakerArguments 2>&1 | Out-File -LiteralPath $script:logPath -Encoding utf8 -Append
    if ($LASTEXITCODE -ne 0) { throw "$label exited with code $LASTEXITCODE" }
}

try {
    $bakeArguments = @('bake', '--config', $configPath, '--out', $OutputDir)
    if (Test-Path -LiteralPath (Join-Path $OutputDir 'asset.json')) { $bakeArguments += '--resume' }
    Invoke-Baker 'baking' $bakeArguments
    Invoke-Baker 'verifying_spectral' @('inspect', $OutputDir, '--verify')
    $manifest = Get-Content -LiteralPath (Join-Path $OutputDir 'asset.json') -Raw | ConvertFrom-Json
    $capped = @()
    for ($i = 0; $i -lt $manifest.records.Count; $i++) {
        $record = $manifest.records[$i]
        if ($null -eq $record) { throw "band $i was not committed" }
        if (-not $record.stopped_by_tolerance) {
            $capped += [ordered]@{ index = $i; nm = $manifest.bands[$i].center_nm; orders = $record.orders.Count }
        }
    }
    [ordered]@{ bands = $manifest.records.Count; capped_bands = $capped; accepted_reference = $false } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runPath 'convergence.json') -Encoding utf8
    # A new output directory prevents an interrupted export from silently replacing data.
    Invoke-Baker 'exporting_rgb_cpu' @('export-rgb', $OutputDir, '--out', $RgbDir)
    Invoke-Baker 'verifying_rgb_cpu' @('inspect', $RgbDir, '--verify')
    $script:stage = 'finished'
    Write-RunState 'complete' 'Spectral and Rec.2020 resources verified. Quality acceptance and image comparisons remain pending.'
    exit 0
} catch {
    Write-RunState 'failed' $_.Exception.Message
    $_ | Out-File -LiteralPath $logPath -Encoding utf8 -Append
    exit 1
}
