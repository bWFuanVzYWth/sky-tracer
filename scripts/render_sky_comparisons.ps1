param(
    [string]$Plan = 'scripts/sky_comparison_cases.json',
    [string]$Lut = 'out/lut_reference_v4_rgb',
    [string]$OutputDir = 'out/lut_validation_v4',
    [string[]]$Solvers = @('offline-lut','unreal-8wave'),
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $workspace
$jobsPath = Join-Path $workspace $Plan
$jobs = @(Get-Content -LiteralPath $jobsPath -Raw | ConvertFrom-Json)
$demo = Join-Path $workspace 'target/release/sky-realtime-demo.exe'
$lutManifest = Join-Path (Join-Path $workspace $Lut) 'asset.json'
$outputRoot = Join-Path $workspace $OutputDir
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$invariant = [Globalization.CultureInfo]::InvariantCulture
foreach ($job in $jobs) {
    $manifestPath = Join-Path $workspace $job.Asset
    if (-not (Test-Path -LiteralPath $manifestPath)) {continue}
    $reference = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($reference.transport_version -ne 'wgpu-layered-surface-v2') {continue}
    $imagePath = Join-Path (Split-Path $manifestPath) $reference.files.rgb_exr
    if (-not (Test-Path -LiteralPath $imagePath)) {continue}
    foreach ($solver in $Solvers) {
        if ($solver -notin @('offline-lut','unreal-8wave')) {throw "Unknown solver: $solver"}
        $prefix = if ($solver -eq 'offline-lut') {'lut'} else {'ue'}
        $output = Join-Path $outputRoot ($prefix + '_' + $job.Name + '.f32')
        $metadataPath = [IO.Path]::ChangeExtension($output, 'json')
        $previewPath = [IO.Path]::ChangeExtension($output, 'png')
        if (-not $Force -and (Test-Path -LiteralPath $output) -and (Test-Path -LiteralPath $previewPath) -and (Test-Path -LiteralPath $metadataPath)) {
            $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
            $pose = $metadata.view_yaw_pitch_fov_exposure
            $fresh = (Get-Item -LiteralPath $output).LastWriteTimeUtc -gt (Get-Item -LiteralPath $manifestPath).LastWriteTimeUtc -and
                (Get-Item -LiteralPath $output).LastWriteTimeUtc -gt (Get-Item -LiteralPath $imagePath).LastWriteTimeUtc -and
                (Get-Item -LiteralPath $output).LastWriteTimeUtc -gt (Get-Item -LiteralPath $lutManifest).LastWriteTimeUtc -and
                (Get-Item -LiteralPath $output).LastWriteTimeUtc -gt (Get-Item -LiteralPath $demo).LastWriteTimeUtc
            if ($metadata.comparison_pipeline -eq 'pt-uv-inverse-v2' -and $metadata.asset -eq $manifestPath -and $metadata.lut -eq $Lut -and $fresh -and
                $pose[0] -eq $job.Yaw -and $pose[1] -eq $job.Pitch -and $pose[2] -eq $job.Fov -and $pose[3] -eq $job.Exposure) {continue}
        }
        if (@(Get-Process -Name microhh,mircohh -ErrorAction SilentlyContinue).Count -gt 0) {
            Write-Output 'MicroHH running; remaining comparisons deferred.'
            exit 0
        }
        Write-Output "Rendering $prefix $($job.Name)"
        $arguments = @('--asset',$manifestPath,'--lut',$Lut,'--experiment',$solver,
            '--snapshot',$output,'--snapshot-linear',
            ('--snapshot-yaw-deg=' + ([single]$job.Yaw).ToString('R',$invariant)),
            ('--snapshot-pitch-deg=' + ([single]$job.Pitch).ToString('R',$invariant)),
            ('--snapshot-fov-deg=' + ([single]$job.Fov).ToString('R',$invariant)),
            ('--snapshot-exposure-ev=' + ([single]$job.Exposure).ToString('R',$invariant)))
        & $demo @arguments *> ($output + '.log')
        if ($LASTEXITCODE -ne 0) {throw "Snapshot failed: $output"}
    }
}
