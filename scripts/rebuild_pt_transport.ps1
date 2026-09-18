param(
    [string]$Plan = 'out/pt_surface_v2_rebuild_jobs.json',
    [int]$WaitForProcess = 0
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $workspace
$planPath = [IO.Path]::GetFullPath((Join-Path $workspace $Plan))
$jobs = @(Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json)
$logDir = Join-Path $workspace 'out/pt_transport_rebuild_logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$renderer = Join-Path $workspace 'target/release/sky-reference-render.exe'
if ($WaitForProcess -gt 0) {
    Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue
}
$invariant = [Globalization.CultureInfo]::InvariantCulture
for ($i = 0; $i -lt $jobs.Count; $i++) {
    $job = $jobs[$i]
    if ($job.Status -eq 'complete') { continue }
    $directory = [IO.Path]::GetFullPath($job.Directory)
    if (-not $directory.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output outside workspace: $directory"
    }
    $m = $job.OriginalManifest
    # User gives MicroHH priority. Do not start another GPU workload until all
    # MicroHH instances have exited; an already running render can finish.
    while (@(Get-Process -Name microhh,mircohh -ErrorAction SilentlyContinue).Count -gt 0) {
        Write-Output 'GPU work paused for MicroHH.'
        Start-Sleep -Seconds 30
    }
    $arguments = @('--out', $directory, '--width', [string]$m.dimensions[0], '--height', [string]$m.dimensions[1],
        '--spp', [string]$m.spp, '--seed', [string]$m.seed,
        ('--sun-elevation-deg=' + ([single]$m.sun_elevation_deg).ToString('R', $invariant)),
        ('--sun-azimuth-deg=' + ([single]$m.sun_azimuth_deg).ToString('R', $invariant)),
        ('--observer-altitude-km=' + ([single]$m.observer_altitude_km).ToString('R', $invariant)))
    if ($m.kind -eq 'spectral_sky_view_lut_v0') { $arguments += '--sky-view-lut' }
    $job.Status = 'rendering'
    $jobs | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $planPath -Encoding utf8
    Write-Output "Rendering $($i + 1)/$($jobs.Count): $directory"
    & $renderer @arguments *> (Join-Path $logDir ('asset_{0:D2}.log' -f $i))
    if ($LASTEXITCODE -ne 0) { throw "Renderer failed for $directory; inspect its log and resume this script" }
    $result = Get-Content -LiteralPath (Join-Path $directory 'asset.json') -Raw | ConvertFrom-Json
    if ($result.transport_version -ne 'wgpu-layered-surface-v2' -or $result.band_centers_nm.Count -ne 41) {
        throw "Unexpected transport version or spectrum in $directory"
    }
    foreach ($relative in @($result.files.rgb_exr, $result.files.rgb_png) + $result.files.band_exrs) {
        $file = Join-Path $directory $relative
        if (-not (Test-Path -LiteralPath $file) -or (Get-Item -LiteralPath $file).Length -eq 0) {
            throw "Missing rendered file: $file"
        }
    }
    $job.Status = 'complete'
    $jobs | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $planPath -Encoding utf8
    Write-Output "Verified $($i + 1)/$($jobs.Count): $directory"
}
Write-Output 'All old PT assets have been recomputed and verified.'
