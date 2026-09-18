$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\WorkSpace\sky_tracer'
$cases = @('warp_baseline', 'warp_candidate', 'compact24', 'compact40', 'compact40_q16', 'compact40_q32', 'warp_oracle')
foreach ($case in $cases) {
    & .\target\release\examples\evaluate.exe --out "out/hybrid_v1/$case" --queries out/four_wave_sun_v1/queries.json --config "out/hybrid_v1/configs/$case.json"
    if ($LASTEXITCODE -ne 0) { throw "hybrid case failed: $case" }
}
