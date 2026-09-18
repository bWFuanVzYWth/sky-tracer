$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\WorkSpace\sky_tracer'
$cases = @('balanced', 'balanced_q40', 'balanced_i12', 'dense_check')
foreach ($case in $cases) {
    & .\target\release\examples\evaluate.exe --out "out/hybrid_v1/$case" --queries out/four_wave_sun_v1/queries.json --config "out/hybrid_v1/configs/$case.json"
    if ($LASTEXITCODE -ne 0) { throw "hybrid case failed: $case" }
}
