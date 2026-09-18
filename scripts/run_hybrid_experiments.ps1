$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\WorkSpace\sky_tracer'
$cases = @('iter4', 'iter6', 'iter12', 'height48', 'sun177', 'grid48x177', 'angles24x32', 'ray128', 'candidate', 'oracle')
foreach ($case in $cases) {
    & .\target\release\examples\evaluate.exe --out "out/hybrid_v1/$case" --queries out/four_wave_sun_v1/queries.json --config "out/hybrid_v1/configs/$case.json"
    if ($LASTEXITCODE -ne 0) { throw "hybrid case failed: $case" }
}
