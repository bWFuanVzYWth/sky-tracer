$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\WorkSpace\sky_tracer'
& .\target\release\examples\evaluate.exe --out out/hybrid_v1/balanced_final --queries out/four_wave_sun_v1/queries.json --config out/hybrid_v1/configs/balanced_q40.json
if ($LASTEXITCODE -ne 0) { throw 'balanced final failed' }
& .\target\release\examples\evaluate.exe --out out/hybrid_v1/dense_final --queries out/four_wave_sun_v1/queries.json --config out/hybrid_v1/configs/dense48.json
if ($LASTEXITCODE -ne 0) { throw 'dense final failed' }
& .\target\release\examples\validate.exe
if ($LASTEXITCODE -ne 0) { throw 'invariants failed' }
& .\target\release\sky-realtime-demo.exe --experiment hybrid-4d --asset out/pt_noon_085/asset.json --lut-budget-mib 16 --snapshot out/hybrid_v1/demo_noon.png --snapshot-linear --snapshot-yaw-deg 0 --snapshot-pitch-deg 85 --snapshot-fov-deg 12 --benchmark-frames 24
if ($LASTEXITCODE -ne 0) { throw 'demo validation failed' }
