param(
    [string]$OutRoot = "out_skyview_search",
    [int]$Width = 256,
    [int]$Height = 256,
    [int]$Spp = 4096,
    [float]$SunAzimuthDeg = 0.0,
    [float]$ObserverAltitudeKm = 0.2,
    [int]$DirectLightSamples = 1
)

$ErrorActionPreference = "Stop"

$elevations = @(-2, -1, 0, 1, 2, 5, 10, 20, 45, 90)

foreach ($elevation in $elevations) {
    $label = if ($elevation -lt 0) {
        "elev_m{0:D2}" -f [Math]::Abs($elevation)
    } else {
        "elev_{0:D3}" -f $elevation
    }
    $outDir = Join-Path $OutRoot $label
    Write-Host "Baking sky-view LUT elevation=$elevation deg -> $outDir"
    cargo run -p sky-bake --bin sky-bake --release -- `
        --sky-view-lut `
        --width $Width `
        --height $Height `
        --spp $Spp `
        "--sun-elevation-deg=$elevation" `
        "--sun-azimuth-deg=$SunAzimuthDeg" `
        "--observer-altitude-km=$ObserverAltitudeKm" `
        --direct-light-samples $DirectLightSamples `
        --out $outDir
    if ($LASTEXITCODE -ne 0) {
        throw "sky-bake failed for elevation=$elevation with exit code $LASTEXITCODE"
    }
}
