# sky-hybrid-atmosphere

Experimental four-wavelength GPU atmosphere solver, independent of the frozen
reference radiance resource. It shares physical profile types and coordinate
utilities with `sky-atmosphere-lut`, but owns its transport, iteration, 4D source,
SkyView, and projection pipelines. No reference asset or offline SH is loaded.

The four wavelengths are 450, 510, 580, and 650 nm. Transport and accumulation
use f32. The normalized lower-atmosphere source texture uses RGBA16F; brightness
scales, Rayleigh moments, and SkyView use f32. Illumination is a parallel beam;
the visible solar disk remains a separate display term.

Each fixed-point iteration traces the incident field using the previous local
MS source and diffuse ground boundary, integrates the actual species phase
functions, and updates the source and ground. The 4D source includes at least
one previous volume scatter or ground reflection. Direct single scattering is
integrated separately during SkyView generation and finite-segment queries.

Optimizations currently used:

- Four spectral lanes share geometry, extinction sampling, and dispatches.
- Reflection symmetry halves incident ray integrations.
- Phase-table coordinates are computed once per reflection direction and shared
  by all four aerosol species. Physical source-height coefficients are cached.
- Above 35 km, the calibrated preset assigns the incident integral to the
  horizon charts; the Sun chart smoothly fades between 12 and 35 km. Zero-weight
  rays are skipped. This improves shadow convergence as well as startup cost.
- Rayleigh convolution uses second moments; above the 35 km aerosol boundary,
  the retained angular representation is exact for Rayleigh convolution.
- A height/Sun-dependent quadrature splits the horizon and concentrates samples
  near the Sun and horizon. Discrete phase-energy correction preserves the
  independently integrated mass of the tabulated phase function.
- The lower source uses hardware angular interpolation and separately
  interpolated logarithmic brightness scales.
- Fitted height, solar, phase, and cone coordinates allocate samples by measured
  interpolation error. Solar lookup uses precomputed softsign coefficients;
  the cone coordinate is a cubic polynomial. Optical height rows coincide with
  physical profile boundaries. The 160 solar nodes share coordinates with the
  log mean and ground boundary tables.
- SkyView has separate fitted lower/upper/ground/space charts, plus a smoothly
  blended near-ground chart. Row inversion is cached on the CPU; projection
  uses rational CDFs. Space reuses all 192 sky rows for the atmospheric shell.
- The exact high-altitude Rayleigh quadratic occupies five RGBA32F texels per
  state instead of seven, without truncating its angular representation.
- Incident radiance, directions, and moments use a four-height scratch batch.
  All batches read the same previous iteration and write disjoint next-source
  rows, preserving the solve while avoiding full-height scratch allocation.
- All startup work buffers and the spare iteration field are dropped after
  solving. Observer/Sun/camera changes reuse the fixed-medium solution.

`Config::default()` is the original fast, coarse experiment. The demo selects
`Config::balanced()`: 40 heights, 160 solar states, 20 × 12 outgoing directions,
32 × 16 quadrature on each of three charts (one reflection half), 64 ray steps,
8 iterations. Optical depth uses 128 × 512 texels and 256 integration steps.
SkyView stays 256² RGBA32F; runtime integration defaults to 96 steps.
The same azimuth rule is used at every height, without material-layer step
insertion or a separate high-altitude quadrature algorithm.

Six paired measurements on an RTX 4090 give a 0.393 s median solve, versus
1.617 s for the preceding preset. Resident payload is 7.910 MiB including
SkyView; peak solve payload is 44.194 MiB. Device/pipeline creation, driver
allocation, and output targets are additional. At 512 × 384, measured demo
Sun-update GPU time is 0.239 ms, including presentation; camera-only changes
take 0.013 ms. These timings are device- and workload-specific.

This preset accepts small quality losses for performance. Against denser
convergence probes, the worst per-view P95 on 112 near-ground grazing views
is 1.16% (previously 0.92%); the 313-view general/space set reaches 2.42%
(previously 0.99%). These are convergence differences, not physical truth
errors. Full results, aerosol stress tests, remaining limits, and rejected
optimizations are in [the performance budget report](../../reports/lut-performance-budget.md).
The earlier [mapping](../../reports/lut-fitted-mapping-quality.md) and
[importance](../../reports/lut-importance-and-performance.md) reports record
historical configurations. The experimental single-scattering cache and
special quadrature/layer-step options have been removed. Pipeline
specialization remains on.
`mapping_flags = 0` reproduces legacy coordinates for old experiment JSONs;
the balanced preset enables all six mapping groups (`63`). The diagnostic
`sky_size` can change independently of the fixed-medium solution dimensions.

```rust,ignore
let mut sky = Renderer::new(device, &model, &Wavelengths::optimized_four(), 0.18,
                            Config::balanced())?;
sky.rebuild(device, queue)?;
// Later: same inputs return None; physical changes solve the whole field again.
sky.set_medium(device, queue, &updated_model, &wavelengths, albedo)?;
sky.resize(device, [width, height]);
sky.render(queue, encoder, view);
```

Call `resize` after changing output size or switching segment/spectral modes.
Positive `segment_km` or `spectral_output` automatically bypasses SkyView.
Spectral queries return four-lane radiance and transmittance for
`L_ab = L_ac + T_ac L_cb`. RGB background reconstruction, scene occlusion and a
froxel cache are not implemented. The API currently requires zero aerosols at
and above 35 km. Calibration and quality measurements use the original medium
and gray albedo 0.18; other media need their own quality/convergence checks.

Run from the workspace root:

```powershell
cargo run --release -p sky-realtime-demo -- --experiment hybrid-4d --asset out/pt_noon_085/asset.json --lut-budget-mib 8
cargo run --release -p sky-hybrid-atmosphere --example evaluate -- --out out/hybrid_new --queries out/four_wave_sun_v1/queries.json --config crates/sky-hybrid-atmosphere/configs/balanced.json
cargo run --release -p sky-hybrid-atmosphere --example validate
```

The demo exposes an explicit Apply button for aerosol density and gray ground
albedo, so dragging a slider does not start a solve every frame. Camera movement
only projects the existing SkyView; solar elevation and altitude regenerate it.

See `reports/lut-hybrid-and-full-sh.md` for measured comparisons, limitations,
and the separate full-radiance spherical-harmonic experiment.
