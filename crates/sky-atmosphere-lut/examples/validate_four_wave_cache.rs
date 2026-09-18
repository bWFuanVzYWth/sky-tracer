//! Check dependency invalidation with actual GPU execution, not just key tests.
use sky_atmosphere_lut::{
    Result, four_wave::cached::CachedRenderer, renderer::View, solver::GpuBaker,
};
fn main() -> Result<()> {
    let gpu = GpuBaker::new()?;
    let d = gpu.device();
    let q = gpu.queue();
    let mut r = CachedRenderer::new(d, std::path::Path::new("out/four_wave_source_v1"))?;
    r.resize(d, [96, 64]);
    let mut v = View {
        yaw_deg: 0.0,
        pitch_deg: 15.0,
        fov_y_deg: 90.0,
        sun_azimuth_deg: 0.0,
        sun_elevation_deg: 85.0,
        altitude_km: 0.2,
    };
    let draw = |r: &mut CachedRenderer, v: View| -> Result<()> {
        let mut e = d.create_command_encoder(&Default::default());
        r.render(q, &mut e, v);
        q.submit([e.finish()]);
        d.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    };
    let mut rows = Vec::new();
    for (event, expected_sky, expected_projection) in [
        ("initial", 1, 1),
        ("unchanged", 1, 1),
        ("camera", 1, 2),
        ("sun_azimuth", 1, 3),
        ("resize", 1, 4),
        ("altitude", 2, 5),
        ("sun_elevation", 3, 6),
        ("steps", 4, 7),
        ("multiple_toggle", 5, 8),
        ("unchanged_after_toggle", 5, 8),
    ] {
        match event {
            "camera" => {
                v.yaw_deg = 40.0;
                v.pitch_deg = 25.0;
                v.fov_y_deg = 70.0;
            }
            "sun_azimuth" => v.sun_azimuth_deg = 37.0,
            "resize" => {
                r.resize(d, [192, 128]);
            }
            "altitude" => v.altitude_km = 12.0,
            "sun_elevation" => v.sun_elevation_deg = -7.0,
            "steps" => r.steps = 64,
            "multiple_toggle" => r.multiple_scattering = false,
            _ => {}
        }
        draw(&mut r, v)?;
        assert_eq!(r.stats.ms_builds, 1, "{event}");
        assert_eq!(r.stats.sky_updates, expected_sky, "{event}");
        assert_eq!(r.stats.projections, expected_projection, "{event}");
        rows.push(serde_json::json!({"event":event,"stats":r.stats}));
    }
    assert!(r.resident_lut_bytes <= 16 * 1024 * 1024);
    std::fs::create_dir_all("out/four_wave_cache_v1")?;
    std::fs::write(
        "out/four_wave_cache_v1/invalidation.json",
        serde_json::to_vec_pretty(&rows)?,
    )?;
    println!(
        "10 GPU invalidation cases passed; source builds = 1; resident = {} bytes",
        r.resident_lut_bytes
    );
    Ok(())
}
