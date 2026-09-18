//! Actual GPU cache invalidation and finite-segment transport composition.
use clap::Args;
use glam::Vec4;
use sky_realtime::{Config, Renderer, Result, View, Wavelengths, read_texture};
use sky_reference::{mapping::State, solver::GpuBaker};
use std::path::PathBuf;
#[derive(Args)]
pub struct Options {
    #[arg(long, default_value = "out/validation/invariants.json")]
    out: PathBuf,
}
fn frame(gpu: &GpuBaker, r: &mut Renderer, v: View) -> Result<()> {
    let mut e = gpu.device().create_command_encoder(&Default::default());
    r.render(gpu.queue(), &mut e, v);
    gpu.queue().submit([e.finish()]);
    gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
    Ok(())
}
fn segment(gpu: &GpuBaker, r: &mut Renderer, s: State, length: f32) -> Result<(Vec4, Vec4)> {
    let den = ((1.0 - s.mu * s.mu) * (1.0 - s.mu_s * s.mu_s))
        .max(0.0)
        .sqrt();
    let az = ((s.nu - s.mu * s.mu_s) / den.max(1e-10))
        .clamp(-1.0, 1.0)
        .acos();
    r.segment_km = length;
    frame(
        gpu,
        r,
        View {
            yaw_deg: 0.0,
            pitch_deg: s.mu.asin().to_degrees(),
            fov_y_deg: 1.0,
            sun_azimuth_deg: az.to_degrees(),
            sun_elevation_deg: s.mu_s.asin().to_degrees(),
            altitude_km: s.altitude_km,
        },
    )?;
    let l = read_texture(gpu.device(), gpu.queue(), r.target_texture(), 16)?;
    let t = read_texture(gpu.device(), gpu.queue(), r.transmittance_texture(), 16)?;
    Ok((
        Vec4::from_slice(bytemuck::cast_slice(&l)),
        Vec4::from_slice(bytemuck::cast_slice(&t)),
    ))
}
pub fn run(args: Options) -> Result<()> {
    let model = sky_realtime::model::Model::earth()?;
    let wavelengths = Wavelengths::optimized_four();
    let gpu = GpuBaker::new()?;
    let d = gpu.device();
    let q = gpu.queue();
    let mut r = Renderer::new(d, &model, &wavelengths, 0.18, Config::balanced())?;
    let solve = r.rebuild(d, q)?;
    assert!(solve.resident_bytes <= 16 * 1024 * 1024);
    let mut view = View {
        yaw_deg: 0.0,
        pitch_deg: 20.0,
        fov_y_deg: 90.0,
        sun_azimuth_deg: 0.0,
        sun_elevation_deg: 47.0,
        altitude_km: 0.2,
    };
    r.resize(d, [32, 16]);
    let mut events = Vec::new();
    for (name, expected) in [
        ("first", 1),
        ("unchanged", 1),
        ("yaw", 1),
        ("solar azimuth", 1),
        ("solar elevation", 2),
        ("altitude", 3),
        ("resize", 3),
        ("disable MS", 4),
    ] {
        match name {
            "yaw" => view.yaw_deg = 5.0,
            "solar azimuth" => view.sun_azimuth_deg = 10.0,
            "solar elevation" => view.sun_elevation_deg = 20.0,
            "altitude" => view.altitude_km = 12.0,
            "resize" => {
                r.resize(d, [16, 16]);
            }
            "disable MS" => r.multiple_scattering = false,
            _ => {}
        }
        frame(&gpu, &mut r, view)?;
        assert_eq!(r.stats.medium_solves, 1);
        assert_eq!(r.stats.sky_updates, expected);
        events.push(serde_json::json!({"event":name,"stats":r.stats}));
    }
    assert!(r.set_medium(d, q, &model, &wavelengths, 0.18)?.is_none());
    let original = r.export_source(d, q)?;
    let changed = sky_realtime::model::Model::earth_with_aerosol_scale(1.1)?;
    assert!(r.set_medium(d, q, &changed, &wavelengths, 0.18)?.is_some());
    let modified = r.export_source(d, q)?;
    assert_ne!(original, modified);
    assert_eq!(r.stats.medium_solves, 2);
    assert!(r.set_medium(d, q, &changed, &wavelengths, 0.2)?.is_some());
    assert_ne!(modified, r.export_source(d, q)?);
    assert_eq!(r.stats.medium_solves, 3);
    r.use_sky_view = false;
    r.spectral_output = true;
    r.multiple_scattering = true;
    r.steps = 512;
    r.resize(d, [1, 1]);
    let g = sky_reference::mapping::Geometry {
        bottom: model.geometry.bottom,
        top: model.geometry.top,
    };
    let mut segments = Vec::new();
    for (name, h, e, sun, azimuth, split, total) in [
        ("day", 0.2, 5.0_f32, 47.0_f32, 0.0_f32, 10.0, 50.0),
        ("blue", 0.2, 5.0, -6.0, 0.0, 10.0, 50.0),
        ("limb", 108.0, -0.5, 0.0, 0.0, 100.0, 300.0),
        ("ground", 12.0, -30.0, 20.0, 0.0, 10.0, 0.0),
        ("offaxis_shadow", 12.0, -2.0, -3.3, 90.0, 50.0, 300.0),
        ("offaxis_high", 30.0, -3.0, -5.0, 35.0, 100.0, 400.0),
    ] {
        let mu = e.to_radians().sin();
        let mu_s = sun.to_radians().sin();
        let s = State {
            altitude_km: h,
            mu,
            mu_s,
            nu: mu * mu_s
                + (1.0 - mu * mu).sqrt() * (1.0 - mu_s * mu_s).sqrt() * azimuth.to_radians().cos(),
            ground: g.hits_ground(h, mu),
        };
        let (whole, t) = segment(&gpu, &mut r, s, total)?;
        let (first, tfirst) = segment(&gpu, &mut r, s, split)?;
        let (second, tsecond) = segment(
            &gpu,
            &mut r,
            g.advanced(s, split),
            if total > 0.0 { total - split } else { 0.0 },
        )?;
        let error = (whole - first - tfirst * second).length() / whole.length().max(1e-20);
        let terror = (t - tfirst * tsecond).abs().max_element();
        assert!(error < 0.005, "{name}: segment L error {error}");
        assert!(terror < 0.001, "{name}: segment T error {terror}");
        segments.push(serde_json::json!({"name":name,"radiance_relative_percent":100.0*error,"transmittance_absolute":terror}));
    }
    let (light, t) = segment(
        &gpu,
        &mut r,
        State {
            altitude_km: 400.0,
            mu: 0.5,
            mu_s: 0.0,
            nu: 0.5,
            ground: false,
        },
        100.0,
    )?;
    assert_eq!(light, Vec4::ZERO);
    assert_eq!(t, Vec4::ONE);
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        args.out,
        serde_json::to_vec_pretty(&serde_json::json!({"cache_events":events,
        "physical_changes_rebuild":true,"changed_medium_changes_actual_source":true,"medium_solves":r.stats.medium_solves,"segments":segments,
        "vacuum":"L=0,T=1","resident_bytes":r.resident_bytes(),"peak_payload_bytes":solve.peak_payload_bytes}))?,
    )?;
    println!(
        "hybrid cache, medium rebuild, vacuum, and six spectral segment composition checks passed"
    );
    Ok(())
}
