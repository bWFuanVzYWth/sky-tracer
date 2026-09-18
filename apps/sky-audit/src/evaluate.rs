//! Fresh solves, stage GPU timings, camera images, and invalidation checks.
use clap::Args;
use sky_realtime::{Config, Renderer, Result, View, Wavelengths, read_texture};
use std::{fs, path::PathBuf, time::Instant};
#[derive(Args)]
pub struct Options {
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    wavelengths: Option<PathBuf>,
    #[arg(long, default_value_t = 96)]
    steps: u32,
    #[arg(long)]
    verify_invalidation: bool,
    #[arg(long)]
    export_source: bool,
    #[arg(long)]
    export_optical: bool,
    #[arg(long, default_value_t = 1.0)]
    aerosol_scale: f32,
}
pub fn run(a: Options) -> Result<()> {
    pollster::block_on(run_async(a))
}
async fn run_async(a: Options) -> Result<()> {
    fs::create_dir_all(&a.out)?;
    let c: Config = if let Some(path) = a.config {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Config::balanced()
    };
    let w: Wavelengths = if let Some(path) = a.wavelengths {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Wavelengths::optimized_four()
    };
    let model = sky_realtime::model::Model::earth_with_aerosol_scale(a.aerosol_scale)?;
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;
    let mut limits = wgpu::Limits::default();
    let available = adapter.limits();
    limits.max_storage_buffer_binding_size = available
        .max_storage_buffer_binding_size
        .min(2 * 1024 * 1024 * 1024 - 256);
    limits.max_buffer_size = available.max_buffer_size.min(2 * 1024 * 1024 * 1024);
    limits.max_texture_dimension_2d = 8192;
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (d, q) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("hybrid experiment"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        })
        .await?;
    eprintln!("adapter: {:?}", adapter.get_info());
    let compile = Instant::now();
    let mut r = Renderer::new(&d, &model, &w, 0.18, c.clone())?;
    let compile_ms = compile.elapsed().as_secs_f32() * 1000.0;
    let solve = r.rebuild(&d, &q)?;
    r.steps = a.steps;
    fs::write(
        a.out.join("solve.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"config":c,"aerosol_scale":a.aerosol_scale,"wavelengths":w,"adapter":adapter.get_info().name,"pipeline_create_ms":compile_ms,"solve":solve,
                "mapping_calibration":serde_json::from_str::<serde_json::Value>(sky_realtime::mapping::CALIBRATION_JSON)?,
                "shader_checksum":sky_realtime::checksum(sky_realtime::shader_source().as_bytes())}),
        )?,
    )?;
    if a.export_source {
        fs::write(a.out.join("source.rgba16f"), r.export_source(&d, &q)?)?;
    }
    if a.export_optical {
        fs::write(a.out.join("optical.rgba16f"), r.export_optical(&d, &q)?)?;
    }
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let mut results = Vec::new();
    for scene in plan["images"].as_array().ok_or("missing image cameras")? {
        let number = |key: &str| scene[key].as_f64().unwrap() as f32;
        let size = [number("width") as u32, number("height") as u32];
        let name = scene["name"].as_str().unwrap();
        let view = View {
            yaw_deg: number("yaw"),
            pitch_deg: number("pitch"),
            fov_y_deg: (2.0
                * ((number("horizontal_fov").to_radians() * 0.5).tan() * size[1] as f32
                    / size[0] as f32)
                    .atan())
            .to_degrees(),
            sun_azimuth_deg: 0.0,
            sun_elevation_deg: number("sun_elevation_deg"),
            altitude_km: number("altitude_km"),
        };
        for (sky, mode) in [(false, "source"), (true, "sky")] {
            r.use_sky_view = sky;
            r.resize(&d, size);
            let now = Instant::now();
            let mut e = d.create_command_encoder(&Default::default());
            r.render(&q, &mut e, view);
            q.submit([e.finish()]);
            d.poll(wgpu::PollType::wait_indefinitely())?;
            let wall_ms = now.elapsed().as_secs_f32() * 1000.0;
            let bytes = read_texture(&d, &q, r.target_texture(), 16)?;
            if bytemuck::cast_slice::<u8, f32>(&bytes)
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(format!("nonfinite image {name}/{mode}").into());
            }
            fs::write(a.out.join(format!("{name}_{mode}.f32")), bytes)?;
            results.push(
                serde_json::json!({"scene":name,"mode":mode,"encode_submit_wait_ms":wall_ms}),
            );
        }
        eprintln!("rendered {name}");
    }
    let mut invalidation = Vec::new();
    if a.verify_invalidation {
        if r.set_medium(&d, &q, &model, &w, 0.18)?.is_some() {
            return Err("unchanged medium incorrectly recomputed".into());
        }
        invalidation
            .push(serde_json::json!({"event":"identical medium","solves":r.stats.medium_solves}));
        let mut changed = model.clone();
        for band in &mut changed.bands {
            for (_, coeff) in &mut band.profile {
                coeff.extinction *= 1.1;
                for x in &mut coeff.scattering {
                    *x *= 1.1;
                }
            }
        }
        if r.set_medium(&d, &q, &changed, &w, 0.18)?.is_none() {
            return Err("changed atmosphere failed to recompute".into());
        }
        invalidation
            .push(serde_json::json!({"event":"density x1.1","solves":r.stats.medium_solves}));
        if r.set_medium(&d, &q, &changed, &w, 0.2)?.is_none() {
            return Err("changed albedo failed to recompute".into());
        }
        invalidation.push(serde_json::json!({"event":"albedo 0.2","solves":r.stats.medium_solves}));
    }
    fs::write(
        a.out.join("runs.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"queries":a.queries,"steps":a.steps,
        "cache_stats":r.stats,"resident_bytes":r.resident_bytes(),"results":results,"invalidation":invalidation,
        "timing_note":"image wall times include CPU encode/submit/wait; solver stages are GPU timestamps"}))?,
    )?;
    Ok(())
}
