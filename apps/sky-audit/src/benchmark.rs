//! Alternating configuration timing with explicit startup/runtime scopes.
use clap::Args;
use sky_realtime::{Config, Renderer, Result, View, Wavelengths};
use std::{fs, path::PathBuf, sync::mpsc, time::Instant};

#[derive(Args)]
pub struct Options {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, num_args = 1..)]
    configs: Vec<PathBuf>,
    #[arg(long)]
    wavelengths: Option<PathBuf>,
    /// Offline-fitted coordinate JSON; immutable for this process.
    #[arg(long)]
    mapping: Option<PathBuf>,
    #[arg(long, default_value_t = 6)]
    startup_repeats: u32,
    #[arg(long, default_value_t = 3)]
    runtime_rounds: u32,
    #[arg(long, default_value_t = 4)]
    warmup_frames: u32,
    #[arg(long, default_value_t = 64)]
    frames: u32,
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 384)]
    height: u32,
    #[arg(long, default_value_t = sky_realtime::DEFAULT_RUNTIME_STEPS)]
    steps: u32,
}

pub fn run(options: Options) -> Result<()> {
    pollster::block_on(run_async(options))
}

async fn run_async(options: Options) -> Result<()> {
    if let Some(path) = &options.mapping {
        sky_realtime::mapping::load_calibration_json(&fs::read_to_string(path)?)?;
    }
    if options.out.exists() {
        return Err("benchmark output exists; choose a new file".into());
    }
    if options.configs.is_empty()
        || options.configs.len() > 16
        || !(1..=32).contains(&options.startup_repeats)
        || !(1..=16).contains(&options.runtime_rounds)
        || options.warmup_frames > 64
        || !(1..=256).contains(&options.frames)
        || !(1..=4096).contains(&options.width)
        || !(1..=4096).contains(&options.height)
        || !(4..=512).contains(&options.steps)
    {
        return Err("invalid benchmark configuration/counts".into());
    }
    let candidates: Vec<_> = options
        .configs
        .iter()
        .map(|path| {
            let config: Config = serde_json::from_slice(&fs::read(path)?)?;
            config.validate()?;
            Ok((
                path.file_stem()
                    .ok_or("config has no file stem")?
                    .to_string_lossy()
                    .into_owned(),
                path.clone(),
                config,
            ))
        })
        .collect::<Result<_>>()?;
    let wavelengths = if let Some(path) = &options.wavelengths {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Wavelengths::optimized_four()
    };
    let model = sky_realtime::model::Model::earth()?;
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;
    let required =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !adapter.features().contains(required) {
        return Err("benchmark requires GPU timestamps inside command encoders".into());
    }
    let available = adapter.limits();
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = available
        .max_storage_buffer_binding_size
        .min(2 * 1024 * 1024 * 1024 - 256);
    limits.max_buffer_size = available.max_buffer_size.min(2 * 1024 * 1024 * 1024);
    limits.max_texture_dimension_2d = 8192;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("realtime configuration benchmark"),
            required_features: required,
            required_limits: limits,
            ..Default::default()
        })
        .await?;
    let mut startup = Vec::new();
    let mut runtime = Vec::new();
    // One discarded sweep warms every pipeline/configuration. Subsequent sweeps
    // rotate the starting configuration, with only one live renderer at a time.
    for round in 0..=options.startup_repeats {
        for offset in 0..candidates.len() {
            let index = (offset + round as usize) % candidates.len();
            let (name, _, config) = &candidates[index];
            let mut renderer = Renderer::new(&device, &model, &wavelengths, 0.18, config.clone())?;
            let report = renderer.rebuild(&device, &queue)?;
            if round > 0 {
                let mut stages = vec![0.0f64; report.stages.len()];
                for iteration in &report.iterations {
                    for (sum, value) in stages.iter_mut().zip(&iteration.stage_gpu_ms) {
                        *sum += f64::from(*value);
                    }
                }
                startup.push(
                    serde_json::json!({"config":name,"round":round,"report":report,
                    "stage_gpu_ms":stages,"stage_total_gpu_ms":stages.iter().sum::<f64>()}),
                );
                eprintln!("startup {name} round {round}/{}", options.startup_repeats);
            }
        }
    }
    for round in 0..options.runtime_rounds {
        for offset in 0..candidates.len() {
            let index = (offset + round as usize) % candidates.len();
            let (name, _, config) = &candidates[index];
            let mut renderer = Renderer::new(&device, &model, &wavelengths, 0.18, config.clone())?;
            renderer.rebuild(&device, &queue)?;
            renderer.resize(&device, [options.width, options.height]);
            renderer.steps = options.steps;
            for (scene, altitude, sun) in [
                ("noon", 0.2, 85.0),
                ("sunset", 0.2, 0.0),
                ("blue", 0.2, -6.0),
            ] {
                let base = View {
                    yaw_deg: 0.0,
                    pitch_deg: 4.0,
                    fov_y_deg: 60.0,
                    sun_azimuth_deg: 0.0,
                    sun_elevation_deg: sun,
                    altitude_km: altitude,
                };
                for workload in [
                    "direct_source",
                    "sun_change",
                    "height_change",
                    "camera_change",
                    "cached_frame",
                ] {
                    let values =
                        time_workload(&device, &queue, &mut renderer, base, workload, &options)?;
                    runtime.push(
                        serde_json::json!({"config":name,"round":round+1,"scene":scene,
                        "workload":workload,"gpu_ms":values.0,"encode_submit_wait_ms":values.1,
                        "gpu_median_ms":median(&values.0),"gpu_p95_ms":percentile(&values.0,0.95)}),
                    );
                }
            }
            eprintln!(
                "runtime {name} round {}/{}",
                round + 1,
                options.runtime_rounds
            );
        }
    }
    let mut summary = Vec::new();
    for (name, _, _) in &candidates {
        let rows: Vec<_> = startup
            .iter()
            .filter(|row| row["config"].as_str() == Some(name))
            .collect();
        let total: Vec<_> = rows
            .iter()
            .map(|row| row["stage_total_gpu_ms"].as_f64().unwrap())
            .collect();
        let mut stage_medians = Vec::new();
        for stage in 0..4 {
            let times: Vec<_> = rows
                .iter()
                .map(|row| row["stage_gpu_ms"][stage].as_f64().unwrap())
                .collect();
            stage_medians.push(median(&times));
        }
        let mut workloads = Vec::new();
        for scene in ["noon", "sunset", "blue"] {
            for workload in [
                "direct_source",
                "sun_change",
                "height_change",
                "camera_change",
                "cached_frame",
            ] {
                let medians: Vec<_> = runtime
                    .iter()
                    .filter(|row| {
                        row["config"].as_str() == Some(name)
                            && row["scene"].as_str() == Some(scene)
                            && row["workload"].as_str() == Some(workload)
                    })
                    .map(|row| row["gpu_median_ms"].as_f64().unwrap())
                    .collect();
                workloads.push(serde_json::json!({"scene":scene,"workload":workload,
                    "median_of_round_medians_ms":median(&medians),"round_medians_ms":medians}));
            }
        }
        summary.push(
            serde_json::json!({"config":name,"startup_stage_total_gpu_median_ms":median(&total),
            "startup_stage_medians_ms":stage_medians,"runtime":workloads}),
        );
    }
    let info = adapter.get_info();
    let result = serde_json::json!({"adapter":{"name":info.name,"backend":format!("{:?}",info.backend),
        "driver":info.driver,"driver_info":info.driver_info},"configs":candidates.iter().map(|(name,path,config)|
        serde_json::json!({"name":name,"path":path,"config":config})).collect::<Vec<_>>(),
        "wavelengths":wavelengths,"mapping_calibration":serde_json::from_str::<serde_json::Value>(sky_realtime::mapping::calibration_json())?,
        "shader_checksum":sky_realtime::checksum(sky_realtime::shader_source().as_bytes()),
        "dimensions":[options.width,options.height],"steps":options.steps,"warmup_frames":options.warmup_frames,
        "frames":options.frames,"startup_repeats":options.startup_repeats,"runtime_rounds":options.runtime_rounds,
        "startup_scope":"GPU timestamps of trace_incident/incident_moments/scattering_source/ground_irradiance summed across all iterations; excludes optical_depth and prepare_directions, device/pipeline creation and readback. SolveReport wall time separately includes encode/submit/wait and timing readback.",
        "runtime_scope":"GPU timestamps around realtime core rendering only; no presentation, device/pipeline creation, reference readback or timing readback. Four warmup frames per workload; each frame is submitted/waited independently. Configurations rotate between rounds.",
        "startup":startup,"runtime":runtime,"summary":summary});
    if let Some(parent) = options.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.out, serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}

fn time_workload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    base: View,
    workload: &str,
    options: &Options,
) -> Result<(Vec<f64>, Vec<f64>)> {
    renderer.use_sky_view = workload != "direct_source";
    let count = options.warmup_frames + options.frames;
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("realtime runtime timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("runtime resolved"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mapped = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("runtime timing samples"),
        size: u64::from(count) * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut wall = Vec::new();
    for frame in 0..count {
        let mut view = base;
        match workload {
            "direct_source" | "camera_change" => view.yaw_deg += frame as f32 * 0.01,
            "sun_change" => view.sun_elevation_deg += frame as f32 * 0.01,
            "height_change" => view.altitude_km += frame as f32 * 0.001,
            "cached_frame" => (),
            _ => return Err("unknown runtime workload".into()),
        }
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.write_timestamp(&queries, 0);
        renderer.render(queue, &mut encoder, view);
        encoder.write_timestamp(&queries, 1);
        encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
        encoder.copy_buffer_to_buffer(&resolved, 0, &mapped, u64::from(frame) * 16, 16);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely())?;
        if frame >= options.warmup_frames {
            wall.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let (tx, rx) = mpsc::channel();
    mapped.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = mapped.get_mapped_range(..);
    let ticks: &[u64] = bytemuck::cast_slice(&bytes);
    let gpu = ticks
        .chunks_exact(2)
        .skip(options.warmup_frames as usize)
        .map(|ticks| {
            ticks[1].saturating_sub(ticks[0]) as f64
                * f64::from(queue.get_timestamp_period())
                * 1e-6
        })
        .collect();
    drop(bytes);
    mapped.unmap();
    Ok((gpu, wall))
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() as f64 * p) as usize).min(values.len() - 1)]
}
fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}
