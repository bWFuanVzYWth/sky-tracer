//! Pure-sky moving-sun frames: actual SkyView, full-size projection and the
//! demo's existing display shader. No workbench UI or swapchain/vsync is used.
use clap::Args;
use serde::{Deserialize, Serialize};
use sky_realtime::{Config, FrameTimestamps, Renderer, Result, View, Wavelengths};
use std::{fs, path::PathBuf, sync::mpsc, time::Instant};

#[derive(Args)]
pub struct Options {
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    wavelengths: Option<PathBuf>,
    /// Offline-fitted coordinate JSON; immutable for this process.
    #[arg(long)]
    mapping: Option<PathBuf>,
    /// JSON object with a trajectories array; camera stays fixed per trajectory.
    #[arg(long)]
    trajectories: Option<PathBuf>,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    #[arg(long, default_value_t = sky_realtime::DEFAULT_RUNTIME_STEPS)]
    steps: u32,
    #[arg(long, default_value_t = 120)]
    frames: u32,
    #[arg(long, default_value_t = 8)]
    warmup_frames: u32,
    #[arg(long, default_value_t = 3)]
    rounds: u32,
    /// Untimed full-resolution linear/display frames per trajectory, 0 disables.
    #[arg(long, default_value_t = 3)]
    quality_frames: u32,
    /// The demo displays the separately evaluated visible sun disk by default.
    #[arg(long)]
    no_sun_disk: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Trajectory {
    name: String,
    altitude_km: f32,
    yaw_deg: f32,
    pitch_deg: f32,
    fov_y_deg: f32,
    sun_elevation_start_deg: f32,
    sun_elevation_end_deg: f32,
    sun_azimuth_start_deg: f32,
    sun_azimuth_end_deg: f32,
    exposure: f32,
}
impl Trajectory {
    fn view(&self, fraction: f32) -> View {
        View {
            altitude_km: self.altitude_km,
            yaw_deg: self.yaw_deg,
            pitch_deg: self.pitch_deg,
            fov_y_deg: self.fov_y_deg,
            sun_elevation_deg: (self.sun_elevation_start_deg
                + fraction * (self.sun_elevation_end_deg - self.sun_elevation_start_deg))
                .clamp(-90.0, 90.0),
            sun_azimuth_deg: self.sun_azimuth_start_deg
                + fraction * (self.sun_azimuth_end_deg - self.sun_azimuth_start_deg),
        }
    }
    fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self
                .name
                .chars()
                .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            || [
                self.altitude_km,
                self.yaw_deg,
                self.pitch_deg,
                self.fov_y_deg,
                self.sun_elevation_start_deg,
                self.sun_elevation_end_deg,
                self.sun_azimuth_start_deg,
                self.sun_azimuth_end_deg,
                self.exposure,
            ]
            .iter()
            .any(|v| !v.is_finite())
            || self.altitude_km < 0.0
            || self.exposure <= 0.0
            || !(1.0..179.0).contains(&self.fov_y_deg)
            || self.pitch_deg.abs() > 90.0
            || self.sun_elevation_start_deg.abs() > 90.0
            || self.sun_elevation_end_deg.abs() > 90.0
            || (self.sun_elevation_start_deg == self.sun_elevation_end_deg
                && self.sun_azimuth_start_deg == self.sun_azimuth_end_deg)
        {
            return Err("invalid or stationary sun trajectory".into());
        }
        Ok(())
    }
}
fn defaults() -> Vec<Trajectory> {
    let mut result: Vec<_> = [
        ("noon", 0.2, 55.0, 60.0, 75.0, 87.0, 0.1),
        ("sunset_crossing", 0.2, 15.0, 30.0, 1.35, -1.65, 0.5),
        ("blue_shadow", 0.2, 15.0, 30.0, -4.3, -8.7, 30.0),
        ("high_shadow", 30.0, 25.0, 60.0, -5.3, -6.4, 0.5),
    ]
    .into_iter()
    .map(
        |(name, altitude, pitch, fov, start, end, exposure)| Trajectory {
            name: name.into(),
            altitude_km: altitude,
            yaw_deg: 0.0,
            pitch_deg: pitch,
            fov_y_deg: fov,
            sun_elevation_start_deg: start,
            sun_elevation_end_deg: end,
            sun_azimuth_start_deg: -1.0,
            sun_azimuth_end_deg: 1.0,
            exposure,
        },
    )
    .collect();
    result.push(Trajectory {
        name: "azimuth_only".into(),
        altitude_km: 0.2,
        yaw_deg: 0.0,
        pitch_deg: 15.0,
        fov_y_deg: 30.0,
        sun_elevation_start_deg: 1.0,
        sun_elevation_end_deg: 1.0,
        sun_azimuth_start_deg: -6.0,
        sun_azimuth_end_deg: 6.0,
        exposure: 0.5,
    });
    result
}

pub fn run(options: Options) -> Result<()> {
    let out = options.out.clone();
    let existed = out.exists();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pollster::block_on(run_async(options))
    }))
    .unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(format!("profile panicked: {message}").into())
    });
    if let Err(error) = &result {
        if !existed && out.is_dir() {
            let _=fs::write(out.join("failure.json"),serde_json::to_vec_pretty(&serde_json::json!({"error":error.to_string(),"device_reused":false,"completed":false})).unwrap());
        }
    }
    result
}
async fn run_async(options: Options) -> Result<()> {
    if let Some(path) = &options.mapping {
        sky_realtime::mapping::load_calibration_json(&fs::read_to_string(path)?)?;
    }
    if options.out.exists() {
        return Err("profile output exists; choose a new directory".into());
    }
    if !(1..=4096).contains(&options.width)
        || !(1..=4096).contains(&options.height)
        || !(2..=2048).contains(&options.frames)
        || options.warmup_frames > 128
        || !(1..=16).contains(&options.rounds)
        || !(4..=512).contains(&options.steps)
        || options.quality_frames > 2048
    {
        return Err("invalid profile dimensions/counts".into());
    }
    let config: Config = if let Some(path) = &options.config {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Config::balanced()
    };
    config.validate()?;
    let wavelengths: Wavelengths = if let Some(path) = &options.wavelengths {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Wavelengths::optimized_four()
    };
    let trajectories: Vec<Trajectory> = if let Some(path) = &options.trajectories {
        let data: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        serde_json::from_value(data["trajectories"].clone())?
    } else {
        defaults()
    };
    if trajectories.is_empty() || trajectories.len() > 16 {
        return Err("profile needs 1..16 trajectories".into());
    }
    for trajectory in &trajectories {
        trajectory.validate()?;
    }
    if trajectories
        .iter()
        .enumerate()
        .any(|(i, t)| trajectories[..i].iter().any(|p| p.name == t.name))
    {
        return Err("duplicate trajectory names".into());
    }
    fs::create_dir_all(&options.out)?;
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
        return Err("profile requires GPU timestamps inside encoders".into());
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
            label: Some("pure sky moving sun profile"),
            required_features: required,
            required_limits: limits,
            ..Default::default()
        })
        .await?;
    let info = adapter.get_info();
    let shader_checksum = sky_realtime::checksum(sky_realtime::shader_source().as_bytes());
    fs::write(
        options.out.join("sky_shader.wgsl"),
        sky_realtime::shader_source(),
    )?;
    fs::write(options.out.join("display_shader.wgsl"), display_source())?;
    fs::write(
        options.out.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    fs::write(
        options.out.join("wavelengths.json"),
        serde_json::to_vec_pretty(&wavelengths)?,
    )?;
    fs::write(
        options.out.join("mapping_calibration.json"),
        sky_realtime::mapping::CALIBRATION_JSON,
    )?;
    fs::write(
        options.out.join("trajectories.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"trajectories":trajectories}))?,
    )?;
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|v| v.status.success())
        .map(|v| String::from_utf8_lossy(&v.stdout).trim().to_owned());
    fs::write(
        options.out.join("inputs.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"kind":"moving_sun_pure_sky_profile_v1","head":head,
        "config":config,"wavelengths":wavelengths,"mapping_calibration":serde_json::from_str::<serde_json::Value>(sky_realtime::mapping::calibration_json())?,
        "shader_checksum":shader_checksum,"display_shader_checksum":sky_realtime::checksum(display_source().as_bytes()),
        "adapter":{"name":info.name,"backend":format!("{:?}",info.backend),"driver":info.driver,"driver_info":info.driver_info},
        "size":[options.width,options.height],"runtime_steps":options.steps,"frames":options.frames,"warmup_frames":options.warmup_frames,
        "rounds":options.rounds,"include_visible_sun_disk":!options.no_sun_disk,"trajectories":trajectories,
        "scope":"Fresh explicit device. Fixed atmosphere, no medium rebuild during frames. Full-size core projection and existing demo display shader into RGBA8UnormSrgb; no UI or swapchain/vsync. Diagnostic image readback runs after all timed frames."}),
        )?,
    )?;
    let mut renderer = Renderer::new(&device, &model, &wavelengths, 0.18, config)?;
    let startup = renderer.rebuild(&device, &queue)?;
    renderer.resize(&device, [options.width, options.height]);
    renderer.steps = options.steps;
    renderer.include_sun_disk = !options.no_sun_disk;
    let display = DisplayPass::new(
        &device,
        &queue,
        renderer.target_view(),
        [options.width, options.height],
    );
    let mut runs = Vec::new();
    for round in 0..options.rounds {
        for offset in 0..trajectories.len() {
            let trajectory = &trajectories[(offset + round as usize) % trajectories.len()];
            display.set_exposure(&queue, trajectory.exposure);
            let result = time_trajectory(
                &device,
                &queue,
                &mut renderer,
                &display,
                trajectory,
                &options,
            )?;
            eprintln!(
                "profile {} round {}: full GPU {:.4} ms, observer CPU {:.4} ms",
                trajectory.name,
                round + 1,
                result["full_gpu_ms"]["median"].as_f64().unwrap(),
                result["observer_mapping_cpu_ms"]["median"]
                    .as_f64()
                    .unwrap()
            );
            runs.push(
                serde_json::json!({"round":round+1,"trajectory":trajectory.name,"timings":result}),
            );
            fs::write(
                options.out.join("progress.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"completed":runs.len(),"expected":options.rounds as usize*trajectories.len(),"runs":runs}),
                )?,
            )?;
        }
    }
    let mut summaries = Vec::new();
    for trajectory in &trajectories {
        let rows: Vec<_> = runs
            .iter()
            .filter(|r| r["trajectory"].as_str() == Some(&trajectory.name))
            .collect();
        let mut metrics = serde_json::Map::new();
        for metric in [
            "full_gpu_ms",
            "skyview_gpu_ms",
            "projection_gpu_ms",
            "display_gpu_ms",
            "core_gpu_ms",
            "observer_mapping_cpu_ms",
            "observer_update_cpu_ms",
            "core_encode_cpu_ms",
            "encode_submit_cpu_ms",
            "encode_submit_wait_cpu_ms",
        ] {
            let medians: Vec<f64> = rows
                .iter()
                .map(|r| r["timings"][metric]["median"].as_f64().unwrap())
                .collect();
            metrics.insert(metric.into(),serde_json::json!({"median_of_round_medians":percentile(&medians,0.5),"round_medians":medians}));
        }
        summaries.push(serde_json::json!({"trajectory":trajectory.name,"metrics":metrics}));
    }
    let captures = capture_frames(
        &device,
        &queue,
        &mut renderer,
        &display,
        &trajectories,
        &options,
    )?;
    fs::write(
        options.out.join("profile.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"startup":startup,"runs":runs,"summary":summaries,"captures":captures,
        "query_indices":{"skyview":[0,1],"projection":[2,3],"display":[4,5],"full_frame":[6,7]},
        "gpu_scope":"Full span brackets actual sky/projection/display passes. Queue-managed uniform staging writes precede it. Per-pass timestamps distinguish SkyView from full-size projection; display is the same shader used by sky-demo.",
        "cpu_scope":"Observer mapping and observer update including queue writes are independently timed inside renderer. Encode/submit CPU and encode/submit/wait wall are captured before a separate timestamp-resolve submission. Resolve/copy completion is waited outside the frame clock before starting the next frame, and no timing/image buffer mapping is included. Uniform uploads and driver scheduling are included in host wall. One frame in flight, excludes device/pipeline setup and fixed-medium startup.",
        "output_scope":"RGBA8 sRGB display attachment write, no OS presentation or vsync. Linear RGBA32F captures and SDR RGBA8 captures are generated in separate untimed replays."}),
        )?,
    )?;
    Ok(())
}

fn time_trajectory(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    display: &DisplayPass,
    trajectory: &Trajectory,
    options: &Options,
) -> Result<serde_json::Value> {
    let count = options.frames + options.warmup_frames;
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("moving sun stage queries"),
        ty: wgpu::QueryType::Timestamp,
        count: 8,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("profile resolve"),
        size: 64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mapped = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("profile batched timestamps"),
        size: u64::from(count) * 64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut mapping = Vec::new();
    let mut observer_update = Vec::new();
    let mut core_encode = Vec::new();
    let mut encode_submit = Vec::new();
    let mut wall = Vec::new();
    let mut profiles = Vec::new();
    let mut previous = None;
    for frame in 0..count {
        let measured_index = frame as i32 - options.warmup_frames as i32;
        let fraction = measured_index as f32 / (options.frames - 1) as f32;
        let view = trajectory.view(fraction);
        if measured_index >= 0 && previous == Some(view) {
            return Err("trajectory rounds to an unchanged f32 sun direction; cached frames cannot be counted as moving-sun performance".into());
        }
        previous = Some(view);
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moving sun full frame"),
        });
        encoder.write_timestamp(&queries, 6);
        let core_start = Instant::now();
        let profile = renderer.render_profiled(
            queue,
            &mut encoder,
            view,
            Some(FrameTimestamps {
                query_set: &queries,
                sky_view_begin: 0,
                projection_begin: 2,
            }),
        );
        let core_ms = core_start.elapsed().as_secs_f64() * 1000.0;
        if !profile.projected {
            return Err("moving-sun frame unexpectedly reused the complete frame cache".into());
        }
        if !profile.sky_updated {
            // Fresh query sets must have every resolved query initialized.
            // A camera/azimuth-only trajectory can reuse SkyView for every
            // frame, so no actual SkyView pass will ever write this pair.
            // The host keeps its stage duration at zero using sky_updated.
            encoder.write_timestamp(&queries, 0);
            encoder.write_timestamp(&queries, 1);
        }
        display.render(&mut encoder, Some((&queries, 4)));
        encoder.write_timestamp(&queries, 7);
        queue.submit([encoder.finish()]);
        let submitted_ms = start.elapsed().as_secs_f64() * 1000.0;
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        // Resolve and copy *after* the frame's host clock has been stopped.
        let mut resolve = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("untimed query resolve"),
        });
        resolve.resolve_query_set(&queries, 0..8, &resolved, 0);
        resolve.copy_buffer_to_buffer(&resolved, 0, &mapped, u64::from(frame) * 64, 64);
        queue.submit([resolve.finish()]);
        // Drain diagnostic GPU work outside the frame clock. Otherwise a fast
        // next frame (especially azimuth-only) can wait for this previous copy.
        device.poll(wgpu::PollType::wait_indefinitely())?;
        profiles.push(profile);
        if measured_index >= 0 {
            mapping.push(profile.observer_mapping_cpu_ms);
            observer_update.push(profile.observer_update_cpu_ms);
            core_encode.push(core_ms);
            encode_submit.push(submitted_ms);
            wall.push(wall_ms);
        }
    }
    let (tx, rx) = mpsc::channel();
    mapped.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = mapped.get_mapped_range(..);
    let ticks: &[u64] = bytemuck::cast_slice(&bytes);
    let period = f64::from(queue.get_timestamp_period()) * 1e-6;
    let mut sky = Vec::new();
    let mut project = Vec::new();
    let mut present = Vec::new();
    let mut full = Vec::new();
    let mut core = Vec::new();
    for (i, t) in ticks
        .chunks_exact(8)
        .enumerate()
        .skip(options.warmup_frames as usize)
    {
        let duration = |a: usize, b: usize| t[b].saturating_sub(t[a]) as f64 * period;
        let s = if profiles[i].sky_updated {
            duration(0, 1)
        } else {
            0.0
        };
        let p = duration(2, 3);
        let d = duration(4, 5);
        let f = duration(6, 7);
        if f <= 0.0 || ![s, p, d, f].iter().all(|v| v.is_finite()) {
            return Err("invalid GPU profile timestamp".into());
        }
        sky.push(s);
        project.push(p);
        present.push(d);
        full.push(f);
        core.push(s + p);
    }
    drop(bytes);
    mapped.unmap();
    Ok(
        serde_json::json!({"full_gpu_ms":stats(full),"skyview_gpu_ms":stats(sky),"projection_gpu_ms":stats(project),"display_gpu_ms":stats(present),"core_gpu_ms":stats(core),
        "observer_mapping_cpu_ms":stats(mapping),"observer_update_cpu_ms":stats(observer_update),"core_encode_cpu_ms":stats(core_encode),"encode_submit_cpu_ms":stats(encode_submit),"encode_submit_wait_cpu_ms":stats(wall),
        "measured_sky_updates":profiles[options.warmup_frames as usize..].iter().filter(|v|v.sky_updated).count(),"measured_projections":options.frames}),
    )
}
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let x = (v.len() - 1) as f64 * q;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (x - lo as f64)
}
fn stats(values: Vec<f64>) -> serde_json::Value {
    serde_json::json!({"median":percentile(&values,0.5),"p95":percentile(&values,0.95),"min":percentile(&values,0.0),"max":percentile(&values,1.0),"samples":values})
}

fn display_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../../sky-demo/src/shaders/reinhard_gamut.wgsl"),
        include_str!("../../sky-demo/src/shaders/present_texture.wgsl")
    )
}
struct DisplayPass {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}
impl DisplayPass {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        size: [u32; 2],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("existing sky-demo display"),
            source: wgpu::ShaderSource::Wgsl(display_source().into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pure sky display parameters"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let reference = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("unused reference placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let reference_view = reference.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pure sky existing display bindings"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pure sky display group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&reference_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pure sky display pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pure sky SDR display"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pure sky SDR attachment"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let result = Self {
            pipeline,
            group,
            uniform,
            target,
            view,
        };
        result.set_exposure(queue, 0.1);
        result
    }
    fn set_exposure(&self, queue: &wgpu::Queue, exposure: f32) {
        let values = [
            [exposure, 0.0, 0.0, 4.0],
            [0.0, 0.0, 60.0, 16.0 / 9.0],
            [0.0; 4],
            [1.0, 0.0, 1.1, 203.0 / 80.0],
            [1000.0 / 80.0, 0.0, 0.0, 0.0],
        ];
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&values));
    }
    fn render(&self, encoder: &mut wgpu::CommandEncoder, timer: Option<(&wgpu::QuerySet, u32)>) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pure sky necessary display pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: timer.map(|(query_set, i)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            }),
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn capture_frames(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    display: &DisplayPass,
    trajectories: &[Trajectory],
    options: &Options,
) -> Result<Vec<serde_json::Value>> {
    let mut captures = Vec::new();
    if options.quality_frames == 0 {
        return Ok(captures);
    }
    let count = options.quality_frames.min(options.frames);
    for trajectory in trajectories {
        display.set_exposure(queue, trajectory.exposure);
        for i in 0..count {
            let frame = if count == 1 {
                options.frames / 2
            } else {
                i * (options.frames - 1) / (count - 1)
            };
            let view = trajectory.view(frame as f32 / (options.frames - 1) as f32);
            let mut encoder = device.create_command_encoder(&Default::default());
            renderer.render(queue, &mut encoder, view);
            display.render(&mut encoder, None);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let name = format!("{}_f{frame:04}", trajectory.name);
            let linear = sky_realtime::read_texture(device, queue, renderer.target_texture(), 16)?;
            if bytemuck::cast_slice::<u8, f32>(&linear)
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err("nonfinite profile quality frame".into());
            }
            fs::write(options.out.join(format!("{name}.rgba32f")), linear)?;
            let sdr = sky_realtime::read_texture(device, queue, &display.target, 4)?;
            fs::write(options.out.join(format!("{name}.srgba8")), sdr)?;
            captures.push(serde_json::json!({"trajectory":trajectory.name,"frame":frame,"view":{"altitude_km":view.altitude_km,"yaw_deg":view.yaw_deg,"pitch_deg":view.pitch_deg,"fov_y_deg":view.fov_y_deg,"sun_elevation_deg":view.sun_elevation_deg,"sun_azimuth_deg":view.sun_azimuth_deg},"linear":format!("{name}.rgba32f"),"display":format!("{name}.srgba8"),"size":[options.width,options.height],"exposure":trajectory.exposure}));
        }
    }
    Ok(captures)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moving_sun_trajectories_are_distinct_and_keep_fixed_cameras() {
        for t in defaults() {
            t.validate().unwrap();
            let mut previous = None;
            for i in 0..120 {
                let view = t.view(i as f32 / 119.0);
                assert_ne!(previous, Some(view));
                assert_eq!(view.altitude_km, t.altitude_km);
                assert_eq!(view.pitch_deg, t.pitch_deg);
                previous = Some(view);
            }
        }
    }
}
