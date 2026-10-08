//! Explicit GPU QA; run only when the device is available. No window or UI.
#[path = "../src/output.rs"]
mod output;
#[path = "../src/sky_bridge.rs"]
mod sky_bridge;
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    gpu::ProgressiveRenderer,
    transport::TransportSettings,
    vdb,
};
use glam::{DVec3, Vec3};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use wgpu::util::DeviceExt;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
    #[arg(
        long,
        default_value = "assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb"
    )]
    vdb: PathBuf,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Display {
    image: [u32; 4],
    viewport: [f32; 4],
    exposure: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    ambient: [f32; 4],
}
fn sun(elevation: f64, azimuth: f64) -> DVec3 {
    let (s, c) = elevation.to_radians().sin_cos();
    let (a, b) = azimuth.to_radians().sin_cos();
    DVec3::new(a * c, s, b * c)
}
fn floats(d: &wgpu::Device, q: &wgpu::Queue, t: &wgpu::Texture) -> Result<Vec<[f32; 4]>> {
    let b = sky_realtime::read_texture(d, q, t, 16)?;
    decode(&b)
}
fn decode(b: &[u8]) -> Result<Vec<[f32; 4]>> {
    let v: Vec<_> = b
        .chunks_exact(16)
        .map(|b| {
            std::array::from_fn(|k| f32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect();
    if v.iter().flatten().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("nonfinite/negative texture".into());
    }
    Ok(v)
}
fn means(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    film: &wgpu::Buffer,
    size: [u32; 2],
) -> Result<Vec<[f32; 4]>> {
    let count = u64::from(size[0]) * u64::from(size[1]) * 16;
    let b = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mean diagnostic copy"),
        size: count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(film, 0, &b, 0, count);
    q.submit([e.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    b.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let view = b.get_mapped_range(..);
    let pixels = decode(&view)?;
    drop(view);
    b.unmap();
    Ok(pixels)
}
fn mapped(v: f32, ev: f32) -> f32 {
    let t = v / (2.0f32.powf(-ev) + v);
    if t <= 0.0031308 {
        12.92 * t
    } else {
        1.055 * t.powf(1.0 / 2.4) - 0.055
    }
}
fn environment(d: Vec3, table: &[[f32; 4]]) -> [f32; 3] {
    let x = d.x.atan2(d.z).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * 1024.0 - 0.5;
    let y = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI * 512.0 - 0.5;
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let (tx, ty) = (x - x.floor(), y - y.floor());
    let p = |x: i32, y: i32| table[y.clamp(0, 511) as usize * 1024 + x.rem_euclid(1024) as usize];
    std::array::from_fn(|k| {
        (p(ix, iy)[k] * (1.0 - tx) + p(ix + 1, iy)[k] * tx) * (1.0 - ty)
            + (p(ix, iy + 1)[k] * (1.0 - tx) + p(ix + 1, iy + 1)[k] * tx) * ty
    })
}
fn composite(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    film: &wgpu::Buffer,
    sky: &wgpu::TextureView,
    camera: &Camera,
    size: [u32; 2],
    ev: f32,
    path: &Path,
) -> Result<Vec<[f32; 4]>> {
    let (f, r, u) = camera.basis()?;
    let v = |x: DVec3, w: f32| [x.x as f32, x.y as f32, x.z as f32, w];
    let display = Display {
        image: [size[0], size[1], 1280, 720],
        viewport: [0.0, 0.0, 1280.0, 720.0],
        exposure: [ev, 1.0, 1.0, 0.0],
        forward: v(
            f,
            (camera.horizontal_fov_deg.to_radians() * 0.5).tan() as f32,
        ),
        right: v(r, 0.0),
        up: v(u, 0.0),
        ambient: [0.0; 4],
    };
    let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("composite QA"),
        contents: bytemuck::bytes_of(&display),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("actual viewer composite"),
        source: wgpu::ShaderSource::Wgsl(sky_bridge::COMPOSITE_SHADER.into()),
    });
    let p = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("offscreen viewer composite"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let g = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite resources"),
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: film.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(sky),
            },
        ],
    });
    let target = d.create_texture(&wgpu::TextureDescriptor {
        label: Some("composite SDR readback"),
        size: wgpu::Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut e = d.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("actual viewer display"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&p);
        pass.set_bind_group(0, &g, &[]);
        pass.draw(0..3, 0..1);
    }
    q.submit([e.finish()]);
    let pixels = floats(d, q, &target)?;
    let bytes: Vec<u8> = pixels
        .iter()
        .flat_map(|v| {
            v[..3]
                .iter()
                .map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        })
        .collect();
    image::RgbImage::from_raw(1280, 720, bytes)
        .ok_or("image size mismatch")?
        .save(path)?;
    Ok(pixels)
}
fn main() -> Result<()> {
    let args = Args::parse();
    let out = args.out.clone();
    let existed = out.exists();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pollster::block_on(run(args))
    })) {
        Ok(Ok(())) => Ok(()),
        result => {
            let error = match result {
                Ok(Err(e)) => e.to_string(),
                Err(p) => p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or("GPU QA panic".into()),
                _ => unreachable!(),
            };
            if !existed && out.is_dir() {
                fs::write(
                    out.join("fatal.json"),
                    serde_json::to_vec_pretty(
                        &json!({"error":error,"stopped":true,"device_reused":false}),
                    )?,
                )?;
            }
            Err(error.into())
        }
    }
}
async fn run(a: Args) -> Result<()> {
    if a.out.exists() {
        return Err("choose a new output directory".into());
    }
    fs::create_dir_all(&a.out)?;
    let adapter = wgpu::Instance::default()
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 8;
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (d, q) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            required_features: features,
            ..Default::default()
        })
        .await?;
    let camera = Camera::default();
    let mut settings = TransportSettings {
        ground: None,
        ..Default::default()
    };
    let mut sky = sky_bridge::SkyBackground::new(&d, &q, &settings)?;
    let mut cases = Vec::<Value>::new();
    for elevation in [0.2, -0.2, -1.2] {
        settings.sun_direction = sun(elevation, 65.0);
        let mut e = d.create_command_encoder(&Default::default());
        sky.encode(&d, &q, &mut e, &camera, &settings, true)?;
        q.submit([e.finish()]);
        let table = floats(&d, &q, sky.environment_texture())?;
        let sunlight = floats(&d, &q, sky.sunlight_texture())?[0];
        if (elevation > -1.0) != (sunlight[..3].iter().any(|x| *x > 0.0)) {
            return Err("Earth-horizon sunlight visibility failed".into());
        }
        let metadata = sky.metadata().ok_or("missing sky metadata")?;
        if (metadata.altitude_km - 0.917527).abs() > 1e-5 {
            return Err("wrong world-to-altitude datum".into());
        }
        cases.push(json!({"elevation_deg":elevation,"sunlight_rgb":sunlight,"environment_pixels":table.len(),"metadata":metadata}));
    }
    // Asymmetric aureole makes a mirrored camera basis apparent in the image.
    settings.sun_direction = sun(20.0, -75.0);
    let mut e = d.create_command_encoder(&Default::default());
    sky.encode(&d, &q, &mut e, &camera, &settings, true)?;
    q.submit([e.finish()]);
    let table = floats(&d, &q, sky.environment_texture())?;
    let empty = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("empty film"),
        contents: bytemuck::cast_slice(&vec![[0.0f32; 4]; 64 * 36]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pixels = composite(
        &d,
        &q,
        &empty,
        sky.view(),
        &camera,
        [64, 36],
        1.25,
        &a.out.join("empty_sky.png"),
    )?;
    let (f, r, u) = camera.basis()?;
    let (f, r, u) = (f.as_vec3(), r.as_vec3(), u.as_vec3());
    let t = (camera.horizontal_fov_deg.to_radians() * 0.5).tan() as f32;
    let mut alignment_error = 0.0f32;
    for (x, y) in [(64, 64), (640, 360), (1200, 360), (640, 680)] {
        let nx = (x as f32 + 0.5) / 1280.0 * 2.0 - 1.0;
        let ny = (y as f32 + 0.5) / 720.0 * 2.0 - 1.0;
        let ray = (f + r * nx * t - u * ny * t * 36.0 / 64.0).normalize();
        let expected = environment(ray, &table).map(|v| mapped(v, 1.25));
        for k in 0..3 {
            alignment_error = alignment_error.max((pixels[y * 1280 + x][k] - expected[k]).abs());
        }
    }
    if alignment_error > 3e-4 {
        return Err("empty-film camera/exposure mismatch".into());
    }
    let color = [0.25f32, 0.5, 1.0, 1.0];
    let lit = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("already-composed film"),
        contents: bytemuck::cast_slice(&vec![color; 64 * 36]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let pixels = composite(
        &d,
        &q,
        &lit,
        sky.view(),
        &camera,
        [64, 36],
        1.25,
        &a.out.join("film_no_double_sky.png"),
    )?;
    let film_error = (0..3)
        .map(|k| (pixels[0][k] - mapped(color[k], 1.25)).abs())
        .fold(0.0f32, f32::max);
    if film_error > 3e-6 {
        return Err("film sky/exposure was added twice".into());
    }
    let cpu_volume = vdb::load_vdb(&a.vdb, "density")?;
    let volume = cpu_volume.pack_gpu()?;
    let mut reports = Vec::new();
    for (name, size, spp, transparent) in [
        ("transparent", [128, 72], 1, true),
        ("cloud", [64, 36], 4, false),
    ] {
        settings.extinction_scale = if transparent { 0.0 } else { 4.0 };
        settings.sun_direction = TransportSettings::default().sun_direction;
        let mut e = d.create_command_encoder(&Default::default());
        sky.encode(&d, &q, &mut e, &camera, &settings, true)?;
        q.submit([e.finish()]);
        d.poll(wgpu::PollType::wait_indefinitely())?;
        let config = RenderConfig {
            width: size[0],
            height: size[1],
            spp,
            ..Default::default()
        };
        let mut renderer =
            ProgressiveRenderer::new_preview(&d, &q, &volume, &camera, &settings, &config)?;
        renderer.set_environment(&d, &q, Some((sky.view(), sky.sunlight_view())))?;
        let start = Instant::now();
        let mut chunks = Vec::new();
        let mut stopped = None;
        while renderer.sample_count() < spp {
            let chunk = Instant::now();
            let mut e = d.create_command_encoder(&Default::default());
            renderer.encode_work(&d, &q, &mut e, 1)?;
            q.submit([e.finish()]);
            d.poll(wgpu::PollType::wait_indefinitely())?;
            let wall_ms = chunk.elapsed().as_secs_f64() * 1000.0;
            let progress = renderer.read_progress(&d, &q)?;
            let gpu_ms = renderer.read_work_milliseconds(&d, &q)?;
            chunks.push(json!({"wall_ms":wall_ms,"gpu_ms":gpu_ms,"spp":progress.samples_per_pixel,"completed_paths":progress.completed_paths}));
            if wall_ms > 100.0 || gpu_ms.is_some_and(|ms| ms > 100.0) {
                stopped = Some("chunk exceeded100ms; no further tracing submissions");
                break;
            }
            if start.elapsed().as_secs_f64() > 30.0 {
                stopped = Some("30s tracing budget exceeded; partial display only");
                break;
            }
        }
        renderer.read_diagnostics(&d, &q)?.validate()?;
        let samples = means(&d, &q, renderer.film_buffer(), size)?;
        let minimum = samples.iter().map(|v| v[3]).fold(f32::INFINITY, f32::min);
        let maximum = samples.iter().map(|v| v[3]).fold(0.0f32, f32::max);
        let row_minima: Vec<_> = samples
            .chunks(size[0] as usize)
            .map(|row| row.iter().map(|v| v[3]).fold(f32::INFINITY, f32::min))
            .collect();
        if transparent
            && stopped.is_none()
            && (minimum != 1.0 || maximum != 1.0 || chunks.len() != 3)
        {
            return Err(
                "128x72 first full preview sweep was not uniformly sampled in3chunks".into(),
            );
        }
        composite(
            &d,
            &q,
            renderer.film_buffer(),
            sky.view(),
            &camera,
            size,
            0.0,
            &a.out.join(format!("{name}_preview.png")),
        )?;
        let reference_valid = renderer.sample_count() > 0 && !renderer.has_pending_work();
        if reference_valid {
            let film = renderer.read_film(&d, &q)?;
            let record = output::Record {
                source: a.vdb.clone(),
                source_bytes: fs::metadata(&a.vdb)?.len(),
                grid: "density".into(),
                volume_stats: cpu_volume.stats.clone(),
                camera: camera.clone(),
                transport: settings.clone(),
                render: config,
                backend: "gpu-f32".into(),
                adapter: Some(adapter.get_info().name.clone()),
                asset_attribution: Some(
                    "Walt Disney Animation Studios Cloud Data Set; CC BY-SA3.0".into(),
                ),
                environment: sky.metadata(),
            };
            output::save(
                &a.out.join(format!("{name}_reference")),
                &film,
                &record,
                0.0,
            )?;
        }
        reports.push(json!({"name":name,"size":size,"spp":renderer.sample_count(),"reference_valid":reference_valid,"stopped":stopped,"chunks":chunks,"minimum_pixel_spp":minimum,"maximum_pixel_spp":maximum,"row_minimum_spp":row_minima,"environment":sky.metadata()}));
        fs::write(
            a.out.join("report.json"),
            serde_json::to_vec_pretty(
                &json!({"adapter":adapter.get_info().name,"sun_cases":cases,"alignment_error":alignment_error,"film_no_double_sky_error":film_error,"renders":reports}),
            )?,
        )?;
    }
    Ok(())
}
