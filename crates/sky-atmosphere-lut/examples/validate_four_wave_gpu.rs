//! Small GPU integration/step-convergence audit; does not run a reference bake.
use clap::Parser;
use sky_atmosphere_lut::four_wave::cached::CachedRenderer;
use sky_atmosphere_lut::{Result, four_wave::renderer::Renderer, renderer::View, solver::GpuBaker};
use std::{fs, path::PathBuf, sync::mpsc, time::Instant};
#[derive(Parser)]
struct Args {
    resource: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "32,64,128")]
    steps: Vec<u32>,
    #[arg(long)]
    no_ms: bool,
    #[arg(long, value_enum, default_value = "reference")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "finite16")]
    sun_model: sky_atmosphere_lut::four_wave::solar::SunModel,
}
#[derive(Clone, Copy, clap::ValueEnum)]
enum Mode {
    Reference,
    Source,
    Sky,
}
enum Engine {
    Reference(Renderer),
    Cached(CachedRenderer),
}
impl Engine {
    fn resize(&mut self, d: &wgpu::Device, size: [u32; 2]) {
        match self {
            Self::Reference(r) => {
                r.resize(d, size);
            }
            Self::Cached(r) => {
                r.resize(d, size);
            }
        }
    }
    fn steps(&mut self, n: u32) {
        match self {
            Self::Reference(r) => r.steps = n,
            Self::Cached(r) => r.steps = n,
        }
    }
    fn render(&mut self, q: &wgpu::Queue, e: &mut wgpu::CommandEncoder, v: View) {
        match self {
            Self::Reference(r) => r.render(q, e, v),
            Self::Cached(r) => r.render(q, e, v),
        }
    }
    fn target_texture(&self) -> &wgpu::Texture {
        match self {
            Self::Reference(r) => r.target_texture(),
            Self::Cached(r) => r.target_texture(),
        }
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    fs::create_dir_all(&a.out)?;
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let gpu = GpuBaker::new()?;
    let device = gpu.device();
    let queue = gpu.queue();
    let mut renderer = if matches!(a.mode, Mode::Reference) {
        let mut r = Renderer::new_with_sun_model(device, &a.resource, a.sun_model)?;
        r.multiple_scattering = !a.no_ms;
        Engine::Reference(r)
    } else {
        let mut r = CachedRenderer::new_with_sun_model(device, &a.resource, a.sun_model)?;
        r.multiple_scattering = !a.no_ms;
        r.use_sky_view = matches!(a.mode, Mode::Sky);
        Engine::Cached(r)
    };
    let mut results = Vec::new();
    for scene in plan["images"].as_array().ok_or("missing image cameras")? {
        let width = scene["width"].as_u64().ok_or("width")? as u32;
        let height = scene["height"].as_u64().ok_or("height")? as u32;
        let number = |key: &str| scene[key].as_f64().unwrap() as f32;
        let view = View {
            yaw_deg: number("yaw"),
            pitch_deg: number("pitch"),
            fov_y_deg: (2.0
                * ((number("horizontal_fov").to_radians() * 0.5).tan() * height as f32
                    / width as f32)
                    .atan())
            .to_degrees(),
            sun_azimuth_deg: 0.0,
            sun_elevation_deg: number("sun_elevation_deg"),
            altitude_km: number("altitude_km"),
        };
        let name = scene["name"].as_str().unwrap();
        renderer.resize(device, [width, height]);
        let row = (width * 16).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("four-wave audit"),
            size: (row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for &steps in &a.steps {
            renderer.steps(steps);
            let now = Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            renderer.render(queue, &mut encoder, view);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: renderer.target_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let elapsed = now.elapsed().as_secs_f32() * 1000.0;
            let (tx, rx) = mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely())?;
            rx.recv()??;
            let bytes = readback.slice(..).get_mapped_range();
            let mut rgba = Vec::new();
            for y in 0..height {
                rgba.extend_from_slice(&bytes[(y * row) as usize..(y * row + width * 16) as usize]);
            }
            fs::write(a.out.join(format!("{name}_{steps}.f32")), rgba)?;
            drop(bytes);
            readback.unmap();
            eprintln!("{name}, {steps} steps: {elapsed:.1} ms encode/submit/wait");
            results.push(
                serde_json::json!({"scene":name,"steps":steps,"encode_submit_wait_ms":elapsed}),
            );
        }
    }
    let (bytes, stats) = match &renderer {
        Engine::Reference(r) => (r.resource.payload_bytes as u64, serde_json::Value::Null),
        Engine::Cached(r) => (r.resident_lut_bytes, serde_json::to_value(r.stats)?),
    };
    fs::write(
        a.out.join("runs.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"resource":a.resource,"sun_model":a.sun_model,"payload_bytes":bytes,"cache_stats":stats,"results":results,"timing_note":"single cold frame per setting, includes CPU encode/submit/wait; not a steady-state GPU benchmark"}),
        )?,
    )?;
    Ok(())
}
