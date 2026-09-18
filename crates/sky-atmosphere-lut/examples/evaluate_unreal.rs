//! Read-only audit of the existing UE-style renderer on the frozen query cameras.
use clap::Parser;
use glam::{Mat4, Vec3, Vec4};
use sky_atmosphere_lut::{Result, solver::GpuBaker};
use sky_unreal_atmosphere_8wave::{
    Gpu, HillaireAtmosphere, NonZeroRenderSize, RenderTargets, Sun, UnrealAtmosphereContext,
    UnrealFrameParams, ViewFrame,
};
use std::{fs, path::PathBuf, sync::mpsc, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/wavelength_search_dataset_v1/queries.json")]
    queries: PathBuf,
    #[arg(long, default_value = "out/realtime_comparison_v1/ue")]
    out: PathBuf,
    /// The old demo defaults to 100 km; 120 isolates this geometry difference.
    #[arg(long, default_value_t = 100.0)]
    thickness_km: f32,
}

fn camera(s: &serde_json::Value) -> ViewFrame {
    let n = |key: &str| s[key].as_f64().unwrap() as f32;
    let (yaw, pitch) = (n("yaw").to_radians(), n("pitch").to_radians());
    let aspect = n("width") / n("height");
    let tan_x = (n("horizontal_fov").to_radians() * 0.5).tan();
    let tan_y = tan_x / aspect;
    let forward = Vec3::new(
        yaw.sin() * pitch.cos(),
        pitch.sin(),
        yaw.cos() * pitch.cos(),
    )
    .normalize();
    let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin()).normalize();
    let up = forward.cross(right).normalize();
    ViewFrame {
        clip_from_world: Mat4::IDENTITY.to_cols_array_2d(),
        world_from_clip: Mat4::IDENTITY.to_cols_array_2d(),
        clip_from_relative_world: Mat4::IDENTITY.to_cols_array_2d(),
        relative_world_from_clip: Mat4::from_cols(
            (right * tan_x).extend(0.0),
            (up * tan_y).extend(0.0),
            Vec4::ZERO,
            forward.extend(1.0),
        )
        .to_cols_array_2d(),
        world_position: [0.0, 0.0, 0.0, 1.0],
        world_forward: forward.extend(0.0).to_array(),
        world_right: right.extend(0.0).to_array(),
        world_up: up.extend(0.0).to_array(),
        view_params: [tan_y, aspect, 0.1, 0.0],
        light_dir: [0.0, 1.0, 0.0, 0.0],
        viewport: [
            n("width"),
            n("height"),
            n("width").recip(),
            n("height").recip(),
        ],
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    if !a.thickness_km.is_finite() || a.thickness_km <= 0.0 {
        return Err("invalid thickness".into());
    }
    fs::create_dir_all(&a.out)?;
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let gpu = GpuBaker::new_with_features(sky_unreal_atmosphere_8wave::REQUIRED_FEATURES)?;
    let (d, q) = (gpu.device(), gpu.queue());
    let now = Instant::now();
    let mut renderer = UnrealAtmosphereContext::new(&Gpu::borrowed(d, q))?;
    let create_ms = now.elapsed().as_secs_f64() * 1000.0;
    // Copy through textureLoad to preserve exact f32 output without changing the renderer API.
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("UE diagnostic readback"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(8, 8)
fn copy(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(input);
    if any(id.xy >= size) { return; }
    output[id.y * size.x + id.x] = textureLoad(input, vec2i(id.xy), 0);
}
"#
            .into(),
        ),
    });
    let copy = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("copy"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut startup = Vec::new();
    let mut images = Vec::new();
    for (index, s) in plan["images"]
        .as_array()
        .ok_or("missing images")?
        .iter()
        .enumerate()
    {
        let n = |key: &str| s[key].as_f64().unwrap() as f32;
        let size =
            NonZeroRenderSize::new(n("width") as u32, n("height") as u32).ok_or("empty image")?;
        let targets = RenderTargets::new(d, size);
        let mut frame = UnrealFrameParams::new(camera(s));
        let se = n("sun_elevation_deg").to_radians();
        frame.sun = Sun {
            sun_to_scene: -Vec3::new(0.0, se.sin(), se.cos()).normalize(),
            // Only the visible disk uses this RGB irradiance. Atmospheric source
            // illumination uses the unchanged eight spectral irradiance constants.
            irradiance_rec2020_w_m2: Vec3::ZERO,
            ..Sun::default()
        };
        frame.atmosphere = HillaireAtmosphere {
            top_radius_m: (6360.0 + a.thickness_km) * 1000.0,
            world_y0_radius_m: (6360.0 + n("altitude_km")) * 1000.0,
            ..HillaireAtmosphere::default()
        };
        if index == 0 {
            // Recompute identical-sized tables; alternate albedo to invalidate
            // the cache, then restore exactly the production value for images.
            for repeat in 0..7 {
                frame.settings.ground_albedo_spectral =
                    [if repeat % 2 == 0 { 0.18 } else { 0.181 }; 4];
                let now = Instant::now();
                let mut e = d.create_command_encoder(&Default::default());
                renderer.update_static_luts(d, q, &mut e, &frame);
                q.submit([e.finish()]);
                d.poll(wgpu::PollType::wait_indefinitely())?;
                startup.push(now.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let bytes = size.width() as u64 * size.height() as u64 * 16;
        let output = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &copy.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(targets.post_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut e = d.create_command_encoder(&Default::default());
        renderer.prepare(d, q, &mut e, &frame);
        renderer.render(&mut e, &targets);
        {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.set_pipeline(&copy);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.width().div_ceil(8), size.height().div_ceil(8), 1);
        }
        e.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        q.submit([e.finish()]);
        let (tx, rx) = mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let mapped = readback.slice(..).get_mapped_range();
        if bytemuck::cast_slice::<u8, f32>(&mapped)
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("nonfinite UE output".into());
        }
        let name = s["name"].as_str().ok_or("name")?;
        fs::write(a.out.join(format!("{name}_sky.f32")), &mapped)?;
        images.push(
            serde_json::json!({"scene":name,"requested_altitude_km":n("altitude_km"),
            "effective_altitude_km":n("altitude_km").clamp(0.001,a.thickness_km-0.001)}),
        );
        eprintln!("rendered {name}");
    }
    fs::write(
        a.out.join("runs.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "thickness_km":a.thickness_km,"pipeline_resource_creation_ms":create_ms,
            "static_lut_encode_submit_wait_ms":startup,"images":images,"queries":a.queries,
            "note":"Existing UE default physics and eight wavelengths; visible disk disabled, full linear Rec.2020. Heights clamp inside atmosphere as in old demo."
        }))?,
    )?;
    Ok(())
}
