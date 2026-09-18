//! Finite-segment composition check using spectral radiance and transmittance.
use clap::Parser;
use glam::Vec4;
use sky_atmosphere_lut::{
    Result, four_wave::renderer::Renderer, mapping::State, renderer::View, solver::GpuBaker,
};
use std::{path::Path, sync::mpsc};
#[derive(Parser)]
struct Args {
    #[arg(long, value_enum, default_value = "finite16")]
    sun_model: sky_atmosphere_lut::four_wave::solar::SunModel,
    #[arg(long, default_value = "out/four_wave_validation_v1/segments.json")]
    out: std::path::PathBuf,
}

fn read(gpu: &GpuBaker, renderer: &mut Renderer, s: State, length: f32) -> Result<(Vec4, Vec4)> {
    let denominator = ((1.0 - s.mu * s.mu) * (1.0 - s.mu_s * s.mu_s))
        .max(0.0)
        .sqrt();
    let azimuth = ((s.nu - s.mu * s.mu_s) / denominator.max(1e-10))
        .clamp(-1.0, 1.0)
        .acos();
    let view = View {
        yaw_deg: 0.0,
        pitch_deg: s.mu.asin().to_degrees(),
        fov_y_deg: 1.0,
        sun_azimuth_deg: azimuth.to_degrees(),
        sun_elevation_deg: s.mu_s.asin().to_degrees(),
        altitude_km: s.altitude_km,
    };
    renderer.segment_km = length;
    let device = gpu.device();
    let queue = gpu.queue();
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("segment validation"),
        size: 512,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.render(queue, &mut encoder, view);
    for (i, texture) in [renderer.target_texture(), renderer.transmittance_texture()]
        .iter()
        .enumerate()
    {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: (i * 256) as u64,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = readback.slice(..).get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&bytes);
    Ok((
        Vec4::from_slice(&floats[..4]),
        Vec4::from_slice(&floats[64..68]),
    ))
}
fn main() -> Result<()> {
    let args = Args::parse();
    let gpu = GpuBaker::new()?;
    let mut renderer = Renderer::new_with_sun_model(
        gpu.device(),
        Path::new("out/four_wave_source_v1"),
        args.sun_model,
    )?;
    renderer.spectral_output = true;
    renderer.steps = 512;
    let g = renderer.resource.geometry;
    let mut rows = Vec::new();
    for (name, h, view_deg, sun_deg, azimuth, split, total) in [
        ("day", 0.2, 5.0_f32, 47.0_f32, 0.0_f32, 10.0, 50.0),
        ("blue", 0.2, 5.0, -6.0, 0.0, 10.0, 50.0),
        ("limb", 108.0, -0.5, 0.0, 0.0, 100.0, 300.0),
        ("ground", 12.0, -30.0, 20.0, 0.0, 10.0, 0.0),
        ("offaxis_shadow", 12.0, -2.0, -3.3, 90.0, 50.0, 300.0),
        ("offaxis_high", 30.0, -3.0, -5.0, 35.0, 100.0, 400.0),
    ] {
        let mu = view_deg.to_radians().sin();
        let mu_s = sun_deg.to_radians().sin();
        let s = State {
            altitude_km: h,
            mu,
            mu_s,
            nu: mu * mu_s
                + (1.0 - mu * mu).sqrt() * (1.0 - mu_s * mu_s).sqrt() * azimuth.to_radians().cos(),
            ground: g.hits_ground(h, mu),
        };
        let (whole, t) = read(&gpu, &mut renderer, s, total)?;
        let (first, tfirst) = read(&gpu, &mut renderer, s, split)?;
        let second_state = g.advanced(s, split);
        let (second, tsecond) = read(
            &gpu,
            &mut renderer,
            second_state,
            if total > 0.0 { total - split } else { 0.0 },
        )?;
        let combined = first + tfirst * second;
        let radiance_error = (whole - combined).length() / whole.length().max(1e-20);
        let trans_error = (t - tfirst * tsecond).abs().max_element();
        assert!(
            radiance_error < 0.005,
            "{name}: segment radiance error {radiance_error}"
        );
        assert!(
            trans_error < 0.001,
            "{name}: segment transmittance error {trans_error}"
        );
        rows.push(serde_json::json!({"name":name,"relative_radiance_percent":radiance_error*100.0,"absolute_transmittance_error":trans_error}));
    }
    let vacuum = State {
        altitude_km: 400.0,
        mu: 0.5,
        mu_s: 0.0,
        nu: 0.5,
        ground: false,
    };
    let (l, t) = read(&gpu, &mut renderer, vacuum, 100.0)?;
    assert_eq!(l, Vec4::ZERO);
    assert_eq!(t, Vec4::ONE);
    let output = serde_json::json!({"cases":rows,"sun_model":args.sun_model,"vacuum":"L=0,T=1","steps":512,"note":"four spectral lanes, includes ground and non-coplanar endpoints; not an RGB fog transmittance approximation"});
    std::fs::write(args.out, serde_json::to_vec_pretty(&output)?)?;
    println!("{output}");
    Ok(())
}
