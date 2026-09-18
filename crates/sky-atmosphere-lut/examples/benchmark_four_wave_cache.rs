use clap::Parser;
use sky_atmosphere_lut::{
    Result, four_wave::cached::CachedRenderer, renderer::View, solver::GpuBaker,
};
use std::sync::mpsc;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/four_wave_cache_v1/benchmark.json")]
    out: std::path::PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let gpu = GpuBaker::new_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )?;
    let d = gpu.device();
    let q = gpu.queue();
    let mut renderer = CachedRenderer::new(d, std::path::Path::new("out/four_wave_source_v1"))?;
    renderer.include_sun_disk = true;
    let queries = d.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("cache benchmark"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let original = View {
        yaw_deg: 0.0,
        pitch_deg: 85.0,
        fov_y_deg: 12.0,
        sun_azimuth_deg: 0.0,
        sun_elevation_deg: 85.0,
        altitude_km: 0.2,
    };
    let mut rows = Vec::new();
    for size in [[512, 384], [1920, 1080], [3840, 2160]] {
        renderer.resize(d, size);
        for workload in [
            "camera",
            "sun_azimuth",
            "sun_elevation",
            "altitude",
            "unchanged",
        ] {
            let mut times = Vec::new();
            let before = renderer.stats;
            for i in 0..16 {
                let mut view = original;
                match workload {
                    "camera" => view.yaw_deg += i as f32 * 0.01,
                    "sun_azimuth" => view.sun_azimuth_deg += i as f32 * 0.01,
                    "sun_elevation" => view.sun_elevation_deg += i as f32 * 0.01,
                    "altitude" => view.altitude_km += i as f32 * 0.001,
                    _ => {}
                }
                let mut encoder = d.create_command_encoder(&Default::default());
                encoder.write_timestamp(&queries, 0);
                renderer.render(q, &mut encoder, view);
                encoder.write_timestamp(&queries, 1);
                encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
                encoder.copy_buffer_to_buffer(&resolved, 0, &readback, 0, 16);
                q.submit([encoder.finish()]);
                let (tx, rx) = mpsc::channel();
                readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                d.poll(wgpu::PollType::wait_indefinitely())?;
                rx.recv()??;
                let bytes = readback.slice(..).get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&bytes);
                if i >= 4 {
                    times.push((ticks[1] - ticks[0]) as f32 * q.get_timestamp_period() * 1e-6);
                }
                drop(bytes);
                readback.unmap();
            }
            times.sort_by(f32::total_cmp);
            let median = times[times.len() / 2];
            let p95 = times[times.len() * 95 / 100];
            println!(
                "{}x{} {workload}: median {median:.4} ms, P95 {p95:.4} ms",
                size[0], size[1]
            );
            rows.push(serde_json::json!({"size":size,"workload":workload,"gpu_median_ms":median,"gpu_p95_ms":p95,"samples":times.len(),"stats_before":before,"stats_after":renderer.stats}));
        }
    }
    assert_eq!(renderer.stats.ms_builds, 1);
    std::fs::write(
        args.out,
        serde_json::to_vec_pretty(
            &serde_json::json!({"adapter":gpu.adapter_name,"resident_bytes":renderer.resident_lut_bytes,"rows":rows,"scope":"GPU compute only, four warmup frames then 12 samples; immutable MS initialization excluded by warmup; no frame readback/presentation"}),
        )?,
    )?;
    Ok(())
}
