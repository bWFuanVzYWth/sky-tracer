//! Timing ablations, not quality presets. Does not change the demo shader.
use sky_atmosphere_lut::{
    Result,
    four_wave::{
        cached::{CachedRenderer, shader_source},
        solar::{SunModel, shader_for_model},
    },
    renderer::View,
    solver::GpuBaker,
};
use std::{path::Path, sync::mpsc};

fn main() -> Result<()> {
    let gpu = GpuBaker::new_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )?;
    let d = gpu.device();
    let q = gpu.queue();
    let queries = d.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("cost ablation"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut results = Vec::new();
    for (sun_model, shader) in [
        ("finite_16", shader_source()),
        (
            "point_1",
            shader_for_model(&shader_source(), SunModel::Parallel),
        ),
        (
            "phase_averaged",
            shader_for_model(&shader_source(), SunModel::PhaseAveraged),
        ),
        (
            "finite_4",
            shader_for_model(&shader_source(), SunModel::Finite4),
        ),
    ] {
        let mut renderer = CachedRenderer::new_with_shader_for_diagnostics(
            d,
            Path::new("out/four_wave_source_v1"),
            &shader,
        )?;
        renderer.resize(d, [512, 384]);
        renderer.include_sun_disk = true;
        for (scene, elevation) in [("noon", 85.0), ("blue", -6.0)] {
            for (steps, multiple) in [(128, true), (128, false), (64, true), (32, true)] {
                renderer.steps = steps;
                renderer.multiple_scattering = multiple;
                let before = renderer.stats;
                let mut times = Vec::new();
                for i in 0..28 {
                    let view = View {
                        yaw_deg: 0.0,
                        pitch_deg: 15.0,
                        fov_y_deg: 90.0,
                        sun_azimuth_deg: 0.0,
                        sun_elevation_deg: elevation + i as f32 * 0.001,
                        altitude_km: 0.2,
                    };
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
                    let mapped = readback.slice(..).get_mapped_range();
                    let timestamps: &[u64] = bytemuck::cast_slice(&mapped);
                    if i >= 4 {
                        times.push(
                            (timestamps[1] - timestamps[0]) as f32
                                * q.get_timestamp_period()
                                * 1e-6,
                        );
                    }
                    drop(mapped);
                    readback.unmap();
                }
                times.sort_by(f32::total_cmp);
                let median = times[times.len() / 2];
                println!("{scene} {sun_model}, {steps} steps, MS {multiple}: {median:.4} ms");
                assert_eq!(renderer.stats.ms_builds, 1);
                assert_eq!(renderer.stats.sky_updates - before.sky_updates, 28);
                results.push(serde_json::json!({
                    "scene":scene,"sun_model":sun_model,"steps":steps,"multiple_scattering":multiple,
                    "gpu_median_ms":median,"gpu_p95_ms":times[times.len()*95/100],
                    "samples":times.len(),"cache_stats":renderer.stats,
                }));
            }
        }
    }
    std::fs::write(
        "out/four_wave_sun_v1/cost_ablation.json",
        serde_json::to_vec_pretty(&serde_json::json!({
            "adapter":gpu.adapter_name,"sky_size":[256,256],"screen_size":[512,384],"results":results,
            "scope":"GPU SkyView update plus projection, no presentation; 4 warmup + 24 samples, initialization excluded",
            "warning":"Point-Sun and no-MS variants are timing diagnostics, not quality-validated runtime modes."
        }))?,
    )?;
    Ok(())
}
