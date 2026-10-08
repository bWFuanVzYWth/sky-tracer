//! Explicit, tiny GPU continuation QA. This executable never opens a window.
//! Usage: bounded_smoke WIDTH HEIGHT SPP NEW_OUTPUT_DIRECTORY.
#[path = "../src/app.rs"]
#[allow(dead_code)]
mod app;
#[path = "../src/output.rs"]
#[allow(dead_code)]
mod output;

use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    gpu::{MAX_WORK_PATHS, ProgressiveRenderer, WORK_PROGRESS_BYTES, WORK_TRANSITIONS},
    transport::TransportSettings,
    vdb,
};
use std::{fs, path::PathBuf, sync::mpsc, time::Instant};

fn read_progress_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &ProgressiveRenderer,
) -> Result<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bounded QA progress"),
        size: WORK_PROGRESS_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        renderer.progress_buffer(),
        0,
        &staging,
        0,
        WORK_PROGRESS_BYTES,
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = staging.get_mapped_range(..).to_vec();
    staging.unmap();
    Ok(bytes)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: bounded_smoke WIDTH HEIGHT SPP NEW_OUTPUT_DIRECTORY".into());
    }
    let config = RenderConfig {
        width: args[0].parse()?,
        height: args[1].parse()?,
        spp: args[2].parse()?,
        ..Default::default()
    };
    if config.width > 8 || config.height > 4 || config.spp > 32 {
        return Err("bounded smoke intentionally limited to8x4x32".into());
    }
    let out = PathBuf::from(&args[3]);
    if out.exists() {
        return Err("smoke output already exists".into());
    }
    fs::create_dir_all(&out)?;
    let source = PathBuf::from("assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb");
    let volume = vdb::load_vdb(&source, "density")?;
    let packed = volume.pack_gpu()?;
    let camera = Camera::default();
    let settings = TransportSettings::default();
    let (device, queue, adapter) = pollster::block_on(app::create_device(None))?;
    let mut renderer =
        ProgressiveRenderer::new(&device, &queue, &packed, &camera, &settings, &config)?;
    let count = renderer.sample_batch_capacity().min(config.spp);
    // Exercise reset rejection in flight, partial-export rejection, duplicate
    // completion, and reset/stale epoch before tracing the actual reference.
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.encode_work(&device, &queue, &mut encoder, count)?;
    assert!(renderer.reset(&device, &queue, &camera, &settings).is_err());
    queue.submit([encoder.finish()]);
    let old = read_progress_bytes(&device, &queue, &renderer)?;
    let first = renderer.complete_work(&old)?;
    if !first.batch_finished {
        assert!(renderer.read_film(&device, &queue).is_err());
    }
    assert!(renderer.complete_work(&old).is_err());
    let warm_ms = renderer
        .read_work_milliseconds(&device, &queue)?
        .unwrap_or(0.0);
    if warm_ms > 100.0 {
        return Err(format!("first bounded chunk exceeds100ms: {warm_ms}").into());
    }
    renderer.reset(&device, &queue, &camera, &settings)?;
    assert_eq!(renderer.sample_count(), 0);
    assert!(!renderer.has_pending_work());
    let mut times = Vec::new();
    let mut stale_checked = false;
    let trace_start = Instant::now();
    while renderer.sample_count() < config.spp {
        let count = renderer
            .sample_batch_capacity()
            .min(config.spp - renderer.sample_count());
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.encode_work(&device, &queue, &mut encoder, count)?;
        if !stale_checked {
            assert!(renderer.complete_work(&old).is_err());
            stale_checked = true;
        }
        queue.submit([encoder.finish()]);
        renderer.read_progress(&device, &queue)?;
        if let Some(ms) = renderer.read_work_milliseconds(&device, &queue)? {
            times.push(ms);
            if times.len() % 128 == 0 || renderer.sample_count() == config.spp || ms > 100.0 {
                fs::write(
                    out.join("progress.json"),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"chunks":times.len(),"samples":renderer.sample_count(),"max_chunk_ms":times.iter().copied().reduce(f64::max)}),
                    )?,
                )?;
            }
            if ms > 100.0 {
                return Err(format!("bounded chunk exceeds100ms; no new submission: {ms}").into());
            }
        }
    }
    let film = renderer.read_film(&device, &queue)?;
    let trace_wall_seconds = trace_start.elapsed().as_secs_f64();
    output::save(
        &out.join("reference"),
        &film,
        &output::Record {
            source: source.clone(),
            source_bytes: fs::metadata(source)?.len(),
            grid: "density".into(),
            volume_stats: volume.stats,
            camera,
            transport: settings,
            render: config,
            backend: "gpu-f32".into(),
            adapter: Some(adapter),
            asset_attribution: Some("WDAS Cloud Data Set, CC BY-SA3.0".into()),
        },
        0.0,
    )?;
    fs::write(
        out.join("qa.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"complete":true,"chunks":times.len(),"max_gpu_chunk_ms":times.iter().copied().reduce(f64::max),"gpu_chunk_ms":times,"initial_discarded_reset_chunk_ms":warm_ms,"inflight_reset_rejected":true,"partial_export_rejected":!first.batch_finished,"duplicate_progress_rejected":true,"old_epoch_progress_rejected":stale_checked,"samples_per_pixel":film.samples_per_pixel,"paths_dropped":0,"max_work_paths":MAX_WORK_PATHS,"transitions_per_path_per_chunk":WORK_TRANSITIONS,"trace_wall_seconds":trace_wall_seconds,"wall_scope":"Reference tracing, progress/timestamp readbacks, periodic checkpoint writing and final film readback; excludes setup/reset warmup and EXR export"}),
        )?,
    )?;
    println!(
        "bounded smoke complete: {} chunks, max {:.6} ms",
        times.len(),
        times.iter().copied().reduce(f64::max).unwrap_or(0.0)
    );
    Ok(())
}
