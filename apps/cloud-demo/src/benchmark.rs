use crate::{app, output::Record};
use cloud_pt::{
    Result, gpu::ProgressiveRenderer, transport::TransportSettings, volume::SparseGpuVolume,
};
use std::{
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    time::Instant,
};

pub fn run(
    volume: &SparseGpuVolume,
    mut record: Record,
    rounds: u32,
    include_no_shadow_roulette: bool,
    output: &Path,
) -> Result<()> {
    if rounds == 0 || rounds > 100 {
        return Err("benchmark rounds must be in [1, 100]".into());
    }
    let (device, queue, adapter) = pollster::block_on(app::create_device(None))?;
    record.adapter = Some(adapter);
    let mut variants = vec![
        ("global_roulette", false, true),
        ("spatial_roulette", true, true),
    ];
    if include_no_shadow_roulette {
        variants.extend([("global", false, false), ("spatial", true, false)]);
    }
    let mut runs = Vec::new();
    let mut invalid_variants = std::collections::HashSet::new();
    for round in 0..rounds {
        // Rotate order to reduce systematic warm-cache and temperature effects.
        for offset in 0..variants.len() {
            let (name, spatial, roulette) = variants[(round as usize + offset) % variants.len()];
            if invalid_variants.contains(name) {
                runs.push(serde_json::json!({"round":round,"variant":name,
                    "status":"not_repeated_after_invalid","reason":"Identical scene and RNG stream already failed validation in an earlier round."}));
                continue;
            }
            let mut settings = record.transport.clone();
            settings.spatial_majorants = spatial;
            settings.shadow_roulette = roulette;
            let mut row = serde_json::json!({"round":round,"variant":name,
                "spatial_majorants":spatial,"shadow_roulette":roulette,"status":"running"});
            let row_index = runs.len();
            runs.push(row.clone());
            checkpoint(&record, rounds, output, &runs)?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                measure(volume, &record, &settings, &device, &queue)
            }));
            let mut fatal = None;
            match result {
                Ok(Ok(metrics)) => {
                    eprintln!(
                        "round {} / {}: {} GPU ms",
                        round + 1,
                        name,
                        metrics["gpu_compute_ms"]
                    );
                    row["status"] = "complete".into();
                    row["metrics"] = metrics;
                }
                Ok(Err(error)) => {
                    // Keep a failed ablation as an explicit failure, never as
                    // a successful timing or a partially valid reference film.
                    eprintln!("round {} / {} failed: {}", round + 1, name, error);
                    row["status"] = "invalid".into();
                    row["error"] = error.to_string().into();
                    invalid_variants.insert(name);
                    let message = error.to_string().to_lowercase();
                    if message.contains("device") && message.contains("lost") {
                        fatal = Some(error.to_string());
                    }
                }
                Err(payload) => {
                    let message = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "GPU/validation panic".into());
                    row["status"] = "device_or_validation_failure".into();
                    row["error"] = message.clone().into();
                    fatal = Some(message);
                }
            }
            runs[row_index] = row;
            checkpoint(&record, rounds, output, &runs)?;
            if let Some(message) = fatal {
                return Err(format!("cloud benchmark stopped; device is discarded, checkpoint saved to {}: {message}", output.display()).into());
            }
        }
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"cloud_pt_benchmark_v1","scene":record,"rounds":rounds,
            "warmup_samples":record.render.spp.min(2),"complete":true,"runs":runs,
            "timing_note":"GPU timestamps enclose path sampling and accumulation compute. Wall time includes encoding, submit and diagnostic readback, excludes the additional timestamp readback. Setup, VDB load, pipeline creation and film export are excluded. Invalid ablations are recorded without exporting or timing a partial reference. Different proposals use the same declared physical model and target spp, but different random trajectories. Repeated rounds use identical sample streams; variance-times-compute is a noisy finite-sample diagnostic, not a quality guarantee."
        }))?,
    )?;
    Ok(())
}

fn checkpoint(
    record: &Record,
    rounds: u32,
    output: &Path,
    runs: &[serde_json::Value],
) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"cloud_pt_benchmark_v1","scene":record,"rounds":rounds,
            "complete":false,"runs":runs
        }))?,
    )?;
    Ok(())
}

fn measure(
    volume: &SparseGpuVolume,
    record: &Record,
    settings: &TransportSettings,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<serde_json::Value> {
    let mut renderer = ProgressiveRenderer::new(
        device,
        queue,
        volume,
        &record.camera,
        settings,
        &record.render,
    )?;
    let warmup = record.render.spp.min(2);
    while renderer.sample_count() < warmup {
        let count = renderer
            .sample_batch_capacity()
            .min(warmup - renderer.sample_count());
        sample(&mut renderer, device, queue, count, false)?;
    }
    renderer.reset(device, queue, &record.camera, settings)?;
    let mut gpu_ms = Vec::new();
    let mut wall_ms = Vec::new();
    let mut wall_including_timing_ms = Vec::new();
    let mut batch_counts = Vec::new();
    let mut chunk_gpu_ms = Vec::new();
    let mut chunk_counts = Vec::new();
    while renderer.sample_count() < record.render.spp {
        let count = renderer
            .sample_batch_capacity()
            .min(record.render.spp - renderer.sample_count());
        let start = Instant::now();
        let (elapsed_gpu, elapsed_wall, chunks) =
            sample(&mut renderer, device, queue, count, true)?;
        batch_counts.push(count);
        wall_including_timing_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        wall_ms.push(elapsed_wall);
        chunk_counts.push(chunks);
        if !elapsed_gpu.is_empty() {
            gpu_ms.push(elapsed_gpu.iter().sum::<f64>());
            chunk_gpu_ms.extend(elapsed_gpu);
        }
    }
    let film = renderer.read_film(device, queue)?;
    let count = film.mean.len() as f64;
    let mut mean = [0.0f64; 3];
    let mut variance = [0.0f64; 3];
    for (pixel, var) in film.mean.iter().zip(&film.sample_variance) {
        for channel in 0..3 {
            mean[channel] += f64::from(pixel[channel]) / count;
            variance[channel] += f64::from(var[channel]) / count;
        }
    }
    let gpu_total: f64 = gpu_ms.iter().sum();
    let gpu_sum = (!gpu_ms.is_empty()).then_some(gpu_total);
    let variance_times_compute =
        gpu_sum.map(|ms| variance.map(|v| v * ms / f64::from(record.render.spp)));
    Ok(
        serde_json::json!({"gpu_compute_ms":gpu_sum,"gpu_batch_ms":gpu_ms,"batch_samples":batch_counts,
        "sample_batch_capacity":renderer.sample_batch_capacity(),
        "bounded_chunk_counts":chunk_counts,"gpu_chunk_ms":chunk_gpu_ms,
        "max_gpu_chunk_ms":chunk_gpu_ms.iter().copied().reduce(f64::max),
        "encode_submit_diagnostics_ms":wall_ms.iter().sum::<f64>(),"wall_batch_ms":wall_ms,
        "wall_including_timestamp_reads_ms":wall_including_timing_ms.iter().sum::<f64>(),
        "spatial_mean_rgb":mean,"spatial_mean_sample_variance_rgb":variance,
        "spatial_mean_variance_times_gpu_ms_per_spp":variance_times_compute,
        "sample_variance_valid":film.samples_per_pixel>1}),
    )
}

fn sample(
    renderer: &mut ProgressiveRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    count: u32,
    collect_timing: bool,
) -> Result<(Vec<f64>, f64, u64)> {
    let mut gpu = Vec::new();
    let mut wall = 0.0;
    let mut chunks = 0;
    loop {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.encode_work(device, queue, &mut encoder, count)?;
        queue.submit([encoder.finish()]);
        let progress = renderer.read_progress(device, queue)?;
        wall += start.elapsed().as_secs_f64() * 1000.0;
        chunks += 1;
        if collect_timing {
            if let Some(ms) = renderer.read_work_milliseconds(device, queue)? {
                gpu.push(ms);
            }
        }
        if progress.batch_finished {
            return Ok((gpu, wall, chunks));
        }
    }
}
