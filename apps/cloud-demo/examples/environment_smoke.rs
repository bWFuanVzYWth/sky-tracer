//! Explicit tiny GPU QA, independent of the application and realtime renderer.
//! Usage: cargo run -p cloud-demo --example environment_smoke --release -- NEW_DIR [--preview-only|--preview-work-probe|--grouped-preview|--grouped-work-probe]
use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    film::Film,
    gpu::{
        MAX_WORK_PATHS, PREVIEW_WORK_TRANSITIONS, ProgressiveRenderer, WORK_PROGRESS_BYTES,
        WORK_TRANSITIONS,
    },
    transport::{Bounds, TransportSettings},
    vdb,
    volume::{SparseGpuVolume, UniformTransform},
};
use glam::DVec3;
use serde_json::json;
use std::{fs, path::Path, sync::mpsc, time::Instant};

async fn create_device() -> Result<(wgpu::Device, wgpu::Queue, String)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return Err("this explicit safety QA requires GPU timestamp support".into());
    }
    let mut limits = adapter.limits();
    limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.min(1 << 30);
    let name = adapter.get_info().name;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("tiny directional boundary QA"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?;
    Ok((device, queue, name))
}

fn constant_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgb: [f32; 3],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tiny constant sky or sun"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&[rgb[0], rgb[1], rgb[2], 1.0]),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(1),
        },
        texture.size(),
    );
    texture.create_view(&Default::default())
}

fn chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut ProgressiveRenderer,
) -> Result<(cloud_pt::gpu::WorkProgress, f64)> {
    let count = renderer
        .sample_batch_capacity()
        .min(renderer.target_samples() - renderer.sample_count());
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.encode_work(device, queue, &mut encoder, count)?;
    queue.submit([encoder.finish()]);
    let progress = renderer.read_progress(device, queue)?;
    let ms = renderer
        .read_work_milliseconds(device, queue)?
        .ok_or("missing GPU timestamp")?;
    if ms > 100.0 {
        return Err(
            format!("GPU chunk exceeded 100ms ({ms}); no new work will be submitted").into(),
        );
    }
    Ok((progress, ms))
}

fn finish(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut ProgressiveRenderer,
) -> Result<(Film, Vec<f64>, f64)> {
    let start = Instant::now();
    let mut times = Vec::new();
    while renderer.sample_count() < renderer.target_samples() {
        times.push(chunk(device, queue, renderer)?.1);
        if times.len() > 20_000 {
            return Err("tiny QA exceeded host chunk budget; no new work will be submitted".into());
        }
    }
    let film = renderer.read_film(device, queue)?;
    Ok((film, times, start.elapsed().as_secs_f64()))
}

fn read_rgb(path: &Path) -> Result<Vec<[f32; 3]>> {
    Ok(exr::prelude::read_first_rgba_layer_from_file(
        path,
        |size, _| (size.width(), vec![[0.0; 3]; size.width() * size.height()]),
        |pixels, position, (r, g, b, _): (f32, f32, f32, f32)| {
            pixels.1[position.y() * pixels.0 + position.x()] = [r, g, b];
        },
    )?
    .layer_data
    .channel_data
    .pixels
    .1)
}
fn bit_differences(a: &[[f32; 3]], b: &[[f32; 3]]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}
fn metrics(times: &[f64], wall: f64) -> serde_json::Value {
    json!({"chunks": times.len(), "max_gpu_chunk_ms": times.iter().copied().reduce(f64::max),
        "total_gpu_ms": times.iter().sum::<f64>(), "wall_seconds": wall, "gpu_chunk_ms": times})
}
fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn read_buffer_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiny preview pixel coverage"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;
    let mapped = staging.get_mapped_range(..);
    let result = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(result)
}
fn read_pixel_counts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &ProgressiveRenderer,
) -> Result<Vec<u32>> {
    let [width, height] = renderer.size();
    let bytes = read_buffer_bytes(
        device,
        queue,
        renderer.film_buffer(),
        u64::from(width) * u64::from(height) * 16,
    )?;
    Ok(bytes
        .chunks_exact(16)
        .map(|pixel| f32::from_le_bytes(pixel[12..16].try_into().unwrap()) as u32)
        .collect())
}

fn preview_qa(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &SparseGpuVolume,
    out: &Path,
    adapter: &str,
) -> Result<()> {
    let config = RenderConfig {
        width: 4,
        height: 2,
        spp: 4,
        ..Default::default()
    };
    let camera = Camera::default();
    let settings = TransportSettings::default();
    let mut preview =
        ProgressiveRenderer::new_preview(device, queue, volume, &camera, &settings, &config)?;
    assert_eq!(preview.sample_batch_capacity(), 1);
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(preview.encode_work(device, queue, &mut encoder, 2).is_err());
    preview.encode_work(device, queue, &mut encoder, 1)?;
    assert!(preview.set_environment(device, queue, None).is_err());
    assert!(preview.set_target_samples(8).is_err());
    queue.submit([encoder.finish()]);
    let old_ticket = read_buffer_bytes(
        device,
        queue,
        preview.progress_buffer(),
        WORK_PROGRESS_BYTES,
    )?;
    let first = preview.complete_work(&old_ticket)?;
    assert!(preview.complete_work(&old_ticket).is_err());
    let first_ms = preview.read_work_milliseconds(device, queue)?.unwrap();
    if first_ms > 100.0 {
        return Err(format!("preview first chunk exceeded 100ms: {first_ms}").into());
    }
    assert!(!first.batch_finished);
    assert!(preview.set_environment(device, queue, None).is_err());
    assert!(preview.set_target_samples(8).is_err());
    assert!(preview.read_film(device, queue).is_err());
    let (film, mut times, wall) = finish(device, queue, &mut preview)?;
    times.insert(0, first_ms);
    let frozen = Path::new("out/cloud_bounded_qa_v1/budget128_4x2x4/reference");
    let mean_delta = bit_differences(&film.mean, &read_rgb(&frozen.join("radiance.exr"))?);
    let variance_delta = bit_differences(
        &film.sample_variance,
        &read_rgb(&frozen.join("sample_variance.exr"))?,
    );
    write_json(
        &out.join("full_frame_preview.json"),
        &json!({"frozen_reference": frozen,
        "mean_bit_differences": mean_delta, "variance_bit_differences": variance_delta,
        "mean": film.mean, "sample_variance": film.sample_variance, "timing": metrics(&times, wall),
        "wall_scope": "remaining tracing and readbacks after first guard-check chunk",
        "inflight_environment_and_target_rejected": true, "partial_environment_and_target_rejected": true,
        "partial_film_rejected": true, "batch_count2_rejected": true}),
    )?;
    if mean_delta != 0 || variance_delta != 0 {
        return Err("preview changed frozen mean/variance bits".into());
    }
    preview.reset(device, queue, &camera, &settings)?;
    preview.set_environment(device, queue, None)?;
    preview.set_target_samples(8)?;
    assert_eq!(preview.target_samples(), 8);
    assert!(!preview.has_pending_work());
    let mut encoder = device.create_command_encoder(&Default::default());
    preview.encode_work(device, queue, &mut encoder, 1)?;
    assert!(preview.complete_work(&old_ticket).is_err());
    queue.submit([encoder.finish()]);
    preview.read_progress(device, queue)?;
    let epoch_check_ms = preview.read_work_milliseconds(device, queue)?.unwrap();
    if epoch_check_ms > 100.0 {
        return Err(format!("epoch-check chunk exceeded100ms: {epoch_check_ms}").into());
    }
    preview.reset(device, queue, &camera, &settings)?;

    let vacuum = SparseGpuVolume {
        hash: vec![[0, 0, 0, u32::MAX]],
        values: Vec::new(),
        tiles: Vec::new(),
        majorant_hash: vec![[0, 0, 0, u32::MAX]],
        transform: UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        index_bounds: Bounds {
            min: DVec3::splat(-1.0),
            max: DVec3::splat(1.0),
        },
        majorant: 1.0,
    };
    let config = RenderConfig {
        width: 128,
        height: 72,
        spp: 2,
        sample_batch_size: 1,
        ..Default::default()
    };
    let camera = Camera {
        origin: DVec3::ZERO,
        target: DVec3::Z,
        up: DVec3::Y,
        horizontal_fov_deg: 54.43,
    };
    let rgb = [0.15f32, 0.3, 0.6];
    let settings = TransportSettings {
        ground: None,
        sun_irradiance: DVec3::ZERO,
        sky_radiance: DVec3::from_array(rgb.map(f64::from)),
        ..Default::default()
    };
    let sky = constant_texture(device, queue, rgb);
    let sun = constant_texture(device, queue, [0.0; 3]);
    let mut preview =
        ProgressiveRenderer::new_preview(device, queue, &vacuum, &camera, &settings, &config)?;
    preview.set_environment(device, queue, Some((&sky, &sun)))?;
    assert_eq!(preview.sample_batch_capacity(), 1);
    assert_eq!(preview.sample_batch_storage_bytes(), 128 * 72 * 288);
    let mut vacuum_times = Vec::new();
    let mut completed = Vec::new();
    let mut first_rows = [0u32; 72];
    for sweep_chunk in 0..3 {
        let (progress, ms) = chunk(device, queue, &mut preview)?;
        vacuum_times.push(ms);
        completed.push(progress.completed_paths);
        let expected = ((sweep_chunk + 1) * MAX_WORK_PATHS).min(128 * 72);
        assert_eq!(progress.completed_paths, expected);
        assert_eq!(progress.total_paths, 128 * 72);
        let counts = read_pixel_counts(device, queue, &preview)?;
        assert_eq!(counts.iter().filter(|&&v| v == 1).count() as u32, expected);
        if sweep_chunk == 0 {
            for (pixel, count) in counts.iter().enumerate() {
                first_rows[pixel / 128] += count;
            }
            assert!(
                first_rows.iter().all(|&v| v > 0),
                "first chunk must show every image row"
            );
        }
        if sweep_chunk < 2 {
            assert_eq!(preview.sample_count(), 0);
            assert!(preview.read_film(device, queue).is_err());
        } else {
            assert_eq!(preview.sample_count(), 1);
            assert!(counts.iter().all(|&v| v == 1));
        }
    }
    let (film, tail_times, wall) = finish(device, queue, &mut preview)?;
    vacuum_times.extend(tail_times);
    assert!(
        film.mean
            .iter()
            .all(|v| v.map(f32::to_bits) == rgb.map(f32::to_bits))
    );
    assert!(film.sample_variance.iter().flatten().all(|&v| v == 0.0));
    write_json(
        &out.join("round_robin_vacuum.json"),
        &json!({"dimensions": [128,72], "samples": 2,
        "first_sweep_completion_counts": completed, "first_chunk_complete_pixels_per_row": first_rows.to_vec(),
        "all_pixels_completed_after_three_chunks": true, "mean_rgb_exact": rgb, "variance_zero": true,
        "path_storage_bytes": preview.sample_batch_storage_bytes(), "timing": metrics(&vacuum_times, wall),
        "wall_scope": "remaining second-sample tracing and final readback; first-sweep diagnostics excluded"}),
    )?;
    write_json(
        &out.join("summary.json"),
        &json!({"complete": true, "adapter": adapter, "mode": "preview-only",
        "bounded_only": true, "transitions_per_path_per_chunk": PREVIEW_WORK_TRANSITIONS,
        "preview_mean_and_variance_frozen_bitexact": true, "round_robin_non_aligned_fullframe": true,
        "first_chunk_spans_every_row": true, "partial_and_inflight_mutations_rejected": true,
        "reset_environment_target_succeeded": true, "old_epoch_progress_rejected": true,
        "duplicate_progress_rejected": true, "discarded_epoch_check_chunk_ms": epoch_check_ms,
        "full_frame_preview_file": "full_frame_preview.json",
        "vacuum_file": "round_robin_vacuum.json"}),
    )?;
    println!(
        "preview-only QA complete: frozen mean/variance exact,9216 vacuum pixels complete in3 bounded chunks"
    );
    Ok(())
}

fn preview_work_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &SparseGpuVolume,
    out: &Path,
    adapter: &str,
) -> Result<()> {
    let config = RenderConfig {
        width: 128,
        height: 72,
        spp: 2,
        ..Default::default()
    };
    let camera = Camera::default();
    let settings = TransportSettings {
        ground: None,
        ..Default::default()
    };
    let sky = constant_texture(device, queue, [0.15, 0.3, 0.6]);
    let sun = constant_texture(device, queue, [2.6, 2.5, 2.3]);
    let mut preview =
        ProgressiveRenderer::new_preview(device, queue, volume, &camera, &settings, &config)?;
    preview.set_environment(device, queue, Some((&sky, &sun)))?;
    let start = Instant::now();
    let mut times = Vec::new();
    let mut completed = Vec::new();
    for _ in 0..64 {
        let (progress, ms) = match chunk(device, queue, &mut preview) {
            Ok(result) => result,
            Err(error) => {
                write_json(
                    &out.join("failure.json"),
                    &json!({"error": error.to_string(), "chunks_completed": times.len(), "no_more_submissions": true}),
                )?;
                return Err(error);
            }
        };
        times.push(ms);
        completed.push(progress.completed_paths);
        write_json(
            &out.join("progress.json"),
            &json!({"chunks": times.len(), "max_gpu_chunk_ms": times.iter().copied().reduce(f64::max),
            "samples": preview.sample_count(), "completed_paths": progress.completed_paths, "total_paths": progress.total_paths}),
        )?;
        if ms > 16.0 || preview.sample_count() == config.spp {
            break;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let counts = read_pixel_counts(device, queue, &preview)?;
    let mut rows = [0; 72];
    for (pixel, count) in counts.iter().enumerate() {
        rows[pixel / 128] += u32::from(*count > 0);
    }
    let mut ordered = times.clone();
    ordered.sort_by(f64::total_cmp);
    let quantile = |fraction: f64| ordered[((ordered.len() - 1) as f64 * fraction).ceil() as usize];
    write_json(
        &out.join("summary.json"),
        &json!({"complete": true, "adapter": adapter,
        "dimensions": [128,72], "transitions_per_path_per_chunk": PREVIEW_WORK_TRANSITIONS,
        "max_work_slots": MAX_WORK_PATHS, "single_submission_inflight": true,
        "max_chunks": 64, "timing": metrics(&times, wall), "p50_ms": quantile(0.5), "p95_ms": quantile(0.95), "p99_ms": quantile(0.99),
        "all_chunks_under16ms": ordered.last().unwrap() <= &16.0, "samples_completed": preview.sample_count(),
        "visible_pixels": counts.iter().filter(|&&n| n > 0).count(), "visible_pixels_by_row": rows.to_vec(),
        "completed_paths_per_chunk": completed, "reference_exported": false, "partial_display_only": true,
        "transport": settings, "camera": camera,
        "environment": {"kind": "constant_table", "radiance": [0.15,0.3,0.6], "sun_irradiance": [2.6,2.5,2.3]}}),
    )?;
    println!(
        "preview work probe complete: {}chunks,p99 {:.3}ms,max {:.3}ms,{} visible pixels",
        times.len(),
        quantile(0.99),
        ordered.last().unwrap(),
        counts.iter().filter(|&&n| n > 0).count()
    );
    Ok(())
}

fn grouped_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut ProgressiveRenderer,
    chunks: u32,
) -> Result<(cloud_pt::gpu::WorkProgress, f64, Vec<f64>)> {
    let count = renderer
        .sample_batch_capacity()
        .min(renderer.target_samples() - renderer.sample_count());
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.encode_work_group(device, queue, &mut encoder, count, chunks)?;
    assert_eq!(renderer.work_group_size(), chunks);
    assert_eq!(
        renderer.work_chunk_timing_buffer().unwrap().1,
        u64::from(chunks) * 16
    );
    queue.submit([encoder.finish()]);
    let progress = renderer.read_progress(device, queue)?;
    let whole = renderer.read_work_milliseconds(device, queue)?.unwrap();
    let individual = renderer
        .read_work_chunk_milliseconds(device, queue)?
        .unwrap();
    assert_eq!(individual.len(), chunks as usize);
    assert!(individual.iter().sum::<f64>() <= whole + 0.001);
    if whole > 100.0 || individual.iter().any(|&ms| ms > 100.0) {
        return Err(format!("bounded GPU group exceeded100ms: whole{whole}ms, chunks{individual:?}; no new submission").into());
    }
    Ok((progress, whole, individual))
}

fn grouped_preview_qa(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &SparseGpuVolume,
    out: &Path,
    adapter: &str,
) -> Result<()> {
    let config = RenderConfig {
        width: 4,
        height: 2,
        spp: 4,
        sample_batch_size: 4,
        ..Default::default()
    };
    let camera = Camera::default();
    let settings = TransportSettings::default();
    let frozen = Path::new("out/cloud_bounded_qa_v1/budget128_4x2x4/reference");
    let frozen_mean = read_rgb(&frozen.join("radiance.exr"))?;
    let frozen_variance = read_rgb(&frozen.join("sample_variance.exr"))?;
    let mut comparisons = Vec::new();
    for chunks in [1, 4, 8] {
        let mut renderer =
            ProgressiveRenderer::new_preview(device, queue, volume, &camera, &settings, &config)?;
        assert_eq!(renderer.sample_batch_capacity(), 4);
        let start = Instant::now();
        let mut whole_times = Vec::new();
        let mut individual_times = Vec::new();
        while renderer.sample_count() < config.spp {
            let (_, whole, individual) = grouped_chunk(device, queue, &mut renderer, chunks)?;
            whole_times.push(whole);
            individual_times.extend(individual);
            if whole_times.len() > 20_000 {
                return Err("tiny grouped QA exceeds host submission budget".into());
            }
        }
        let film = renderer.read_film(device, queue)?;
        let mean_diff = bit_differences(&film.mean, &frozen_mean);
        let variance_diff = bit_differences(&film.sample_variance, &frozen_variance);
        let result = json!({"group_chunks": chunks, "cohort": 4, "mean_bit_differences": mean_diff,
            "variance_bit_differences": variance_diff, "mean": film.mean, "sample_variance": film.sample_variance,
            "timing": metrics(&whole_times, start.elapsed().as_secs_f64()),
            "individual_dispatch_ms": individual_times,
            "max_kernel_ms": individual_times.iter().copied().reduce(f64::max)});
        write_json(&out.join(format!("group{chunks}.json")), &result)?;
        if mean_diff != 0 || variance_diff != 0 {
            return Err("grouped preview changed frozen mean/variance bits".into());
        }
        comparisons.push(result);
    }
    let vacuum = SparseGpuVolume {
        hash: vec![[0, 0, 0, u32::MAX]],
        values: Vec::new(),
        tiles: Vec::new(),
        majorant_hash: vec![[0, 0, 0, u32::MAX]],
        transform: UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        index_bounds: Bounds {
            min: DVec3::splat(-1.0),
            max: DVec3::splat(1.0),
        },
        majorant: 1.0,
    };
    let config = RenderConfig {
        width: 128,
        height: 72,
        spp: 7,
        sample_batch_size: 4,
        ..Default::default()
    };
    let camera = Camera {
        origin: DVec3::ZERO,
        target: DVec3::Z,
        up: DVec3::Y,
        horizontal_fov_deg: 54.43,
    };
    let rgb = [0.15f32, 0.3, 0.6];
    let settings = TransportSettings {
        ground: None,
        sun_irradiance: DVec3::ZERO,
        sky_radiance: DVec3::from_array(rgb.map(f64::from)),
        ..Default::default()
    };
    let sky = constant_texture(device, queue, rgb);
    let sun = constant_texture(device, queue, [0.0; 3]);
    let mut renderer =
        ProgressiveRenderer::new_preview(device, queue, &vacuum, &camera, &settings, &config)?;
    renderer.set_environment(device, queue, Some((&sky, &sun)))?;
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        renderer
            .encode_work_group(device, queue, &mut encoder, 4, 0)
            .is_err()
    );
    assert!(
        renderer
            .encode_work_group(device, queue, &mut encoder, 4, 9)
            .is_err()
    );
    let (partial, _, _) = grouped_chunk(device, queue, &mut renderer, 4)?;
    assert_eq!(partial.completed_paths, 4 * 4096);
    assert_eq!(partial.total_paths, 4 * 128 * 72);
    assert_eq!(renderer.sample_count(), 0);
    assert!(renderer.read_film(device, queue).is_err());
    assert!(renderer.set_target_samples(8).is_err());
    assert!(renderer.set_environment(device, queue, None).is_err());
    let counts = read_pixel_counts(device, queue, &renderer)?;
    assert_eq!(counts.iter().filter(|&&n| n == 4).count(), 4096);
    assert_eq!(counts.iter().filter(|&&n| n == 0).count(), 5120);
    renderer.reset(device, queue, &camera, &settings)?;
    assert!(
        read_pixel_counts(device, queue, &renderer)?
            .iter()
            .all(|&n| n == 0)
    );
    renderer.set_target_samples(7)?;
    renderer.set_environment(device, queue, None)?;
    grouped_chunk(device, queue, &mut renderer, 4)?;
    let (finished4, _, _) = grouped_chunk(device, queue, &mut renderer, 8)?;
    assert_eq!(finished4.completed_paths, 4 * 128 * 72);
    assert_eq!(renderer.sample_count(), 4);
    assert!(
        read_pixel_counts(device, queue, &renderer)?
            .iter()
            .all(|&n| n == 4)
    );
    let (finished7, _, _) = grouped_chunk(device, queue, &mut renderer, 8)?;
    assert_eq!(finished7.batch_samples, 3);
    assert_eq!(finished7.completed_paths, 3 * 128 * 72);
    assert_eq!(renderer.sample_count(), 7);
    let film = renderer.read_film(device, queue)?;
    assert!(
        film.mean
            .iter()
            .all(|pixel| pixel.map(f32::to_bits) == rgb.map(f32::to_bits))
    );
    assert!(film.sample_variance.iter().flatten().all(|&v| v == 0.0));
    write_json(
        &out.join("summary.json"),
        &json!({"complete": true, "adapter": adapter,
        "grouped_preview_reference_bitexact": true, "tested_groups": [1,4,8], "cohort_capacity": 4,
        "vacuum_dimensions": [128,72], "last_batch_size": 3, "total_samples": 7,
        "group_must_not_start_next_batch": true, "partial_sample_counts_rejected": true,
        "reset_cleared_all_partial_pixels": true, "zero_variance_exact": true,
        "max_slots_per_kernel": MAX_WORK_PATHS, "transitions_per_slot": PREVIEW_WORK_TRANSITIONS,
        "group_comparisons": comparisons}),
    )?;
    println!(
        "grouped preview QA complete: groups1/4/8cohort4 frozen bitexact;vacuum reset and4+3sample batches exact"
    );
    Ok(())
}

fn grouped_work_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &SparseGpuVolume,
    out: &Path,
    adapter: &str,
) -> Result<()> {
    let config = RenderConfig {
        width: 128,
        height: 72,
        spp: 4,
        sample_batch_size: 4,
        ..Default::default()
    };
    let camera = Camera::default();
    let settings = TransportSettings {
        ground: None,
        ..Default::default()
    };
    let sky = constant_texture(device, queue, [0.15, 0.3, 0.6]);
    let sun = constant_texture(device, queue, [2.6, 2.5, 2.3]);
    let mut renderer =
        ProgressiveRenderer::new_preview(device, queue, volume, &camera, &settings, &config)?;
    renderer.set_environment(device, queue, Some((&sky, &sun)))?;
    let mut results = Vec::new();
    let mut previous = None;
    for chunks in [1, 4, 8] {
        if let Some((previous_chunks, previous_ms)) = previous {
            if previous_ms * f64::from(chunks) / f64::from(previous_chunks) > 100.0 {
                results.push(json!({"group_chunks": chunks, "skipped": true, "reason": "preceding measured group predicts over100ms"}));
                break;
            }
        }
        let wall = Instant::now();
        let (progress, whole, individual) = match grouped_chunk(
            device,
            queue,
            &mut renderer,
            chunks,
        ) {
            Ok(result) => result,
            Err(error) => {
                write_json(
                    &out.join("failure.json"),
                    &json!({"error": error.to_string(), "previous_results": results, "no_more_submissions": true}),
                )?;
                return Err(error);
            }
        };
        let entry = json!({"group_chunks": chunks, "whole_gpu_ms": whole,
            "kernel_ms": individual, "sum_kernel_ms": individual.iter().sum::<f64>(),
            "max_kernel_ms": individual.iter().copied().reduce(f64::max),
            "wall_seconds": wall.elapsed().as_secs_f64(), "samples": renderer.sample_count(),
            "completed_paths": progress.completed_paths, "total_paths": progress.total_paths,
            "within12ms_group_target": whole <= 12.0});
        println!(
            "real grouped probe {chunks}:whole{whole:.3}ms,maxkernel{:.3}ms",
            individual.iter().copied().reduce(f64::max).unwrap()
        );
        results.push(entry);
        write_json(&out.join("progress.json"), &json!({"groups": results}))?;
        previous = Some((chunks, whole));
        if progress.batch_finished {
            break;
        }
    }
    let counts = read_pixel_counts(device, queue, &renderer)?;
    write_json(
        &out.join("summary.json"),
        &json!({"complete": true,"adapter": adapter,
        "dimensions": [128,72], "cohort": 4, "transitions_per_slot": PREVIEW_WORK_TRANSITIONS,
        "max_slots_per_kernel": MAX_WORK_PATHS, "group_results": results,
        "samples_completed": renderer.sample_count(), "visible_pixels": counts.iter().filter(|&&n| n > 0).count(),
        "visible_sample_sum": counts.iter().sum::<u32>(), "reference_exported": false,
        "limits": "One measured group per size on a progressing scene, not a statistically stable benchmark; GPU blocks remain fixed and bounded. Group duration governs viewer latency."}),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let preview_only = args.len() == 2 && args[1] == "--preview-only";
    let work_probe = args.len() == 2 && args[1] == "--preview-work-probe";
    let grouped_only = args.len() == 2 && args[1] == "--grouped-preview";
    let grouped_probe = args.len() == 2 && args[1] == "--grouped-work-probe";
    if args.len() != 1 && !preview_only && !work_probe && !grouped_only && !grouped_probe {
        return Err(
            "usage: environment_smoke NEW_OUTPUT_DIRECTORY [--preview-only|--preview-work-probe|--grouped-preview|--grouped-work-probe]"
                .into(),
        );
    }
    let out = Path::new(&args[0]);
    if out.exists() {
        return Err("QA output already exists".into());
    }
    fs::create_dir_all(out)?;
    let config = RenderConfig {
        width: 4,
        height: 2,
        spp: 4,
        ..Default::default()
    };
    let volume = vdb::load_vdb(
        Path::new("assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb"),
        "density",
    )?
    .pack_gpu()?;
    let camera = Camera::default();
    let settings = TransportSettings::default();
    let (device, queue, adapter) = pollster::block_on(create_device())?;
    if preview_only {
        return preview_qa(&device, &queue, &volume, out, &adapter);
    }
    if work_probe {
        return preview_work_probe(&device, &queue, &volume, out, &adapter);
    }
    if grouped_only {
        return grouped_preview_qa(&device, &queue, &volume, out, &adapter);
    }
    if grouped_probe {
        return grouped_work_probe(&device, &queue, &volume, out, &adapter);
    }
    let rgb = [0.15f32, 0.3, 0.6];
    let sky = constant_texture(&device, &queue, rgb);
    let sun = constant_texture(&device, &queue, [0.0; 3]);
    let mut renderer =
        ProgressiveRenderer::new(&device, &queue, &volume, &camera, &settings, &config)?;
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.encode_work(&device, &queue, &mut encoder, 4)?;
    assert!(
        renderer
            .set_environment(&device, &queue, Some((&sky, &sun)))
            .is_err()
    );
    assert!(renderer.set_target_samples(8).is_err());
    queue.submit([encoder.finish()]);
    let first = renderer.read_progress(&device, &queue)?;
    let first_ms = renderer.read_work_milliseconds(&device, &queue)?.unwrap();
    if first_ms > 100.0 {
        return Err(format!("first QA chunk exceeded 100ms: {first_ms}").into());
    }
    assert!(
        !first.batch_finished,
        "Disney QA must exercise a partial batch"
    );
    assert!(
        renderer
            .set_environment(&device, &queue, Some((&sky, &sun)))
            .is_err()
    );
    assert!(renderer.set_target_samples(8).is_err());
    assert!(renderer.read_film(&device, &queue).is_err());
    renderer.reset(&device, &queue, &camera, &settings)?;
    renderer.set_environment(&device, &queue, None)?;
    renderer.set_target_samples(4)?;
    let (film, times, wall) = finish(&device, &queue, &mut renderer)?;
    let frozen = Path::new("out/cloud_bounded_qa_v1/budget128_4x2x4/reference");
    let mean_delta = bit_differences(&film.mean, &read_rgb(&frozen.join("radiance.exr"))?);
    let variance_delta = bit_differences(
        &film.sample_variance,
        &read_rgb(&frozen.join("sample_variance.exr"))?,
    );
    write_json(
        &out.join("constant_mode.json"),
        &json!({"adapter": adapter, "render": config,
        "camera": camera, "transport": settings, "environment": null, "frozen_reference": frozen,
        "mean_bit_differences": mean_delta, "variance_bit_differences": variance_delta,
        "mean": film.mean, "sample_variance": film.sample_variance, "timing": metrics(&times, wall),
        "initial_discarded_chunk_ms": first_ms}),
    )?;
    if mean_delta != 0 || variance_delta != 0 {
        return Err(format!("disabled environment changed frozen reference: {mean_delta} mean, {variance_delta} variance bits").into());
    }
    println!(
        "constant-mode frozen result: exact mean and variance, max {:.3}ms",
        times.iter().copied().reduce(f64::max).unwrap()
    );

    let mut preview =
        ProgressiveRenderer::new_preview(&device, &queue, &volume, &camera, &settings, &config)?;
    assert_eq!(preview.sample_batch_capacity(), 1);
    assert!(
        preview
            .encode_work(
                &device,
                &queue,
                &mut device.create_command_encoder(&Default::default()),
                2
            )
            .is_err()
    );
    let (preview_film, preview_times, preview_wall) = finish(&device, &queue, &mut preview)?;
    let preview_mean_delta = bit_differences(&film.mean, &preview_film.mean);
    let preview_variance_delta =
        bit_differences(&film.sample_variance, &preview_film.sample_variance);
    write_json(
        &out.join("full_frame_preview.json"),
        &json!({
            "mean_bit_differences_from_frozen": preview_mean_delta,
            "variance_bit_differences_from_frozen": preview_variance_delta,
            "mean": preview_film.mean, "sample_variance": preview_film.sample_variance,
            "path_storage_bytes": preview.sample_batch_storage_bytes(),
            "logical_batch": preview.sample_batch_capacity(), "timing": metrics(&preview_times, preview_wall),
        }),
    )?;
    if preview_mean_delta != 0 || preview_variance_delta != 0 {
        return Err(format!("full-frame preview changed frozen reference: {preview_mean_delta} mean, {preview_variance_delta} variance bits").into());
    }
    println!(
        "full-frame preview frozen result: exact mean and variance, max {:.3}ms",
        preview_times.iter().copied().reduce(f64::max).unwrap()
    );

    // The density field is identically zero; its positive conservative bound
    // remains valid. Empty spatial cells skip all extinction candidates.
    let vacuum = SparseGpuVolume {
        hash: vec![[0, 0, 0, u32::MAX]],
        values: Vec::new(),
        tiles: Vec::new(),
        majorant_hash: vec![[0, 0, 0, u32::MAX]],
        transform: UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        index_bounds: Bounds {
            min: DVec3::splat(-1.0),
            max: DVec3::splat(1.0),
        },
        majorant: 1.0,
    };
    let vacuum_camera = Camera {
        origin: DVec3::ZERO,
        target: DVec3::Z,
        up: DVec3::Y,
        horizontal_fov_deg: 54.43,
    };
    let vacuum_settings = TransportSettings {
        ground: None,
        sun_irradiance: DVec3::ZERO,
        sky_radiance: DVec3::from_array(rgb.map(f64::from)),
        ..Default::default()
    };
    let mut renderer = ProgressiveRenderer::new(
        &device,
        &queue,
        &vacuum,
        &vacuum_camera,
        &vacuum_settings,
        &config,
    )?;
    renderer.set_environment(&device, &queue, Some((&sky, &sun)))?;
    renderer.set_target_samples(8)?;
    assert!(renderer.set_target_samples(0).is_err());
    let (environment, env_times, env_wall) = finish(&device, &queue, &mut renderer)?;
    assert!(
        environment
            .mean
            .iter()
            .all(|v| v.map(f32::to_bits) == rgb.map(f32::to_bits))
    );
    assert!(
        environment
            .sample_variance
            .iter()
            .flatten()
            .all(|&v| v == 0.0)
    );
    assert!(renderer.set_environment(&device, &queue, None).is_err());
    renderer.reset(&device, &queue, &vacuum_camera, &vacuum_settings)?;
    // Camera/transport reset must retain the directional boundary binding.
    renderer.set_target_samples(4)?;
    let (reset_environment, reset_times, reset_wall) = finish(&device, &queue, &mut renderer)?;
    assert_eq!(reset_environment.mean, environment.mean);
    assert_eq!(
        reset_environment.sample_variance,
        environment.sample_variance
    );
    renderer.reset(&device, &queue, &vacuum_camera, &vacuum_settings)?;
    renderer.set_environment(&device, &queue, None)?;
    renderer.set_target_samples(8)?;
    let (constant, const_times, const_wall) = finish(&device, &queue, &mut renderer)?;
    assert_eq!(environment.mean, constant.mean);
    assert_eq!(environment.sample_variance, constant.sample_variance);
    write_json(
        &out.join("environment.json"),
        &json!({"density": 0, "ground": null, "sun": [0,0,0],
        "environment_size": [1,1], "rgb": rgb, "samples_per_pixel": 8, "mean": environment.mean,
        "sample_variance": environment.sample_variance, "constant_mode_exact": true,
        "reset_preserved_environment_exact": true, "directional": metrics(&env_times, env_wall),
        "reset_directional": metrics(&reset_times, reset_wall), "disabled": metrics(&const_times, const_wall)}),
    )?;
    write_json(
        &out.join("summary.json"),
        &json!({"complete": true, "adapter": adapter,
        "offline_transitions_per_path_per_chunk": WORK_TRANSITIONS,
        "preview_transitions_per_path_per_chunk": PREVIEW_WORK_TRANSITIONS,
        "largest_film": [4,2], "bounded_only": true,
        "frozen_constant_mean_bit_differences": 0, "frozen_constant_variance_bit_differences": 0,
        "full_frame_preview_mean_bit_differences": 0, "full_frame_preview_variance_bit_differences": 0,
        "constant_env_and_disabled_exact": true, "constant_env_variance_zero": true,
        "inflight_environment_and_target_rejected": true, "partial_environment_and_target_rejected": true,
        "partial_film_rejected": true, "reset_switch_and_target_succeeded": true,
        "invalid_target_rejected": true, "constant_mode_file": "constant_mode.json", "environment_file": "environment.json",
        "full_frame_preview_file": "full_frame_preview.json"}),
    )?;
    println!("directional environment tiny GPU QA complete; constant RGB and zero variance exact");
    Ok(())
}
