//! Headless exercise of the same experiment and presentation pipeline as the UI.
use crate::{
    assets::RealtimeAsset,
    color::DisplayTransform,
    controls::RealtimeControls,
    experiment::{
        CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, SurfaceViewport,
        UpdateContext,
    },
    passes::offline_lut::OfflineLutExperiment,
};
use std::{error::Error, path::Path, sync::mpsc};

pub fn render(
    reference_path: &Path,
    lut_path: &Path,
    output_path: &Path,
    kind: crate::app::ExperimentKind,
    linear: bool,
    view: [f32; 4],
    benchmark_frames: usize,
) -> Result<(), Box<dyn Error>> {
    if view.iter().any(|v| !v.is_finite())
        || !(-90.0..=90.0).contains(&view[1])
        || !(1.0..170.0).contains(&view[2])
        || benchmark_frames > 256
    {
        return Err("invalid snapshot camera".into());
    }
    let asset = RealtimeAsset::load(reference_path)?;
    let benchmark_features = if benchmark_frames > 0 {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
    } else {
        wgpu::Features::empty()
    };
    let gpu = sky_reference::solver::GpuBaker::new_with_features(benchmark_features)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let device = gpu.device();
    let queue = gpu.queue();
    let format = if linear {
        wgpu::TextureFormat::Rgba32Float
    } else {
        wgpu::TextureFormat::Rgba8UnormSrgb
    };
    let init = ExperimentInit {
        device,
        queue,
        surface_format: format,
        asset: &asset,
        display: DisplayTransform::default(),
    };
    let mut experiment: Box<dyn RealtimeExperiment> = match kind {
        crate::app::ExperimentKind::Hybrid4d => Box::new(
            crate::passes::hybrid::HybridExperiment::new(init).map_err(std::io::Error::other)?,
        ),
        crate::app::ExperimentKind::OfflineLut => {
            Box::new(OfflineLutExperiment::new(init, lut_path).map_err(std::io::Error::other)?)
        }
    };
    experiment.set_linear_output(linear);
    let mut controls = RealtimeControls::from_asset(&asset);
    controls.view.yaw_deg = view[0];
    controls.view.pitch_deg = view[1];
    controls.view.fov_y_deg = view[2];
    controls.exposure_ev = view[3];
    if !experiment.reference_available()
        || asset.manifest().transport_version.as_deref()
            != Some(sky_assets::asset::LAYERED_TRANSPORT_VERSION)
    {
        return Err("snapshot needs a readable reference EXR from the current transport version; re-render legacy PT assets".into());
    }
    let width = 512;
    let height = 384;
    let pixel_bytes = if linear { 16 } else { 4 };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("comparison snapshot"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let mut performance = Vec::new();
    if benchmark_frames > 0 {
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("comparison benchmark"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolved timestamps"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mapped = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let original = controls;
        for workload in ["camera_change", "sun_change", "cached_frame"] {
            controls = original;
            let mut gpu_ms = Vec::new();
            let mut submit_wait_ms = Vec::new();
            for sample in 0..benchmark_frames + 4 {
                if workload == "camera_change" {
                    controls.view.yaw_deg = original.view.yaw_deg + 0.01 * sample as f32;
                }
                if workload == "sun_change" {
                    controls.sun_elevation_deg =
                        (original.sun_elevation_deg + 0.01 * sample as f32).min(89.9);
                }
                let start = std::time::Instant::now();
                experiment.update(UpdateContext {
                    controls: &controls,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.write_timestamp(&queries, 0);
                experiment.render(FrameContext {
                    device,
                    queue,
                    encoder: &mut encoder,
                    target: &target_view,
                    viewport: SurfaceViewport {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    },
                });
                encoder.write_timestamp(&queries, 1);
                encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
                encoder.copy_buffer_to_buffer(&resolved, 0, &mapped, 0, 16);
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely())?;
                let elapsed = start.elapsed().as_secs_f32() * 1000.0;
                let (tx, rx) = mpsc::channel();
                mapped.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                device.poll(wgpu::PollType::wait_indefinitely())?;
                rx.recv()??;
                let bytes = mapped.slice(..).get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&bytes);
                let duration =
                    ticks[1].saturating_sub(ticks[0]) as f32 * queue.get_timestamp_period() * 1e-6;
                if sample >= 4 {
                    gpu_ms.push(duration);
                    submit_wait_ms.push(elapsed);
                }
                drop(bytes);
                mapped.unmap();
            }
            gpu_ms.sort_by(f32::total_cmp);
            submit_wait_ms.sort_by(f32::total_cmp);
            performance.push(serde_json::json!({"workload":workload,"samples":benchmark_frames,
                "gpu_median_ms":gpu_ms[gpu_ms.len()/2],"gpu_p95_ms":gpu_ms[(gpu_ms.len()*95/100).min(gpu_ms.len()-1)],
                "cpu_encode_submit_wait_median_ms":submit_wait_ms[submit_wait_ms.len()/2]}));
        }
        controls = original;
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot readback"),
        size: width as u64 * height as u64 * pixel_bytes as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut image = image::RgbaImage::new(width * 4, height);
    let mut raw = Vec::new();
    // Reuse the loaded resource for a display PNG alongside a linear export.
    // The second pass runs the actual demo tone mapper into the float target;
    // only the ordinary sRGB output transfer is performed on the CPU.
    for export_pass in 0..if linear { 2 } else { 1 } {
        experiment.set_linear_output(linear && export_pass == 0);
        for (panel, mode) in [
            CompareMode::Realtime,
            CompareMode::Reference,
            CompareMode::AbsoluteDifference,
            CompareMode::SignedDifference,
        ]
        .into_iter()
        .enumerate()
        {
            controls.compare_mode = mode;
            experiment.update(UpdateContext {
                controls: &controls,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            experiment.render(FrameContext {
                device,
                queue,
                encoder: &mut encoder,
                target: &target_view,
                viewport: SurfaceViewport {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * pixel_bytes),
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
            let (tx, rx) = mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely())?;
            rx.recv()??;
            let mapped = readback.slice(..).get_mapped_range();
            if linear && export_pass == 0 {
                raw.extend_from_slice(&mapped);
            } else {
                for y in 0..height {
                    for x in 0..width {
                        let base = ((y * width + x) * 4) as usize;
                        let rgba = if linear {
                            let values: &[f32] = bytemuck::cast_slice(&mapped);
                            let mut rgba = [255; 4];
                            for c in 0..3 {
                                let v = values[base + c].clamp(0.0, 1.0);
                                let encoded = if v <= 0.0031308 {
                                    v * 12.92
                                } else {
                                    1.055 * v.powf(1.0 / 2.4) - 0.055
                                };
                                rgba[c] = (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
                            }
                            rgba
                        } else {
                            mapped[base..base + 4].try_into()?
                        };
                        image.put_pixel(panel as u32 * width + x, y, image::Rgba(rgba));
                    }
                }
            }
            drop(mapped);
            readback.unmap();
        }
    }
    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    if linear {
        std::fs::write(output_path.with_extension("f32"), &raw)?;
        image.save(output_path.with_extension("png"))?;
        std::fs::write(
            output_path.with_extension("json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "width":width,"height":height,"panels":["solver","path_traced","absolute_difference","signed_difference"],
                "layout":"panel,y,x,rgba; little-endian f32","color_space":"linear Rec.2020; no exposure or display transform",
                "experiment":experiment.name(),"asset":reference_path,"lut":if matches!(kind, crate::app::ExperimentKind::Hybrid4d) {serde_json::Value::Null} else {serde_json::json!(lut_path)},
                "medium_source":if matches!(kind, crate::app::ExperimentKind::Hybrid4d) {"sky-realtime owned physical profiles; GPU startup solve; embedded four-wave calibration"} else {"experiment-specific"},"view_yaw_pitch_fov_exposure":view,
                "sun_elevation_deg":asset.manifest().sun_elevation_deg,"sun_azimuth_deg":asset.manifest().sun_azimuth_deg,
                "altitude_km":asset.manifest().observer_altitude_km,"pt_spp":asset.manifest().spp,
                "transport_version":sky_assets::asset::LAYERED_TRANSPORT_VERSION
                ,"comparison_pipeline":"pt-uv-inverse-v2",
                "display_preview":"same demo presentation; linear export unaffected by exposure"
                ,"reference_dimensions":asset.manifest().dimensions,"reference_kind":asset.manifest().kind,
                "adapter":gpu.adapter_name,"performance":performance,
                "performance_scope":"GPU timestamps include experiment update rendering and presentation at export resolution; four warmup frames; excludes loading and reference readback"
            }))?,
        )?;
    } else {
        image.save(output_path)?;
    }
    println!(
        "{}: LUT | path traced reference | absolute difference x4 | signed difference x4",
        output_path.display()
    );
    Ok(())
}
