use crate::{Result, config::BakeConfig, model::Model, quadrature};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};
use wgpu::util::DeviceExt;

type OrderObserver<'a> = dyn FnMut(usize, &[f32]) + 'a;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderStats {
    pub order: usize,
    pub max_radiance_increment: f32,
    pub relative_max_increment: f32,
    pub relative_texel_sum_increment: f32,
    pub elapsed_seconds: f32,
    /// CPU wall times including submission and device completion, not GPU timestamps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_wall_seconds: Option<OrderStageSeconds>,
    /// Worst local increment / total, floored at band solar irradiance * 1e-12.
    /// Includes both volume radiance and ground irradiance; not an error bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_local_relative_increment: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OrderStageSeconds {
    pub ground: f32,
    pub density: f32,
    pub expand_density: f32,
    pub integrate: f32,
    pub accumulate: f32,
    pub diagnostics_and_readback: f32,
}

pub struct BakedBand {
    pub optical_depth: Vec<f32>,
    pub radiance: Vec<f32>,
    pub ground_irradiance: Vec<f32>,
    pub orders: Vec<OrderStats>,
    pub stopped_by_tolerance: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dims: [u32; 4],
    tables: [u32; 4],
    integration: [u32; 4],
    work: [u32; 4],
    planet: [f32; 4],
    mapping: [u32; 4],
}

/// One persistent device and pipeline set; bands reuse it and are streamed to disk.
pub struct GpuBaker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
    pub adapter_name: String,
    coordinate_cache: Mutex<Option<CoordinateCache>>,
    reuse_work: bool,
}

struct CoordinateCache {
    geometry: crate::mapping::Geometry,
    config: BakeConfig,
    nodes: Arc<Vec<f32>>,
}

impl GpuBaker {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn new() -> Result<Self> {
        Self::new_with_features(wgpu::Features::empty())
    }
    pub fn new_with_features(features: wgpu::Features) -> Result<Self> {
        pollster::block_on(Self::new_async(features, true))
    }
    /// Diagnostic A/B switch. Both paths use identical transport and dense storage.
    pub fn new_with_work_reuse(reuse: bool) -> Result<Self> {
        pollster::block_on(Self::new_async(wgpu::Features::empty(), reuse))
    }
    async fn new_async(features: wgpu::Features, reuse: bool) -> Result<Self> {
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no wgpu adapter: {e}"))?;
        let info = adapter.get_info();
        let available = adapter.limits();
        let defaults = wgpu::Limits::default();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: available
                .max_storage_buffer_binding_size
                .min(512 * 1024 * 1024)
                .max(defaults.max_storage_buffer_binding_size),
            max_buffer_size: available
                .max_buffer_size
                .min(512 * 1024 * 1024)
                .max(defaults.max_buffer_size),
            ..defaults
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("spectral LUT baker"),
                required_limits,
                required_features: features,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;
        let entries: Vec<_> = (0..=8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding <= 3,
                        }
                    },
                    has_dynamic_offset: binding == 0,
                    min_binding_size: if binding == 0 {
                        wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64)
                    } else {
                        None
                    },
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LUT layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LUT pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spectral LUT compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bake.wgsl").into()),
        });
        let pipelines = [
            "transmittance",
            "ground_irradiance",
            "scattering_density",
            "integrate",
            "accumulate",
            "reduce_statistics",
            "expand_density",
            "integrate_direct",
        ]
        .iter()
        .map(|&entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("REUSE_DUPLICATE_STATES", if reuse { 1.0 } else { 0.0 })],
                    ..Default::default()
                },
                cache: None,
            })
        })
        .collect();
        if let Some(error) = errors.pop().await {
            return Err(format!("LUT shader validation: {error}").into());
        }
        Ok(Self {
            device,
            queue,
            layout,
            pipelines,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            coordinate_cache: Mutex::new(None),
            reuse_work: reuse,
        })
    }

    pub fn bake_band(
        &self,
        model: &Model,
        c: &BakeConfig,
        index: usize,
        progress: impl FnMut(&str),
    ) -> Result<BakedBand> {
        self.bake_band_internal(model, c, index, progress, None)
    }

    /// Optional per-order cumulative radiance followed by ground irradiance.
    /// Ordinary baking never copies these intermediate volumes to the CPU.
    pub fn bake_band_observed(
        &self,
        model: &Model,
        c: &BakeConfig,
        index: usize,
        progress: impl FnMut(&str),
        mut observer: impl FnMut(usize, &[f32]),
    ) -> Result<BakedBand> {
        self.bake_band_internal(model, c, index, progress, Some(&mut observer))
    }

    fn bake_band_internal(
        &self,
        model: &Model,
        c: &BakeConfig,
        index: usize,
        mut progress: impl FnMut(&str),
        mut observer: Option<&mut OrderObserver<'_>>,
    ) -> Result<BakedBand> {
        c.validate(model.bands.len())?;
        c.validate_top_height(model.geometry.top_height())?;
        let band = model.bands.get(index).ok_or("band index out of range")?;
        let len = c.scattering_len();
        let tau_len = c.optical_depth_len();
        let stats_len = len.div_ceil(64) * 5;
        let reduced_stats_len = len.div_ceil(64).div_ceil(256) * 5;
        let largest = (len + c.ground_sun_samples)
            .max(tau_len + 3 * c.ground_sun_samples + stats_len + reduced_stats_len)
            * 4;
        if largest as u64 > self.device.limits().max_storage_buffer_binding_size {
            return Err(
                format!("band buffer {largest} bytes exceeds GPU storage binding limit").into(),
            );
        }
        let profile: Vec<[f32; 8]> = band
            .profile
            .iter()
            .map(|(h, a)| {
                [
                    *h,
                    a.extinction,
                    a.scattering[0],
                    a.scattering[1],
                    a.scattering[2],
                    a.scattering[3],
                    a.scattering[4],
                    0.0,
                ]
            })
            .collect();
        let sphere = quadrature::sphere(c.angular_mu, c.angular_phi);
        let hemi = quadrature::hemisphere(c.angular_mu, c.angular_phi);
        let sun = quadrature::sun_disk(c.sun_mu, c.sun_phi, model.sun_radius);
        let quadrature: Vec<[f32; 12]> = sphere
            .iter()
            .chain(&hemi)
            .chain(&sun)
            .map(|&(v, w)| {
                let ph = band.phases(v.z);
                [
                    v.x, v.y, v.z, w, ph[0], ph[1], ph[2], ph[3], ph[4], 0.0, 0.0, 0.0,
                ]
            })
            .collect();
        let mut params = Params {
            dims: c.scattering.map(|n| n as u32),
            tables: [
                c.optical_depth[0] as u32,
                c.optical_depth[1] as u32,
                c.ground_sun_samples as u32,
                profile.len() as u32,
            ],
            integration: [
                c.ray_steps as u32,
                c.optical_depth_steps as u32,
                sphere.len() as u32,
                hemi.len() as u32,
            ],
            work: [sun.len() as u32, 1, 0, 0],
            planet: [
                model.geometry.bottom,
                model.geometry.top_height(),
                c.ground_albedo,
                band.info.solar_irradiance_w_m2,
            ],
            mapping: [
                c.mapping as u32,
                c.solar_interpolation as u32
                    | ((c.phase_interpolation as u32) << 8)
                    | ((c.view_interpolation as u32) << 16)
                    | (u32::from(!c.scattering_altitudes_km.is_empty()) << 24)
                    | ((c.height_interpolation as u32) << 25),
                c.angular_integration as u32
                    | ((c.iteration_scheme as u32) << 8)
                    | ((c.source_mapping as u32) << 16),
                c.cone_mapping as u32,
            ],
        };
        let uniform_stride = (std::mem::size_of::<Params>() as u32)
            .div_ceil(self.device.limits().min_uniform_buffer_offset_alignment)
            * self.device.limits().min_uniform_buffer_offset_alignment;
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LUT parameters"),
            size: uniform_stride as u64 * 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nodes = {
            let mut cache = self
                .coordinate_cache
                .lock()
                .map_err(|_| "coordinate cache lock poisoned")?;
            if cache
                .as_ref()
                .is_none_or(|v| v.geometry != model.geometry || v.config != *c)
            {
                *cache = Some(CoordinateCache {
                    geometry: model.geometry,
                    config: c.clone(),
                    nodes: Arc::new(crate::bake_schedule::angular_nodes(model.geometry, c)),
                });
            }
            Arc::clone(&cache.as_ref().unwrap().nodes)
        };
        let mut phase_and_angles = band.phase.clone();
        phase_and_angles.extend_from_slice(&nodes);
        let active_len = if self.reuse_work {
            crate::bake_schedule::active_work_len(c, &nodes)
        } else {
            len
        };
        if [
            profile.len() * std::mem::size_of::<[f32; 8]>(),
            phase_and_angles.len() * 4,
            quadrature.len() * std::mem::size_of::<[f32; 12]>(),
        ]
        .into_iter()
        .any(|n| n as u64 > self.device.limits().max_storage_buffer_binding_size)
        {
            return Err("bake lookup tables exceed GPU storage binding limit".into());
        }
        let buffers = [
            self.upload("profile", &profile),
            self.upload("phase and angular nodes", &phase_and_angles),
            self.upload("quadrature", &quadrature),
            self.storage(
                "auxiliary",
                tau_len + 3 * c.ground_sun_samples + stats_len + reduced_stats_len,
            ),
            self.storage("previous", len),
            self.storage("density", len),
            self.storage("next", len),
            self.storage("accumulated", len + c.ground_sun_samples),
        ];
        let entries: Vec<_> = std::iter::once(&uniform)
            .chain(&buffers)
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: if i == 0 {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    })
                } else {
                    b.as_entire_binding()
                },
            })
            .collect();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LUT band"),
            layout: &self.layout,
            entries: &entries,
        });
        self.dispatch(
            0,
            tau_len,
            c,
            &mut params,
            &uniform,
            &bindings,
            &mut progress,
        )?;
        let mut orders = Vec::new();
        let mut consecutive_small = 0;
        for order in 1..=c.max_orders {
            let start = Instant::now();
            let mut stages = OrderStageSeconds::default();
            let mut stage_start = Instant::now();
            params.work[1] = order as u32;
            progress(&format!("band {index}, order {order}/{}", c.max_orders));
            self.dispatch(
                1,
                c.ground_sun_samples,
                c,
                &mut params,
                &uniform,
                &bindings,
                &mut progress,
            )?;
            stages.ground = stage_start.elapsed().as_secs_f32();
            stage_start = Instant::now();
            let direct_first =
                order == 1 && c.mapping == crate::config::CoordinateMapping::RayAlignedReference;
            if !direct_first {
                self.dispatch(
                    2,
                    active_len,
                    c,
                    &mut params,
                    &uniform,
                    &bindings,
                    &mut progress,
                )?;
            }
            stages.density = stage_start.elapsed().as_secs_f32();
            stage_start = Instant::now();
            if c.mapping.is_reference() && !direct_first && self.reuse_work {
                self.dispatch(6, len, c, &mut params, &uniform, &bindings, &mut progress)?;
            }
            stages.expand_density = stage_start.elapsed().as_secs_f32();
            stage_start = Instant::now();
            self.dispatch(
                if direct_first { 7 } else { 3 },
                active_len,
                c,
                &mut params,
                &uniform,
                &bindings,
                &mut progress,
            )?;
            stages.integrate = stage_start.elapsed().as_secs_f32();
            stage_start = Instant::now();
            self.dispatch(4, len, c, &mut params, &uniform, &bindings, &mut progress)?;
            stages.accumulate = stage_start.elapsed().as_secs_f32();
            stage_start = Instant::now();
            self.dispatch(
                5,
                reduced_stats_len / 5 * 64,
                c,
                &mut params,
                &uniform,
                &bindings,
                &mut progress,
            )?;
            let blocks = self.read(
                &buffers[3],
                tau_len + c.ground_sun_samples + stats_len,
                reduced_stats_len,
            )?;
            if blocks.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err("GPU returned invalid transport or overflowing diagnostics".into());
            }
            let mut max_delta = 0.0_f32;
            let mut max_total = 0.0_f32;
            let mut sum_delta = 0.0;
            let mut sum_total = 0.0;
            let mut local_max = 0.0_f32;
            for b in blocks.as_chunks::<5>().0 {
                max_delta = max_delta.max(b[0]);
                max_total = max_total.max(b[1]);
                sum_delta += b[2];
                sum_total += b[3];
                local_max = local_max.max(b[4]);
            }
            stages.diagnostics_and_readback = stage_start.elapsed().as_secs_f32();
            let stats = OrderStats {
                order,
                max_radiance_increment: max_delta,
                relative_max_increment: max_delta / max_total.max(1e-30),
                relative_texel_sum_increment: sum_delta / sum_total.max(1e-30),
                elapsed_seconds: start.elapsed().as_secs_f32(),
                stage_wall_seconds: Some(stages),
                max_local_relative_increment: Some(local_max),
            };
            progress(&format!(
                "order {order}: relative max increment {:.6}, texel sum {:.6}, local {:.6}, {:.2}s",
                stats.relative_max_increment,
                stats.relative_texel_sum_increment,
                local_max,
                stats.elapsed_seconds
            ));
            let small = c.relative_order_tolerance > 0.0
                && order >= c.min_orders
                && stats.relative_max_increment <= c.relative_order_tolerance
                && stats.relative_texel_sum_increment <= c.relative_order_tolerance
                && local_max <= c.relative_order_tolerance;
            consecutive_small = if small { consecutive_small + 1 } else { 0 };
            orders.push(stats);
            if let Some(callback) = &mut observer {
                callback(
                    order,
                    &self.read(&buffers[7], 0, len + c.ground_sun_samples)?,
                );
            }
            if consecutive_small >= 2 {
                break;
            }
        }
        let optical_depth = self.read(&buffers[3], 0, tau_len)?;
        let mut radiance = self.read(&buffers[7], 0, len + c.ground_sun_samples)?;
        let ground_irradiance = radiance.split_off(len);
        if optical_depth
            .iter()
            .chain(&radiance)
            .chain(&ground_irradiance)
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("invalid GPU output".into());
        }
        Ok(BakedBand {
            optical_depth,
            radiance,
            ground_irradiance,
            orders,
            stopped_by_tolerance: consecutive_small >= 2,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        stage: usize,
        len: usize,
        c: &BakeConfig,
        p: &mut Params,
        uniform: &wgpu::Buffer,
        bindings: &wgpu::BindGroup,
        progress: &mut impl FnMut(&str),
    ) -> Result<()> {
        let mut last_update = Instant::now();
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as usize;
        let stride = std::mem::size_of::<Params>().div_ceil(alignment) * alignment;
        for base in (0..len).step_by(c.dispatch_texels * 8) {
            let mut bytes = vec![0u8; stride * 8];
            let mut ranges = Vec::with_capacity(8);
            for (slot, start) in (base..len).step_by(c.dispatch_texels).take(8).enumerate() {
                let end = (start + c.dispatch_texels).min(len);
                p.work[2] = start as u32;
                p.work[3] = end as u32;
                bytes[slot * stride..slot * stride + std::mem::size_of::<Params>()]
                    .copy_from_slice(bytemuck::bytes_of(p));
                ranges.push((end - start, slot * stride));
            }
            self.queue.write_buffer(uniform, 0, &bytes);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines[stage]);
                for (count, offset) in ranges {
                    pass.set_bind_group(0, bindings, &[offset as u32]);
                    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
                }
            }
            self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            if last_update.elapsed().as_secs() >= 10 {
                progress(&format!("stage {stage}: {}/{len} texels", p.work[3]));
                last_update = Instant::now();
            }
        }
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }
    fn upload<T: Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
    fn storage(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
    fn read(&self, source: &wgpu::Buffer, offset: usize, len: usize) -> Result<Vec<f32>> {
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LUT readback"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, (offset * 4) as u64, &output, 0, (len * 4) as u64);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        output.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let mapped = output.slice(..).get_mapped_range();
        let values = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        output.unmap();
        Ok(values)
    }
}
