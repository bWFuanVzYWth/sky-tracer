//! Progressive f32 GPU estimator. Creating a renderer is explicit: merely
//! inspecting a VDB, validating shaders, or running CPU tests creates no device.

use crate::{
    Result,
    config::{Camera, RenderConfig},
    film::Film,
    transport::TransportSettings,
    volume::SparseGpuVolume,
};
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use std::{cell::Cell, sync::mpsc};
use wgpu::util::DeviceExt;

/// Film counts must remain exactly representable by the WGSL f32 update.
const MAX_EXACT_SAMPLES: u32 = 1 << 24;
const AUTO_PARALLEL_PATHS: u32 = 32_768;
const AUTO_MAX_BATCH: u32 = 1024;
pub const MAX_WORK_PATHS: u32 = 4096;
pub const WORK_TRANSITIONS: u32 = 128;
/// Preview retains the measured responsive finite budget. Larger 256/512
/// candidates increased full-frame dispatch latency and were rejected.
pub const PREVIEW_WORK_TRANSITIONS: u32 = 128;
pub const MAX_PREVIEW_BATCH: u32 = 4;
pub const MAX_WORK_GROUP_CHUNKS: u32 = 8;
pub const WORK_PROGRESS_BYTES: u64 = 32;
const PATH_STATE_BYTES: u64 = 288;
const DIRECTIONAL_ENVIRONMENT: u32 = 8;
const FULL_FRAME_PREVIEW: u32 = 16;
// The spatial proposal's one-voxel numerical halo assumes coordinate arithmetic
// is well below a voxel ULP. Extremely distant origins must use CPU f64.
const MAX_INDEX_MAGNITUDE: f64 = 262_144.0;
pub const TRACE_SHADER: &str = include_str!("cloud_trace.wgsl");
pub const PRESENT_SHADER: &str = include_str!("present.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    image: [u32; 4],
    control: [u32; 4],
    storage: [u32; 4],
    camera_origin: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],
    index_min: [f32; 4],
    index_max: [f32; 4],
    transform: [f32; 4],
    optics: [f32; 4],
    albedo: [f32; 4],
    sun_direction: [f32; 4],
    sun_irradiance: [f32; 4],
    sky_radiance: [f32; 4],
    ground_albedo: [f32; 4],
    batch: [u32; 4],
    work: [u32; 4],
}

/// Each flag invalidates the complete accumulated image. A watchdog is a
/// workload diagnostic, never a permitted finite-bounce truncation.
#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    pub flags: u32,
    pub first_pixel: u32,
    pub first_sample: u32,
}
impl Diagnostics {
    pub fn validate(self) -> Result<()> {
        if self.flags == 0 {
            return Ok(());
        }
        let mut causes = Vec::new();
        if self.flags & 1 != 0 {
            causes.push("density majorant violation");
        }
        if self.flags & 2 != 0 {
            causes.push("non-finite arithmetic");
        }
        if self.flags & 4 != 0 {
            causes.push("compensated floating-point ray distance stagnated");
        }
        if self.flags & 8 != 0 {
            causes.push("path event watchdog exhausted");
        }
        if self.flags & 16 != 0 {
            causes.push("ray coordinates exceed the GPU voxel precision budget; use CPU f64");
        }
        if self.flags & 32 != 0 {
            causes.push("sparse hash lookup exhausted its validated capacity");
        }
        if self.flags & 64 != 0 {
            causes.push("directional environment or sunlight contains invalid RGB radiance");
        }
        Err(format!("cloud GPU film invalid at pixel {}, sample {}: {} (flags 0x{:x}); no reference image can be exported",
            self.first_pixel, self.first_sample, causes.join(", "), self.flags).into())
    }
}

/// One completed bounded submission. Samples advance only at a whole-image
/// batch boundary; partially updated pixels are for display, not film export.
/// Preview reports cumulative completed paths for the full current batch;
/// offline mode reports completed paths within the current bounded tile.
#[derive(Clone, Copy, Debug)]
pub struct WorkProgress {
    pub samples_per_pixel: u32,
    pub completed_paths: u32,
    pub total_paths: u32,
    pub tile_start: u32,
    pub total_pixels: u32,
    pub batch_samples: u32,
    pub batch_finished: bool,
}
#[derive(Clone, Copy, Debug)]
struct WorkBatch {
    count: u32,
    tile_start: u32,
    tile_pixels: u32,
    initialized: bool,
}
struct WorkTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    whole_group: wgpu::Buffer,
}

pub struct ProgressiveRenderer {
    params: Params,
    pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    volume_buffers: [wgpu::Buffer; 4],
    default_environment: wgpu::TextureView,
    uniform: wgpu::Buffer,
    mean: wgpu::Buffer,
    m2: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    path_states: wgpu::Buffer,
    timer: Option<WorkTimer>,
    film_bytes: u64,
    sample_batch_capacity: u32,
    sample_batch_bytes: u64,
    samples: u32,
    target_spp: u32,
    poisoned: Cell<bool>,
    work_capacity: u32,
    full_frame_preview: bool,
    active: Option<WorkBatch>,
    in_flight: Option<[u32; 4]>,
    epoch: u64,
    serial: u64,
    encoded_work: bool,
    encoded_chunks: u32,
}

impl ProgressiveRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &SparseGpuVolume,
        camera: &Camera,
        settings: &TransportSettings,
        config: &RenderConfig,
    ) -> Result<Self> {
        Self::new_with_schedule(device, queue, volume, camera, settings, config, false)
    }

    /// Interactive estimator with up to four persistent paths per film pixel. Bounded
    /// submissions visit a fixed permutation of pixels in round-robin order,
    /// spreading each submission over the image and advancing each
    /// slice without waiting for its longest path. Completed pixels commit in
    /// sample order; a completed full frame remains identical to offline mode.
    /// `sample_batch_size` selects 1..=4 parallel samples; zero retains one.
    /// Insufficient full-frame storage is an error, never a tiled fallback.
    pub fn new_preview(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &SparseGpuVolume,
        camera: &Camera,
        settings: &TransportSettings,
        config: &RenderConfig,
    ) -> Result<Self> {
        Self::new_with_schedule(device, queue, volume, camera, settings, config, true)
    }

    fn new_with_schedule(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume: &SparseGpuVolume,
        camera: &Camera,
        settings: &TransportSettings,
        config: &RenderConfig,
        full_frame_preview: bool,
    ) -> Result<Self> {
        config.validate()?;
        settings.validate()?;
        if config.spp > MAX_EXACT_SAMPLES {
            return Err(format!("GPU reference supports at most {MAX_EXACT_SAMPLES} exactly counted samples per pixel; use the f64 CPU backend for longer runs").into());
        }
        validate_volume(volume)?;
        let mut params = Params::zeroed();
        params.image = [config.width, config.height, 0, settings.roulette_start];
        params.control = [
            config.seed as u32,
            (config.seed >> 32) as u32,
            0,
            u32::try_from(volume.tiles.len()).map_err(|_| "too many GPU density tiles")?,
        ];
        params.storage = [
            u32::try_from(volume.hash.len() - 1).map_err(|_| "GPU hash capacity exceeds u32")?,
            u32::try_from(volume.majorant_hash.len() - 1)
                .map_err(|_| "GPU majorant hash capacity exceeds u32")?,
            0,
            0,
        ];
        params.index_min = vec4(volume.index_bounds.min, 0.0)?;
        params.index_max = vec4(volume.index_bounds.max, 0.0)?;
        params.transform = vec4(volume.transform.translation, volume.transform.scale)?;
        // A little more than one ULP is needed for composed f32 trilinear
        // operations. This increases null events; it never clamps density.
        params.optics[1] = volume.majorant * (1.0 + 1.0 / 1_048_576.0);
        configure(&mut params, camera, settings)?;
        if full_frame_preview {
            params.storage[3] |= FULL_FRAME_PREVIEW;
            // Unused bounds padding carries an exact small integer. The
            // multiplication is guarded by preview_pixel_stride on the CPU.
            params.index_min[3] = preview_pixel_stride(config.width * config.height) as f32;
        }
        let film_bytes = u64::from(config.width) * u64::from(config.height) * 16;
        let limits = device.limits();
        let sample_batch_capacity = if full_frame_preview {
            preview_batch_capacity(config)?
        } else {
            batch_capacity(config, &limits)?
        };
        let work_capacity = work_pool_capacity(&limits)?;
        if sample_batch_capacity > work_capacity {
            return Err("cloud sample batch exceeds the bounded dispatch capacity".into());
        }
        let stored_paths = if full_frame_preview {
            (config.width * config.height)
                .checked_mul(sample_batch_capacity)
                .ok_or("full-frame cloud path indexing exceeds u32")?
        } else {
            work_capacity
        };
        let sample_batch_bytes = u64::from(stored_paths) * PATH_STATE_BYTES;
        if full_frame_preview
            && (sample_batch_bytes > u64::from(limits.max_storage_buffer_binding_size)
                || sample_batch_bytes > limits.max_buffer_size)
        {
            return Err(format!("full-frame cloud preview requires {sample_batch_bytes} bytes of path state, exceeding device storage limits; choose a lower preview resolution").into());
        }
        let storage_sizes = [
            film_bytes,
            (volume.hash.len() as u64) * 16,
            ((volume.values.len() as u64) * 4).max(32),
            ((volume.tiles.len() as u64) * 32).max(32),
            (volume.majorant_hash.len() as u64) * 16,
            sample_batch_bytes,
        ];
        for bytes in storage_sizes {
            if bytes > u64::from(limits.max_storage_buffer_binding_size)
                || bytes > limits.max_buffer_size
            {
                return Err(format!("cloud GPU storage buffer requires {bytes} bytes, device binding limit is {} and buffer limit is {}; choose a lower-resolution VDB or request supported adapter limits",
                    limits.max_storage_buffer_binding_size, limits.max_buffer_size).into());
            }
        }
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_bindings_per_bind_group < 11
            || limits.max_sampled_textures_per_shader_stage < 2
            || limits.max_uniform_buffer_binding_size < std::mem::size_of::<Params>() as u64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_invocations_per_workgroup < 64
            || work_capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err("GPU device limits cannot accommodate the cloud compute dispatch".into());
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let hash = storage_buffer(
            device,
            "cloud brick hash",
            bytemuck::cast_slice(&volume.hash),
        );
        let values = storage_buffer(
            device,
            "cloud lattice densities",
            bytemuck::cast_slice(&volume.values),
        );
        let tiles = storage_buffer(
            device,
            "cloud uniform tiles",
            bytemuck::cast_slice(&volume.tiles),
        );
        let majorant_hash = storage_buffer(
            device,
            "cloud spatial majorants",
            bytemuck::cast_slice(&volume.majorant_hash),
        );
        let mean = film_buffer(device, "cloud running mean", film_bytes);
        let m2 = film_buffer(device, "cloud sample M2", film_bytes);
        let diagnostics = film_buffer(
            device,
            "cloud fatal diagnostics and work ticket",
            WORK_PROGRESS_BYTES,
        );
        let path_states = film_buffer(device, "cloud resumable paths", sample_batch_bytes);
        let timer = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| WorkTimer {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("cloud sample or batch timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: MAX_WORK_GROUP_CHUNKS * 2,
                }),
                resolved: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cloud timestamp resolve"),
                    size: u64::from(MAX_WORK_GROUP_CHUNKS) * 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                whole_group: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cloud whole work group timestamps"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud unbiased delta tracking"),
            source: wgpu::ShaderSource::Wgsl(TRACE_SHADER.into()),
        });
        // Every entry point uses this explicit layout, including bindings it
        // does not access. An automatic layout would remove unused bindings.
        let mut layout_entries: Vec<_> = (0..=8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1 | 2 | 3 | 7),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        layout_entries.extend((9..=10).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud common compute bindings"),
            entries: &layout_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud common compute layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = make_pipeline("cloud bounded path continuation", "trace_work");
        let reduce_pipeline =
            make_pipeline("cloud completed-pixel ordered accumulation", "reduce_work");
        // A valid binding remains necessary even when the shader takes the
        // original constant-sky path. Wgpu initializes this texture to zero.
        let default_environment = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("cloud disabled environment placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud path tracing scene"),
            layout: &layout,
            entries: &[
                entry(0, &uniform),
                entry(1, &hash),
                entry(2, &values),
                entry(3, &tiles),
                entry(4, &mean),
                entry(5, &m2),
                entry(6, &diagnostics),
                entry(7, &majorant_hash),
                entry(8, &path_states),
                texture_entry(9, &default_environment),
                texture_entry(10, &default_environment),
            ],
        });
        let result = Self {
            params,
            pipeline,
            reduce_pipeline,
            bind_group,
            bind_group_layout: layout,
            volume_buffers: [hash, values, tiles, majorant_hash],
            default_environment,
            uniform,
            mean,
            m2,
            diagnostics,
            path_states,
            timer,
            film_bytes,
            sample_batch_capacity,
            sample_batch_bytes,
            samples: 0,
            target_spp: config.spp,
            poisoned: Cell::new(false),
            work_capacity,
            full_frame_preview,
            active: None,
            in_flight: None,
            epoch: 0,
            serial: 0,
            encoded_work: false,
            encoded_chunks: 0,
        };
        result.clear(device, queue);
        Ok(result)
    }

    /// Encode one finite continuation chunk, not an entire sample. Exactly one
    /// chunk may be in flight. After submission, copy `progress_buffer` and call
    /// `complete_work` with its mapped bytes before encoding another chunk.
    /// `count` is the logical sample batch and stays fixed until it completes.
    pub fn encode_work(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
    ) -> Result<()> {
        self.encode_work_group(device, queue, encoder, count, 1)
    }

    /// Encode 1..=8 bounded preview chunks into one submission and await only
    /// its final progress ticket. Every kernel still advances at most 4096
    /// paths by 128 transitions. Uniform snapshots are copied in encoder order.
    /// A group never starts another logical batch: if all paths finish early,
    /// the remaining chunks only revisit completed states and commit nothing.
    /// Offline mode permits only one chunk, retaining its original scheduling.
    pub fn encode_work_group(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        chunks: u32,
    ) -> Result<()> {
        validate_work_group(chunks, self.full_frame_preview)?;
        if self.poisoned.get() {
            return Err("cloud film is invalid; reset before rendering again".into());
        }
        if self.in_flight.is_some() {
            return Err("cloud work is already in flight; await its progress before submitting another chunk".into());
        }
        validate_batch_count(
            count,
            self.sample_batch_capacity,
            self.samples,
            self.target_spp,
        )?;
        if let Some(active) = self.active {
            if active.count != count {
                return Err("cannot change sample batch while its paths are pending".into());
            }
        } else {
            self.active = Some(WorkBatch {
                count,
                tile_start: 0,
                tile_pixels: tile_pixels(
                    self.params.image[0] * self.params.image[1],
                    0,
                    count,
                    self.work_capacity,
                ),
                initialized: false,
            });
        }
        let active = self.active.as_mut().unwrap();
        let initialize = !active.initialized;
        if !active.initialized {
            encoder.clear_buffer(&self.path_states, 0, None);
            active.initialized = true;
        }
        self.serial = self
            .serial
            .checked_add(1)
            .ok_or("cloud work serial exhausted")?;
        self.params.image[2] = self.samples;
        self.params.work = [
            self.epoch as u32,
            (self.epoch >> 32) as u32,
            self.serial as u32,
            (self.serial >> 32) as u32,
        ];
        for chunk in 0..chunks {
            let active = self.active.as_ref().unwrap();
            self.params.batch = [
                count,
                active.tile_start,
                active.tile_pixels,
                if self.full_frame_preview {
                    PREVIEW_WORK_TRANSITIONS
                } else {
                    WORK_TRANSITIONS
                },
            ];
            let upload = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cloud chunk constants and epoch ticket"),
                contents: bytemuck::bytes_of(&self.params),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &upload,
                0,
                &self.uniform,
                0,
                std::mem::size_of::<Params>() as u64,
            );
            // Preview accumulates each newly finished pixel exactly once across
            // all slices; offline counts the current tile's terminal paths anew.
            if chunk == 0 && (!self.full_frame_preview || initialize) {
                encoder.clear_buffer(&self.diagnostics, 12, Some(4));
            }
            encoder.copy_buffer_to_buffer(
                &upload,
                std::mem::offset_of!(Params, work) as u64,
                &self.diagnostics,
                16,
                16,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cloud bounded continuation"),
                    timestamp_writes: self.timer.as_ref().map(|timer| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &timer.queries,
                            beginning_of_pass_write_index: Some(chunk * 2),
                            end_of_pass_write_index: None,
                        }
                    }),
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups((count * active.tile_pixels).div_ceil(64), 1, 1);
            }
            {
                // Always execute this pass, including pending tiles, to initialize
                // timestamp1. Each pixel commits only its completed sample prefix.
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cloud completed sample-prefix reduction"),
                    timestamp_writes: self.timer.as_ref().map(|timer| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &timer.queries,
                            beginning_of_pass_write_index: None,
                            end_of_pass_write_index: Some(chunk * 2 + 1),
                        }
                    }),
                });
                pass.set_pipeline(&self.reduce_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(active.tile_pixels.div_ceil(64), 1, 1);
            }
            if chunk + 1 < chunks {
                advance_preview_slice(
                    self.active.as_mut().unwrap(),
                    self.params.image[0] * self.params.image[1],
                    self.work_capacity,
                );
            }
        }
        if let Some(timer) = &self.timer {
            encoder.resolve_query_set(&timer.queries, 0..chunks * 2, &timer.resolved, 0);
            encoder.copy_buffer_to_buffer(&timer.resolved, 0, &timer.whole_group, 0, 8);
            encoder.copy_buffer_to_buffer(
                &timer.resolved,
                u64::from(chunks) * 16 - 8,
                &timer.whole_group,
                8,
                8,
            );
        }
        self.in_flight = Some(self.params.work);
        self.encoded_work = true;
        self.encoded_chunks = chunks;
        Ok(())
    }
    /// Compatibility alias for one bounded chunk of a one-sample batch. This
    /// does not promise a completed sample; commit mapped progress afterward.
    pub fn encode_sample(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        self.encode_work(device, queue, encoder, 1)
    }
    /// Compatibility alias. The returned count names the pending logical batch,
    /// not completed spp. Use `complete_work` to observe actual completion.
    pub fn encode_samples(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
    ) -> Result<u32> {
        self.encode_work(device, queue, encoder, count)?;
        Ok(count)
    }
    pub fn sample_batch_capacity(&self) -> u32 {
        self.sample_batch_capacity
    }
    /// Persistent continuation storage. Preview keeps batch-capacity states per pixel;
    /// offline mode uses the fixed bounded pool.
    pub fn sample_batch_storage_bytes(&self) -> u64 {
        self.sample_batch_bytes
    }
    pub fn progress_buffer(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }
    pub fn has_pending_work(&self) -> bool {
        self.active.is_some()
    }
    /// Pure CPU completion, suitable after a nonblocking staging-map callback.
    /// Epoch/serial validation rejects an old scene or duplicate completion.
    pub fn complete_work(&mut self, bytes: &[u8]) -> Result<WorkProgress> {
        let ticket = self
            .in_flight
            .ok_or("no cloud work is awaiting completion")?;
        let words = progress_words(bytes, ticket)?;
        let diagnostic = Diagnostics {
            flags: words[0],
            first_pixel: words[1],
            first_sample: words[2],
        };
        if let Err(error) = diagnostic.validate() {
            self.poisoned.set(true);
            self.in_flight = None;
            return Err(error);
        }
        let active = self.active.as_mut().ok_or("missing cloud active batch")?;
        let total_pixels = self.params.image[0] * self.params.image[1];
        let total_paths = if self.full_frame_preview {
            active.count * total_pixels
        } else {
            active.count * active.tile_pixels
        };
        if words[3] > total_paths {
            self.poisoned.set(true);
            self.in_flight = None;
            return Err("invalid cloud completed path count".into());
        }
        let mut progress = WorkProgress {
            samples_per_pixel: self.samples,
            completed_paths: words[3],
            total_paths,
            tile_start: active.tile_start,
            total_pixels,
            batch_samples: active.count,
            batch_finished: false,
        };
        self.in_flight = None;
        if self.full_frame_preview {
            if words[3] == total_paths {
                self.samples += active.count;
                self.active = None;
                progress.samples_per_pixel = self.samples;
                progress.batch_finished = true;
            } else {
                advance_preview_slice(active, total_pixels, self.work_capacity);
            }
        } else if words[3] == total_paths {
            if advance_tile(active, progress.total_pixels, self.work_capacity) {
                self.samples += active.count;
                self.active = None;
                progress.samples_per_pixel = self.samples;
                progress.batch_finished = true;
            }
        }
        Ok(progress)
    }
    /// Blocking headless helper. Interactive callers must map progress
    /// asynchronously and invoke `complete_work` instead.
    pub fn read_progress(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<WorkProgress> {
        let bytes = read_buffers(device, queue, &[(&self.diagnostics, WORK_PROGRESS_BYTES)])?;
        self.complete_work(&bytes[0])
    }

    pub fn reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        settings: &TransportSettings,
    ) -> Result<()> {
        if self.in_flight.is_some() {
            return Err("await the current bounded chunk before resetting the cloud scene".into());
        }
        let mut next = self.params;
        configure(&mut next, camera, settings)?;
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or("cloud scene epoch exhausted")?;
        self.params = next;
        self.clear_accumulation(device, queue, epoch);
        Ok(())
    }

    /// Replace the infinite directional boundary and delta-sun irradiance.
    /// Both views must be D2 float textures, with finite, nonnegative linear
    /// RGB. `environment` is an equirectangular table: world Y is up,
    /// u=atan2(direction.x,direction.z)/(2*pi) wrapped, v=acos(direction.y)/pi.
    /// The sunlight view is exactly 1x1 and excludes the solar disk from the
    /// environment, since the path tracer already estimates a delta sun.
    ///
    /// Change bindings only immediately after creation or `reset` (zero
    /// samples, no pending work). This setter submits no GPU work; wait for the
    /// preceding reset clear, then encode environment generation before tracing.
    /// Their contents must remain unchanged until reset, including during
    /// partial batches.
    /// Reset preserves these bindings. Passing None restores offline constants.
    pub fn set_environment(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        environment: Option<(&wgpu::TextureView, &wgpu::TextureView)>,
    ) -> Result<()> {
        if self.in_flight.is_some()
            || self.active.is_some()
            || self.samples != 0
            || self.poisoned.get()
        {
            return Err("reset the cloud film before changing its environment".into());
        }
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or("cloud scene epoch exhausted")?;
        let (sky, sun) =
            environment.unwrap_or((&self.default_environment, &self.default_environment));
        let [hash, values, tiles, majorants] = &self.volume_buffers;
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud path tracing scene and directional boundary"),
            layout: &self.bind_group_layout,
            entries: &[
                entry(0, &self.uniform),
                entry(1, hash),
                entry(2, values),
                entry(3, tiles),
                entry(4, &self.mean),
                entry(5, &self.m2),
                entry(6, &self.diagnostics),
                entry(7, majorants),
                entry(8, &self.path_states),
                texture_entry(9, sky),
                texture_entry(10, sun),
            ],
        });
        if environment.is_some() {
            self.params.storage[3] |= DIRECTIONAL_ENVIRONMENT;
        } else {
            self.params.storage[3] &= !DIRECTIONAL_ENVIRONMENT;
        }
        self.epoch = epoch;
        Ok(())
    }

    /// Change the stopping target after a reset. The continuation pool and
    /// logical batch capacity remain fixed; rendering more samples reuses them.
    pub fn set_target_samples(&mut self, spp: u32) -> Result<()> {
        validate_target_change(
            spp,
            self.samples,
            self.active.is_some() || self.in_flight.is_some(),
        )?;
        self.target_spp = spp;
        Ok(())
    }

    fn clear_accumulation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, epoch: u64) {
        self.samples = 0;
        self.params.image[2] = 0;
        self.epoch = epoch;
        self.active = None;
        self.encoded_work = false;
        self.encoded_chunks = 0;
        self.poisoned.set(false);
        self.clear(device, queue);
    }

    fn clear(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloud film reset"),
        });
        encoder.clear_buffer(&self.mean, 0, None);
        encoder.clear_buffer(&self.m2, 0, None);
        encoder.clear_buffer(&self.diagnostics, 0, None);
        encoder.clear_buffer(&self.path_states, 0, None);
        queue.submit([encoder.finish()]);
    }

    pub fn read_diagnostics(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Diagnostics> {
        let bytes = read_buffers(device, queue, &[(&self.diagnostics, WORK_PROGRESS_BYTES)])?;
        let words: &[u32] = bytemuck::cast_slice(&bytes[0]);
        let value = Diagnostics {
            flags: words[0],
            first_pixel: words[1],
            first_sample: words[2],
        };
        if let Err(error) = value.validate() {
            self.poisoned.set(true);
            return Err(error);
        }
        Ok(value)
    }

    /// GPU time for the latest submitted work group, including every bounded
    /// trace/reduction pair and the ordered uniform copies between them.
    /// Setup and readbacks are excluded. Sum groups for a complete batch.
    /// Absent when timestamp queries are unsupported by the selected device.
    pub fn read_sample_milliseconds(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Option<f64>> {
        let Some(timer) = &self.timer else {
            return Ok(None);
        };
        if !self.encoded_work {
            return Err("no cloud work was submitted for timing".into());
        }
        let bytes = read_buffers(device, queue, &[(&timer.whole_group, 16)])?;
        let ticks: &[u64] = bytemuck::cast_slice(&bytes[0]);
        let elapsed = ticks[1]
            .checked_sub(ticks[0])
            .ok_or("cloud GPU timestamp ordering is invalid")?;
        Ok(Some(
            elapsed as f64 * f64::from(queue.get_timestamp_period()) * 1e-6,
        ))
    }

    pub fn read_work_milliseconds(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Option<f64>> {
        self.read_sample_milliseconds(device, queue)
    }

    /// Two u64 GPU ticks (group start, group end), ready for an asynchronous
    /// copy alongside `progress_buffer` after encoding the group. Use the
    /// queue's timestamp period to convert their difference to milliseconds.
    pub fn work_timing_buffer(&self) -> Option<&wgpu::Buffer> {
        self.timer.as_ref().map(|timer| &timer.whole_group)
    }
    pub fn work_group_size(&self) -> u32 {
        self.encoded_chunks
    }
    /// Ordered per-chunk timestamp pairs and their valid byte count. Copy them
    /// in the same encoder as final progress and decode after its completion.
    pub fn work_chunk_timing_buffer(&self) -> Option<(&wgpu::Buffer, u64)> {
        self.timer
            .as_ref()
            .map(|timer| (&timer.resolved, u64::from(self.encoded_chunks) * 16))
    }
    /// Blocking headless diagnostic for the individual dispatch-pair durations.
    /// Interactive callers should read `work_timing_buffer` asynchronously.
    pub fn read_work_chunk_milliseconds(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Option<Vec<f64>>> {
        let Some(timer) = &self.timer else {
            return Ok(None);
        };
        if !self.encoded_work {
            return Err("no cloud work was submitted for timing".into());
        }
        let bytes = read_buffers(
            device,
            queue,
            &[(&timer.resolved, u64::from(self.encoded_chunks) * 16)],
        )?;
        let ticks: &[u64] = bytemuck::cast_slice(&bytes[0]);
        let period = f64::from(queue.get_timestamp_period()) * 1e-6;
        ticks
            .chunks_exact(2)
            .map(|pair| {
                pair[1]
                    .checked_sub(pair[0])
                    .map(|elapsed| elapsed as f64 * period)
                    .ok_or_else(|| "cloud GPU timestamp ordering is invalid".into())
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Checks the fatal flags before converting statistics into a reference.
    pub fn read_film(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Film> {
        if self.active.is_some() || self.in_flight.is_some() {
            return Err(
                "cloud film has a partial batch; export at a whole-image completion boundary"
                    .into(),
            );
        }
        if self.samples == 0 {
            return Err("cloud film has no samples".into());
        }
        let bytes = read_buffers(
            device,
            queue,
            &[
                (&self.diagnostics, WORK_PROGRESS_BYTES),
                (&self.mean, self.film_bytes),
                (&self.m2, self.film_bytes),
            ],
        )?;
        let words: &[u32] = bytemuck::cast_slice(&bytes[0]);
        let diagnostic = Diagnostics {
            flags: words[0],
            first_pixel: words[1],
            first_sample: words[2],
        };
        if let Err(error) = diagnostic.validate() {
            self.poisoned.set(true);
            return Err(error);
        }
        self.decode_film_readback(&bytes[1], &bytes[2])
    }
    /// Pure CPU decoder for asynchronous viewer exports. Each source must be a
    /// complete film buffer copied at a whole-image batch boundary.
    pub fn decode_film_readback(&self, mean: &[u8], m2: &[u8]) -> Result<Film> {
        if self.poisoned.get()
            || self.active.is_some()
            || self.in_flight.is_some()
            || self.samples == 0
        {
            return Err("cloud reference export requires a valid, complete film batch".into());
        }
        if mean.len() as u64 != self.film_bytes || m2.len() as u64 != self.film_bytes {
            return Err("cloud film readback byte count is invalid".into());
        }
        let means: &[[f32; 4]] =
            bytemuck::try_cast_slice(mean).map_err(|_| "cloud mean readback is misaligned")?;
        let m2: &[[f32; 4]] =
            bytemuck::try_cast_slice(m2).map_err(|_| "cloud M2 readback is misaligned")?;
        if means.iter().any(|value| value[3] != self.samples as f32) {
            return Err("cloud film readback has stale or mixed sample counts".into());
        }
        let film = Film {
            width: self.params.image[0],
            height: self.params.image[1],
            samples_per_pixel: self.samples,
            mean: means.iter().map(|v| [v[0], v[1], v[2]]).collect(),
            sample_variance: m2
                .iter()
                .map(|v| {
                    if self.samples > 1 {
                        [v[0], v[1], v[2]].map(|x| x / (self.samples - 1) as f32)
                    } else {
                        [0.0; 3]
                    }
                })
                .collect(),
        };
        film.validate()?;
        Ok(film)
    }

    /// RGB running mean; display transformations must not modify this buffer.
    pub fn film_buffer(&self) -> &wgpu::Buffer {
        &self.mean
    }
    pub fn film_variance_buffer(&self) -> &wgpu::Buffer {
        &self.m2
    }
    pub fn sample_count(&self) -> u32 {
        self.samples
    }
    pub fn target_samples(&self) -> u32 {
        self.target_spp
    }
    pub fn size(&self) -> [u32; 2] {
        [self.params.image[0], self.params.image[1]]
    }
}

fn work_pool_capacity(limits: &wgpu::Limits) -> Result<u32> {
    let capacity = u64::from(MAX_WORK_PATHS)
        .min(
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
                / PATH_STATE_BYTES,
        )
        .min(u64::from(limits.max_compute_workgroups_per_dimension) * 64);
    if capacity == 0 {
        return Err("GPU cannot store even one cloud continuation state".into());
    }
    Ok(capacity as u32)
}
fn batch_capacity(config: &RenderConfig, limits: &wgpu::Limits) -> Result<u32> {
    let pixels = config
        .width
        .checked_mul(config.height)
        .filter(|&v| v > 0)
        .ok_or("cloud batch requires nonzero u32 film indexing")?;
    if config.spp == 0 {
        return Err("cloud batch requires positive samples per pixel".into());
    }
    let pool = work_pool_capacity(limits)?;
    let requested = if config.sample_batch_size == 0 {
        AUTO_PARALLEL_PATHS.div_ceil(pixels).min(AUTO_MAX_BATCH)
    } else {
        config.sample_batch_size
    }
    .min(config.spp);
    if config.sample_batch_size == 0 {
        return Ok(requested.min(pool).max(1));
    }
    if requested > pool {
        return Err(format!("requested cloud sample batch {requested} exceeds bounded path pool {pool}; use automatic sizing or a smaller batch").into());
    }
    Ok(requested)
}
fn preview_batch_capacity(config: &RenderConfig) -> Result<u32> {
    let requested = config.sample_batch_size.max(1);
    if requested > MAX_PREVIEW_BATCH {
        return Err(format!(
            "preview sample batch must be in1..={MAX_PREVIEW_BATCH}; zero selects1"
        )
        .into());
    }
    Ok(requested.min(config.spp))
}
fn validate_work_group(chunks: u32, preview: bool) -> Result<()> {
    if chunks == 0 || chunks > MAX_WORK_GROUP_CHUNKS {
        return Err(format!(
            "cloud work group must contain1..={MAX_WORK_GROUP_CHUNKS} bounded chunks"
        )
        .into());
    }
    if !preview && chunks != 1 {
        return Err("multi-chunk work groups require full-frame preview scheduling".into());
    }
    Ok(())
}
fn tile_pixels(total: u32, start: u32, samples: u32, pool: u32) -> u32 {
    (total - start).min(pool / samples)
}
fn advance_tile(active: &mut WorkBatch, total_pixels: u32, pool: u32) -> bool {
    active.tile_start += active.tile_pixels;
    if active.tile_start == total_pixels {
        return true;
    }
    active.tile_pixels = tile_pixels(total_pixels, active.tile_start, active.count, pool);
    active.initialized = false;
    false
}
fn advance_preview_slice(active: &mut WorkBatch, total_pixels: u32, pool: u32) {
    active.tile_start += active.tile_pixels;
    if active.tile_start == total_pixels {
        active.tile_start = 0;
    }
    active.tile_pixels = (total_pixels - active.tile_start).min(pool / active.count);
    // The full-image path state persists across slices and sweeps.
    debug_assert!(active.initialized);
}
fn preview_pixel_stride(pixels: u32) -> u32 {
    let mut stride = 511.min(u32::MAX / pixels.saturating_sub(1).max(1));
    if stride % 2 == 0 {
        stride -= 1;
    }
    loop {
        let (mut a, mut b) = (pixels, stride);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        if a == 1 {
            return stride;
        }
        stride -= 2;
    }
}
fn progress_words(bytes: &[u8], ticket: [u32; 4]) -> Result<[u32; 8]> {
    if bytes.len() != WORK_PROGRESS_BYTES as usize {
        return Err("cloud progress must contain exactly32 bytes".into());
    }
    let mut words = [0u32; 8];
    for (word, source) in words.iter_mut().zip(bytes.chunks_exact(4)) {
        *word = u32::from_le_bytes(source.try_into().unwrap());
    }
    if words[4..] != ticket {
        return Err("stale or mismatched cloud scene/submission progress".into());
    }
    Ok(words)
}

fn validate_batch_count(count: u32, capacity: u32, samples: u32, target_spp: u32) -> Result<()> {
    if count == 0 || count > capacity {
        return Err(format!("cloud sample batch count {count} must be in 1..={capacity}").into());
    }
    if samples
        .checked_add(count)
        .is_none_or(|end| end > target_spp)
    {
        return Err("cloud sample batch exceeds its configured samples per pixel".into());
    }
    Ok(())
}

fn validate_target_change(spp: u32, samples: u32, pending: bool) -> Result<()> {
    if spp == 0 || spp > MAX_EXACT_SAMPLES {
        return Err(format!("GPU sample target must be in 1..={MAX_EXACT_SAMPLES}").into());
    }
    if samples != 0 || pending {
        return Err("reset the cloud film before changing its sample target".into());
    }
    Ok(())
}

fn configure(params: &mut Params, camera: &Camera, settings: &TransportSettings) -> Result<()> {
    settings.validate()?;
    let scale = f64::from(params.transform[3]);
    if !params.transform[3].is_normal() || !params.transform[3].recip().is_normal() {
        return Err(
            "GPU uniform scale and its reciprocal must be normal f32 values; use CPU f64".into(),
        );
    }
    if camera
        .origin
        .to_array()
        .into_iter()
        .any(|v| (v / scale).abs() > MAX_INDEX_MAGNITUDE)
        || settings
            .ground
            .is_some_and(|g| (g.height / scale).abs() > MAX_INDEX_MAGNITUDE)
    {
        return Err(
            "GPU camera/ground coordinates exceed the voxel precision budget; use CPU f64".into(),
        );
    }
    let (forward, right, up) = camera.basis()?;
    params.camera_origin = vec4(
        camera.origin,
        (camera.horizontal_fov_deg.to_radians() * 0.5).tan(),
    )?;
    params.camera_forward = vec4(forward, 0.0)?;
    params.camera_right = vec4(right, 0.0)?;
    params.camera_up = vec4(up, 0.0)?;
    params.image[3] = settings.roulette_start;
    params.control[2] =
        u32::try_from(settings.event_limit).map_err(|_| "GPU event watchdog must fit u32")?;
    params.optics[0] = finite_f32(settings.extinction_scale)?;
    params.optics[2] = finite_f32(settings.phase_g)?;
    if params.optics[2].abs() >= 1.0 {
        return Err("HG g rounds to ±1 in GPU f32; use the f64 CPU backend".into());
    }
    let rate = params.optics[0] * params.optics[1];
    if !params.optics[1].is_finite()
        || !rate.is_finite()
        || rate < 0.0
        || (settings.extinction_scale > 0.0 && params.optics[0] == 0.0)
    {
        return Err("GPU majorant extinction must be finite and representable in f32".into());
    }
    params.albedo = vec4(settings.scattering_albedo, 0.0)?;
    let sun_direction = settings.sun_direction.normalize();
    if !sun_direction.is_finite() || sun_direction.length_squared() == 0.0 {
        return Err("sun direction cannot be normalized in finite arithmetic".into());
    }
    params.sun_direction = vec4(sun_direction, 0.0)?;
    params.sun_irradiance = vec4(settings.sun_irradiance, 0.0)?;
    params.sky_radiance = vec4(settings.sky_radiance, 0.0)?;
    params.storage[3] &= DIRECTIONAL_ENVIRONMENT | FULL_FRAME_PREVIEW;
    if let Some(ground) = &settings.ground {
        params.storage[3] |= 1;
        params.optics[3] = finite_f32(ground.height)?;
        params.ground_albedo = vec4(ground.albedo, 0.0)?;
    } else {
        params.optics[3] = 0.0;
        params.ground_albedo = [0.0; 4];
    }
    if !settings.spatial_majorants {
        params.storage[3] |= 2;
    }
    if settings.shadow_roulette {
        params.storage[3] |= 4;
    }
    debug_assert_eq!(
        params.storage[2], 0,
        "TwoSum bit barrier must remain the identity"
    );
    Ok(())
}

fn validate_volume(volume: &SparseGpuVolume) -> Result<()> {
    if volume.majorant_hash.is_empty()
        || !volume.majorant_hash.len().is_power_of_two()
        || !volume
            .majorant_hash
            .iter()
            .any(|entry| entry[3] == u32::MAX)
    {
        return Err(
            "GPU spatial majorant hash requires power-of-two capacity and an empty slot".into(),
        );
    }
    for entry in &volume.majorant_hash {
        if entry[3] != u32::MAX {
            let value = f32::from_bits(entry[3]);
            if !value.is_finite() || value <= 0.0 || value > volume.majorant {
                return Err("GPU cell majorant is invalid or exceeds the global majorant".into());
            }
        }
    }
    if volume.hash.is_empty()
        || !volume.hash.len().is_power_of_two()
        || !volume.hash.iter().any(|entry| entry[3] == u32::MAX)
    {
        return Err("GPU sparse hash requires power-of-two capacity and an empty slot".into());
    }
    if !volume.majorant.is_finite()
        || volume.majorant <= 0.0
        || !volume.transform.scale.is_finite()
        || volume.transform.scale <= 0.0
    {
        return Err("invalid GPU volume transform or density majorant".into());
    }
    let scale = volume.transform.scale as f32;
    if !scale.is_normal()
        || !scale.recip().is_normal()
        || volume
            .transform
            .translation
            .to_array()
            .into_iter()
            .any(|v| !v.is_finite() || (v / f64::from(scale)).abs() > MAX_INDEX_MAGNITUDE)
        || !volume.index_bounds.valid()
    {
        return Err("GPU volume transform exceeds the voxel precision budget; use CPU f64".into());
    }
    for slot in &volume.hash {
        if slot[3] != u32::MAX
            && (slot[3] as usize)
                .checked_add(512)
                .is_none_or(|end| end > volume.values.len())
        {
            return Err("GPU density brick address is out of bounds".into());
        }
    }
    if volume
        .values
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || *v > volume.majorant)
    {
        return Err("GPU density values violate the majorant".into());
    }
    // floor(index/8) and lattice conversion must be exact at voxel resolution.
    for value in volume
        .index_bounds
        .min
        .to_array()
        .into_iter()
        .chain(volume.index_bounds.max.to_array())
    {
        if !value.is_finite() || value.abs() > MAX_INDEX_MAGNITUDE {
            return Err("GPU cloud index coordinates exceed the spatial proposal precision budget; use the f64 CPU backend".into());
        }
    }
    // Paired slab products must themselves be finite, even when a subsequent
    // translation could mathematically cancel an overflowing world plane.
    for axis in 0..3 {
        let translation = volume.transform.translation[axis] as f32;
        for index in [
            volume.index_bounds.min[axis] as f32 - 1.0,
            volume.index_bounds.max[axis] as f32 + 1.0,
        ] {
            let product = index * scale;
            if !product.is_finite() || !(product + translation).is_finite() {
                return Err(
                    "GPU world proposal planes exceed finite f32 arithmetic; use CPU f64".into(),
                );
            }
        }
    }
    for tile in &volume.tiles {
        let value = f32::from_bits(tile[4]);
        if tile[3] == 0
            || tile[3] > i32::MAX as u32
            || !value.is_finite()
            || value < 0.0
            || value > volume.majorant
        {
            return Err("GPU uniform density tile is invalid".into());
        }
        if ![8u32, 128, 4096].contains(&tile[3])
            || tile[..3]
                .iter()
                .any(|v| (*v as i32).rem_euclid(tile[3] as i32) != 0)
        {
            return Err("GPU uniform tiles require aligned 8/128/4096 sample cubes for exact brick interpolation".into());
        }
    }
    Ok(())
}

fn finite_f32(value: f64) -> Result<f32> {
    let packed = value as f32;
    if !packed.is_finite() {
        return Err("cloud parameter cannot be represented as finite GPU f32".into());
    }
    Ok(packed)
}
fn vec4(vector: DVec3, fourth: f64) -> Result<[f32; 4]> {
    Ok([
        finite_f32(vector.x)?,
        finite_f32(vector.y)?,
        finite_f32(vector.z)?,
        finite_f32(fourth)?,
    ])
}
fn storage_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    // Runtime arrays require at least one complete element even for no tiles.
    let empty = [0u8; 32];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: if contents.is_empty() {
            &empty
        } else {
            contents
        },
        usage: wgpu::BufferUsages::STORAGE,
    })
}
fn film_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
fn texture_entry(binding: u32, texture: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(texture),
    }
}
fn read_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sources: &[(&wgpu::Buffer, u64)],
) -> Result<Vec<Vec<u8>>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cloud reference readback"),
    });
    let buffers: Vec<_> = sources
        .iter()
        .map(|(source, size)| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cloud readback"),
                size: *size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, *size);
            buffer
        })
        .collect();
    queue.submit([encoder.finish()]);
    let receivers: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            let (sender, receiver) = mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            receiver
        })
        .collect();
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| error.to_string())?;
    let mut result = Vec::new();
    for (buffer, receiver) in buffers.into_iter().zip(receivers) {
        receiver
            .recv()
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        let view = buffer.get_mapped_range(..);
        result.push(view.to_vec());
        drop(view);
        buffer.unmap();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shaders_validate_without_creating_a_gpu_device() {
        for (name, source) in [("trace", TRACE_SHADER), ("present", PRESENT_SHADER)] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap_or_else(|error| panic!("{name} SPIR-V: {error:?}"));
            if name == "trace" {
                let state = module
                    .types
                    .iter()
                    .find(|(_, ty)| ty.name.as_deref() == Some("PathState"))
                    .unwrap()
                    .1;
                let naga::TypeInner::Struct { span, .. } = state.inner else {
                    panic!("path state is not a struct");
                };
                assert_eq!(u64::from(span), PATH_STATE_BYTES);
                assert_eq!(
                    module
                        .entry_points
                        .iter()
                        .map(|entry| entry.name.as_str())
                        .collect::<Vec<_>>(),
                    ["trace_work", "reduce_work"]
                );
            }
        }
    }
    #[test]
    fn fatal_diagnostics_do_not_allow_reference_export() {
        assert!(
            Diagnostics {
                flags: 0,
                first_pixel: 0,
                first_sample: 0
            }
            .validate()
            .is_ok()
        );
        for flags in [1, 2, 4, 8, 16, 32, 64, 127] {
            assert!(
                Diagnostics {
                    flags,
                    first_pixel: 19,
                    first_sample: 7
                }
                .validate()
                .is_err()
            );
        }
        assert_eq!(std::mem::size_of::<Params>(), 288);
    }
    #[test]
    fn camera_reset_preserves_directional_environment_and_updates_transport_flags() {
        let mut params = Params::zeroed();
        params.transform[3] = 1.0;
        params.optics[1] = 1.0;
        params.storage[3] = DIRECTIONAL_ENVIRONMENT | FULL_FRAME_PREVIEW;
        let mut settings = TransportSettings::default();
        configure(&mut params, &Camera::default(), &settings).unwrap();
        assert_eq!(
            params.storage[3] & DIRECTIONAL_ENVIRONMENT,
            DIRECTIONAL_ENVIRONMENT
        );
        assert_eq!(params.storage[3] & 1, 1);
        assert_eq!(params.storage[3] & FULL_FRAME_PREVIEW, FULL_FRAME_PREVIEW);
        settings.ground = None;
        settings.spatial_majorants = false;
        settings.shadow_roulette = !settings.shadow_roulette;
        configure(&mut params, &Camera::default(), &settings).unwrap();
        assert_eq!(
            params.storage[3] & DIRECTIONAL_ENVIRONMENT,
            DIRECTIONAL_ENVIRONMENT
        );
        assert_eq!(params.storage[3] & 1, 0);
        assert_eq!(params.storage[3] & FULL_FRAME_PREVIEW, FULL_FRAME_PREVIEW);
        assert_eq!(params.storage[3] & 2, 2);
        assert_eq!(params.storage[3] & 4 != 0, settings.shadow_roulette);
        params.storage[3] &= !DIRECTIONAL_ENVIRONMENT;
        configure(&mut params, &Camera::default(), &settings).unwrap();
        assert_eq!(params.storage[3] & DIRECTIONAL_ENVIRONMENT, 0);
    }
    #[test]
    fn target_change_requires_zero_samples_and_no_pending_chunk_or_batch() {
        assert!(validate_target_change(1, 0, false).is_ok());
        assert!(validate_target_change(MAX_EXACT_SAMPLES, 0, false).is_ok());
        assert!(validate_target_change(0, 0, false).is_err());
        assert!(validate_target_change(MAX_EXACT_SAMPLES + 1, 0, false).is_err());
        assert!(validate_target_change(1024, 1, false).is_err());
        assert!(validate_target_change(1024, 0, true).is_err());
    }
    #[test]
    fn directional_boundary_table_is_centered_wrapped_and_clamped() {
        use glam::{Vec2, Vec3};
        // Independent CPU mirror of the shader's manual unfilterable lookup.
        let sample = |direction: Vec3, table: &[[f32; 3]], width: i32, height: i32| {
            let phi = direction
                .x
                .atan2(direction.z)
                .rem_euclid(2.0 * std::f32::consts::PI);
            let uv = Vec2::new(
                phi / (2.0 * std::f32::consts::PI),
                direction.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
            );
            let xy = uv * Vec2::new(width as f32, height as f32) - Vec2::splat(0.5);
            let base = xy.floor();
            let f = xy - base;
            let texel = |x: i32, y: i32| {
                Vec3::from_array(
                    table[(y.clamp(0, height - 1) * width + x.rem_euclid(width)) as usize],
                )
            };
            let a = texel(base.x as i32, base.y as i32);
            let b = texel(base.x as i32 + 1, base.y as i32);
            let c = texel(base.x as i32, base.y as i32 + 1);
            let d = texel(base.x as i32 + 1, base.y as i32 + 1);
            a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y)
        };
        let table: Vec<_> = (0..8)
            .map(|i| [i as f32, (i / 4) as f32, (i % 4) as f32])
            .collect();
        for y in 0..2 {
            for x in 0..4 {
                let phi = (x as f32 + 0.5) / 4.0 * 2.0 * std::f32::consts::PI;
                let theta = (y as f32 + 0.5) / 2.0 * std::f32::consts::PI;
                let direction = Vec3::new(
                    theta.sin() * phi.sin(),
                    theta.cos(),
                    theta.sin() * phi.cos(),
                );
                let actual = sample(direction, &table, 4, 2);
                assert!(
                    (actual - Vec3::from_array(table[y * 4 + x]))
                        .abs()
                        .max_element()
                        < 3e-6
                );
            }
        }
        let left = sample(Vec3::new(-1e-6, 0.0, 1.0).normalize(), &table, 4, 2);
        let right = sample(Vec3::new(1e-6, 0.0, 1.0).normalize(), &table, 4, 2);
        assert!((left - right).abs().max_element() < 1e-5);
        assert_eq!(sample(Vec3::Y, &table, 4, 2).y, 0.0);
        assert_eq!(sample(-Vec3::Y, &table, 4, 2).y, 1.0);
        let constant = [[0.03, 0.07, 0.23]; 1];
        for direction in [Vec3::Y, -Vec3::Y, Vec3::Z, Vec3::X, -Vec3::Z, -Vec3::X] {
            assert_eq!(
                sample(direction, &constant, 1, 1),
                Vec3::from_array(constant[0])
            );
        }
        assert!(sample(Vec3::X, &table, 4, 2).z < sample(-Vec3::X, &table, 4, 2).z);
    }
    #[test]
    fn bounded_pool_tiles_cover_images_and_preserve_sample_order() {
        let limits = wgpu::Limits::default();
        assert_eq!(work_pool_capacity(&limits).unwrap(), 4096);
        for (pixels, samples) in [(640 * 360, 1), (8 * 4, 1024), (4097, 4), (1, 4096)] {
            let mut start = 0;
            let mut covered = 0;
            while start < pixels {
                let size = tile_pixels(pixels, start, samples, 4096);
                assert!(size > 0 && size * samples <= 4096);
                for z in 0..samples {
                    for local in 0..size {
                        let slot = z * size + local;
                        assert_eq!(slot / size, z);
                        assert_eq!(start + slot % size, start + local);
                    }
                }
                covered += size;
                start += size;
            }
            assert_eq!(covered, pixels);
        }
        assert_eq!(
            batch_capacity(&RenderConfig::default(), &limits).unwrap(),
            1
        );
        let mut config = RenderConfig {
            width: 8,
            height: 4,
            ..Default::default()
        };
        assert_eq!(batch_capacity(&config, &limits).unwrap(), 1024);
        config.sample_batch_size = 4097;
        config.spp = 8192;
        assert!(batch_capacity(&config, &limits).is_err());
    }
    #[test]
    fn stale_epoch_and_submission_progress_is_rejected_without_a_device() {
        let mut raw = vec![0u8; 32];
        let ticket = [9, 0, 17, 0];
        for (i, value) in ticket.iter().enumerate() {
            raw[16 + i * 4..20 + i * 4].copy_from_slice(&u32::to_le_bytes(*value));
        }
        assert!(progress_words(&raw, ticket).is_ok());
        assert!(progress_words(&raw, [10, 0, 17, 0]).is_err());
        assert!(progress_words(&raw, [9, 0, 18, 0]).is_err());
        assert!(progress_words(&raw[..16], ticket).is_err());
    }
    #[test]
    fn only_final_tile_completes_the_whole_image_batch() {
        let mut active = WorkBatch {
            count: 4,
            tile_start: 0,
            tile_pixels: 1024,
            initialized: true,
        };
        assert!(!advance_tile(&mut active, 2050, 4096));
        assert_eq!(
            (active.tile_start, active.tile_pixels, active.initialized),
            (1024, 1024, false)
        );
        active.initialized = true;
        assert!(!advance_tile(&mut active, 2050, 4096));
        assert_eq!((active.tile_start, active.tile_pixels), (2048, 2));
        assert!(advance_tile(&mut active, 2050, 4096));
    }
    #[test]
    fn full_frame_preview_visits_all_pixels_before_a_long_path_finishes() {
        let total = 128 * 72;
        let stride = preview_pixel_stride(total);
        let mut active = WorkBatch {
            count: 1,
            tile_start: 0,
            tile_pixels: MAX_WORK_PATHS,
            initialized: true,
        };
        let mut visits = vec![0; total as usize];
        let mut finished = vec![false; total as usize];
        let mut complete = 0;
        // Pixel0 requires seven visits, every other pixel at most two. State
        // survives each unconditional slice advance, including wraparound.
        for dispatch in 0..21 {
            let expected_start = [0, 4096, 8192][dispatch % 3];
            assert_eq!(active.tile_start, expected_start);
            assert!(active.tile_pixels <= MAX_WORK_PATHS);
            for slot in active.tile_start..active.tile_start + active.tile_pixels {
                let pixel = slot * stride % total;
                let i = pixel as usize;
                if !finished[i] {
                    visits[i] += 1;
                    let required = if pixel == 0 { 7 } else { 1 + pixel % 2 };
                    if visits[i] == required {
                        finished[i] = true;
                        complete += 1;
                    }
                }
            }
            assert_eq!(complete, finished.iter().filter(|&&v| v).count() as u32);
            advance_preview_slice(&mut active, total, MAX_WORK_PATHS);
            if dispatch == 2 {
                assert!(visits.iter().all(|&n| n == 1));
                assert!(!finished[0]);
                assert!(complete > 0 && complete < total);
                assert_eq!(active.tile_start, 0);
            }
        }
        assert_eq!(complete, total);
        assert!(finished.iter().all(|&v| v));
        assert!(active.initialized);
    }
    #[test]
    fn preview_permutation_is_bijective_spreads_each_slice_and_cannot_overflow() {
        for pixels in [1, 2, 8, 9, 511, 9216, 1920 * 1080, u32::MAX] {
            let stride = preview_pixel_stride(pixels);
            assert!((1..=511).contains(&stride));
            assert!(pixels.saturating_sub(1).checked_mul(stride).is_some());
            if pixels <= 9216 {
                let mut seen = vec![false; pixels as usize];
                for slot in 0..pixels {
                    let pixel = slot * stride % pixels;
                    assert!(!seen[pixel as usize]);
                    seen[pixel as usize] = true;
                }
                assert!(seen.iter().all(|&v| v));
            }
        }
        let stride = preview_pixel_stride(128 * 72);
        let mut rows = [0; 72];
        for slot in 0..MAX_WORK_PATHS {
            rows[(slot * stride % (128 * 72) / 128) as usize] += 1;
        }
        assert!(
            rows.iter().all(|&n| n > 0),
            "the first chunk must span every image row"
        );
    }
    #[test]
    fn preview_cohorts_keep_every_sample_unique_and_bounded() {
        let mut config = RenderConfig::default();
        assert_eq!(preview_batch_capacity(&config).unwrap(), 1);
        config.sample_batch_size = 4;
        assert_eq!(preview_batch_capacity(&config).unwrap(), 4);
        config.spp = 3;
        assert_eq!(preview_batch_capacity(&config).unwrap(), 3);
        config.sample_batch_size = 5;
        assert!(preview_batch_capacity(&config).is_err());
        for count in 1..=4 {
            let pixels = 128 * 72;
            let stride = preview_pixel_stride(pixels);
            let mut active = WorkBatch {
                count,
                tile_start: 0,
                tile_pixels: tile_pixels(pixels, 0, count, MAX_WORK_PATHS),
                initialized: true,
            };
            let mut seen = vec![false; (pixels * count) as usize];
            loop {
                assert!(active.tile_pixels * count <= MAX_WORK_PATHS);
                for local in 0..active.tile_pixels * count {
                    let pixel_slot = active.tile_start + local % active.tile_pixels;
                    let z = local / active.tile_pixels;
                    let pixel = pixel_slot * stride % pixels;
                    let canonical = (pixel * count + z) as usize;
                    assert!(!seen[canonical]);
                    seen[canonical] = true;
                    let state = pixel_slot * count + z;
                    assert!(state < pixels * count);
                }
                advance_preview_slice(&mut active, pixels, MAX_WORK_PATHS);
                if active.tile_start == 0 {
                    break;
                }
            }
            assert!(seen.iter().all(|&v| v));
        }
    }
    #[test]
    fn work_groups_advance_each_snapshot_without_crossing_a_batch() {
        assert!(validate_work_group(0, true).is_err());
        assert!(validate_work_group(9, true).is_err());
        assert!(validate_work_group(2, false).is_err());
        assert!(validate_work_group(1, false).is_ok());
        for chunks in [1, 4, 8] {
            assert!(validate_work_group(chunks, true).is_ok());
            let mut active = WorkBatch {
                count: 4,
                tile_start: 0,
                tile_pixels: 1024,
                initialized: true,
            };
            let mut cursors = Vec::new();
            for i in 0..chunks {
                cursors.push(active.tile_start);
                assert_eq!(active.count, 4);
                assert!(active.initialized);
                if i + 1 < chunks {
                    advance_preview_slice(&mut active, 2305, MAX_WORK_PATHS);
                }
            }
            assert_eq!(
                &cursors,
                &[0, 1024, 2048, 0, 1024, 2048, 0, 1024][..chunks as usize]
            );
            // Host completion advances only once after the last encoded slice.
            advance_preview_slice(&mut active, 2305, MAX_WORK_PATHS);
            assert_eq!(active.tile_start, [0, 1024, 2048][(chunks % 3) as usize]);
        }
    }
    #[test]
    fn gpu_precision_guards_are_checked_on_cpu() {
        use crate::volume::{SparseVolume, UniformTransform, VolumeStats};
        use std::collections::HashMap;
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 1.0,
                translation: DVec3::ZERO,
            },
            VolumeStats::default(),
            HashMap::from([([0, 0, 0], Box::new([1.0; 512]))]),
            Vec::new(),
        )
        .unwrap();
        let mut packed = volume.pack_gpu().unwrap();
        assert!(validate_volume(&packed).is_ok());
        packed.transform.translation.x = 1e9;
        assert!(validate_volume(&packed).is_err());
        packed.transform.translation.x = 0.0;
        packed.transform.scale = 1e-50;
        assert!(validate_volume(&packed).is_err());
        let mut params = Params::zeroed();
        params.transform[3] = 1.0;
        params.optics[1] = 1.0;
        let mut camera = Camera::default();
        let settings = TransportSettings::default();
        assert!(configure(&mut params, &camera, &settings).is_ok());
        camera.origin.x = 1e9;
        assert!(configure(&mut params, &camera, &settings).is_err());
        camera = Camera::default();
        let mut distant_ground = settings;
        distant_ground.ground.as_mut().unwrap().height = -1e9;
        assert!(configure(&mut params, &camera, &distant_ground).is_err());
    }
    #[test]
    fn compensated_tracking_retains_small_exponential_steps() {
        // Mirrors WGSL TwoSum using only CPU f32 operations; no GPU work. At a
        // long distance, ordinary f32 summation swallows small valid flights.
        let mut rng = crate::sampling::Pcg32::for_sample(17, 0, 0);
        let mut hi = 400.0f32;
        let mut lo = 0.0f32;
        let mut ordinary = hi;
        let mut swallowed = 0u32;
        let mut exact = f64::from(hi);
        for _ in 0..400_000 {
            let u = ((rng.next_u32() >> 9) as f32 + 0.5) / 8_388_608.0;
            let step = -u.ln() / 4.0;
            assert!(step > 0.0);
            let sum = hi + step;
            let recovered = sum - hi;
            let error = (hi - (sum - recovered)) + (step - recovered);
            let tail = lo + error;
            let next_hi = sum + tail;
            let next_lo = tail - (next_hi - sum);
            assert!(next_hi > hi || (next_hi == hi && next_lo > lo));
            hi = next_hi;
            lo = next_lo;
            let next_ordinary = ordinary + step;
            swallowed += u32::from(next_ordinary == ordinary);
            ordinary = next_ordinary;
            exact += f64::from(step);
        }
        assert!(
            swallowed > 0,
            "fixture must exercise the former false stagnation"
        );
        assert!(((f64::from(hi) + f64::from(lo)) - exact).abs() < 1e-5);
    }
}

#[cfg(test)]
#[path = "gpu_math_tests.rs"]
mod geometry_tests;
