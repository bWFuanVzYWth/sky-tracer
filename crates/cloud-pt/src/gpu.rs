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
pub const WORK_PROGRESS_BYTES: u64 = 32;
const PATH_STATE_BYTES: u64 = 288;
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
        Err(format!("cloud GPU film invalid at pixel {}, sample {}: {} (flags 0x{:x}); no reference image can be exported",
            self.first_pixel, self.first_sample, causes.join(", "), self.flags).into())
    }
}

/// One completed bounded submission. Samples advance only at a whole-image
/// batch boundary; partially updated pixels are for display, not film export.
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

pub struct ProgressiveRenderer {
    params: Params,
    pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    mean: wgpu::Buffer,
    m2: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    path_states: wgpu::Buffer,
    timer: Option<(wgpu::QuerySet, wgpu::Buffer)>,
    film_bytes: u64,
    sample_batch_capacity: u32,
    sample_batch_bytes: u64,
    samples: u32,
    target_spp: u32,
    poisoned: Cell<bool>,
    work_capacity: u32,
    active: Option<WorkBatch>,
    in_flight: Option<[u32; 4]>,
    epoch: u64,
    serial: u64,
    encoded_work: bool,
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
        let film_bytes = u64::from(config.width) * u64::from(config.height) * 16;
        let limits = device.limits();
        let sample_batch_capacity = batch_capacity(config, &limits)?;
        let work_capacity = work_pool_capacity(&limits)?;
        let sample_batch_bytes = u64::from(work_capacity) * PATH_STATE_BYTES;
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
            || limits.max_bindings_per_bind_group < 9
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
            .then(|| {
                (
                    device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("cloud sample or batch timestamps"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("cloud timestamp resolve"),
                        size: 16,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                )
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud unbiased delta tracking"),
            source: wgpu::ShaderSource::Wgsl(TRACE_SHADER.into()),
        });
        // Every entry point uses this explicit layout, including bindings it
        // does not access. An automatic layout would remove unused bindings.
        let layout_entries: Vec<_> = (0..=8)
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
            ],
        });
        let result = Self {
            params,
            pipeline,
            reduce_pipeline,
            bind_group,
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
            active: None,
            in_flight: None,
            epoch: 0,
            serial: 0,
            encoded_work: false,
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
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
    ) -> Result<()> {
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
        if !active.initialized {
            encoder.clear_buffer(&self.path_states, 0, None);
            active.initialized = true;
        }
        self.serial = self
            .serial
            .checked_add(1)
            .ok_or("cloud work serial exhausted")?;
        self.params.image[2] = self.samples;
        self.params.batch = [
            count,
            active.tile_start,
            active.tile_pixels,
            WORK_TRANSITIONS,
        ];
        self.params.work = [
            self.epoch as u32,
            (self.epoch >> 32) as u32,
            self.serial as u32,
            (self.serial >> 32) as u32,
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
        encoder.clear_buffer(&self.diagnostics, 12, Some(4));
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
                timestamp_writes: self.timer.as_ref().map(|(query_set, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
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
                timestamp_writes: self.timer.as_ref().map(|(query_set, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.reduce_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(active.tile_pixels.div_ceil(64), 1, 1);
        }
        if let Some((queries, resolved)) = &self.timer {
            encoder.resolve_query_set(queries, 0..2, resolved, 0);
        }
        self.in_flight = Some(self.params.work);
        self.encoded_work = true;
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
    /// Persistent continuation storage, independent of full-film pixel count.
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
        let total_paths = active.count * active.tile_pixels;
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
            total_pixels: self.params.image[0] * self.params.image[1],
            batch_samples: active.count,
            batch_finished: false,
        };
        self.in_flight = None;
        if words[3] == total_paths {
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
        self.params = next;
        self.samples = 0;
        self.params.image[2] = 0;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or("cloud scene epoch exhausted")?;
        self.active = None;
        self.encoded_work = false;
        self.poisoned.set(false);
        self.clear(device, queue);
        Ok(())
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

    /// GPU compute time for the latest bounded chunk (trace + conditional
    /// reduction), excluding setup/readbacks. Sum chunks for a complete batch.
    /// Absent when timestamp queries are unsupported by the selected device.
    pub fn read_sample_milliseconds(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Option<f64>> {
        let Some((_, resolved)) = &self.timer else {
            return Ok(None);
        };
        if !self.encoded_work {
            return Err("no cloud work was submitted for timing".into());
        }
        let bytes = read_buffers(device, queue, &[(resolved, 16)])?;
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
    if let Some(ground) = &settings.ground {
        params.storage[3] = 1;
        params.optics[3] = finite_f32(ground.height)?;
        params.ground_albedo = vec4(ground.albedo, 0.0)?;
    } else {
        params.storage[3] = 0;
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
        for flags in [1, 2, 4, 8, 16, 32, 63] {
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
