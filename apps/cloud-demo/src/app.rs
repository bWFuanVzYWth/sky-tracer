//! Explicit GPU viewer with bounded work, asynchronous readbacks and idle time.
use crate::controller::{Controller, MoveKey, same_pose, sun_angles, sun_direction};
use crate::output::{self, Record};
use crate::sky_bridge::{COMPOSITE_SHADER, SkyBackground};
use crate::viewer_ui::{self, Controls, Progress, ViewerUi};
use bytemuck::{Pod, Zeroable};
use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    gpu::{ProgressiveRenderer, WORK_PROGRESS_BYTES, WorkProgress},
    transport::{GroundPlane, TransportSettings},
    volume::SparseGpuVolume,
};
use image::ImageEncoder;
use std::{
    any::Any,
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread::JoinHandle,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const POLL_INTERVAL: Duration = Duration::from_millis(1);
const MIN_WORK_REST: Duration = Duration::from_millis(1);
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, Default)]
pub struct ViewOptions {
    pub exit_after: Option<Duration>,
    pub smoke_frames: Option<u64>,
    pub capture_preview: Option<PathBuf>,
}
impl ViewOptions {
    fn diagnostic(&self) -> bool {
        self.exit_after.is_some() || self.smoke_frames.is_some() || self.capture_preview.is_some()
    }
}

pub fn run(
    volume: SparseGpuVolume,
    camera: Camera,
    settings: TransportSettings,
    mut config: RenderConfig,
    exposure_ev: f32,
    record: Record,
    options: ViewOptions,
) -> Result<()> {
    camera.basis()?;
    settings.validate()?;
    config.validate()?;
    // Interactive accumulation keeps one persistent path per film pixel. The
    // offline renderer retains its independently configured logical batching.
    config.sample_batch_size = 1;
    if !exposure_ev.is_finite() {
        return Err("display exposure must be finite".into());
    }
    if options
        .capture_preview
        .as_ref()
        .is_some_and(|path| path.exists())
    {
        return Err("--capture-preview path already exists".into());
    }
    println!(
        "Cloud + realtime sky viewer: {}x{}, target {} spp. WASD move; Q/E down/up; Shift fast; right drag look; left drag orbit; wheel zoom. Space pause; R reset; I/K sun elevation; J/L azimuth; [ / ] exposure; sidebar Save (Ctrl+S); Esc exit.",
        config.width, config.height, config.spp
    );
    println!(
        "Snapshots wait for a complete sample batch. GPU work is bounded and yields between submissions."
    );
    if options.diagnostic() {
        println!("Diagnostic window run: automatic exit, no reference export.");
    }
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        volume: Some(volume),
        initial_camera: camera.clone(),
        initial_settings: settings.clone(),
        camera,
        settings,
        config,
        exposure_ev,
        record,
        options,
        started: Instant::now(),
        view: None,
        error_window: None,
        paused: false,
        dragging: false,
        looking: false,
        control_pressed: false,
        controller: Controller::default(),
        sky_enabled: true,
        last_motion: Instant::now(),
        next_motion: Instant::now(),
        cursor: None,
        fatal: None,
        exiting: false,
        reset_pending: false,
        save_requested: false,
        save_job: None,
        preview_job: None,
        capture_written: false,
        message: None,
    };
    match catch_unwind(AssertUnwindSafe(|| event_loop.run_app(&mut app))) {
        Ok(result) => result?,
        Err(payload) => {
            let error = panic_message(payload);
            let log = write_error_log(&error, &app.record);
            return Err(format!("cloud viewer failed: {error}; log: {}", log.display()).into());
        }
    }
    if let Some(job) = app.save_job.take() {
        let _ = job.thread.join();
    }
    if let Some(job) = app.preview_job.take() {
        match job.thread.join() {
            Ok(Ok(path)) => {
                app.capture_written = true;
                println!(
                    "Diagnostic UI preview: {} (not a reference)",
                    path.display()
                );
            }
            Ok(Err(error)) => return Err(error.into()),
            Err(payload) => return Err(panic_message(payload).into()),
        }
    }
    if let Some(error) = app.fatal {
        return Err(error.into());
    }
    if app.options.capture_preview.is_some() && !app.capture_written {
        return Err(
            "diagnostic stopped before the UI preview readback completed; no PNG captured".into(),
        );
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Display {
    image: [u32; 4],
    viewport: [f32; 4],
    exposure: [f32; 4],
    camera_forward: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    ambient: [f32; 4],
}

/// Readback copies are encoded with the corresponding GPU work. Mapping starts
/// only after submission, and the event loop checks completion without waiting.
struct AsyncReadback {
    buffer: wgpu::Buffer,
    receiver: Receiver<std::result::Result<(), String>>,
    sender: Option<Sender<std::result::Result<(), String>>>,
}
impl AsyncReadback {
    fn texture(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<(Self, u32)> {
        let row_bytes = width
            .checked_mul(4)
            .and_then(|n| n.checked_add(255))
            .map(|n| n / 256 * 256)
            .ok_or("preview row size overflow")?;
        let size = u64::from(row_bytes) * u64::from(height);
        if size == 0 || size > device.limits().max_buffer_size {
            return Err("preview readback exceeds buffer limit".into());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diagnostic UI preview readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let (sender, receiver) = mpsc::channel();
        Ok((
            Self {
                buffer,
                receiver,
                sender: Some(sender),
            },
            row_bytes,
        ))
    }
    fn encode(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        sources: &[(&wgpu::Buffer, u64)],
    ) -> Result<Self> {
        let bytes = sources
            .iter()
            .try_fold(0u64, |n, (_, size)| n.checked_add(*size))
            .ok_or("readback size overflow")?;
        if bytes == 0 || bytes > device.limits().max_buffer_size {
            return Err("cloud readback exceeds device buffer limit".into());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud asynchronous readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut offset = 0;
        for &(source, size) in sources {
            encoder.copy_buffer_to_buffer(source, 0, &buffer, offset, size);
            offset += size;
        }
        let (sender, receiver) = mpsc::channel();
        Ok(Self {
            buffer,
            receiver,
            sender: Some(sender),
        })
    }
    fn arm(&mut self) {
        let sender = self.sender.take().expect("readback is mapped once");
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result.map_err(|error| error.to_string()));
            });
    }
    fn try_bytes(&self) -> Result<Option<Vec<u8>>> {
        match self.receiver.try_recv() {
            Ok(Ok(())) => {
                let bytes = self.buffer.get_mapped_range(..).to_vec();
                self.buffer.unmap();
                Ok(Some(bytes))
            }
            Ok(Err(error)) => Err(format!("cloud GPU readback failed: {error}").into()),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err("cloud readback completion channel disconnected".into())
            }
        }
    }
}
enum Pending {
    Preview {
        readback: AsyncReadback,
        started: Instant,
        width: u32,
        height: u32,
        row_bytes: u32,
        format: wgpu::TextureFormat,
        path: PathBuf,
    },
    Work {
        readback: AsyncReadback,
        started: Instant,
    },
    Display {
        receiver: Receiver<()>,
        started: Instant,
    },
    Barrier {
        receiver: Receiver<()>,
        started: Instant,
    },
    Snapshot {
        readback: AsyncReadback,
        started: Instant,
        film_bytes: usize,
        record: Record,
        exposure: f32,
        path: PathBuf,
    },
}
impl Pending {
    fn started(&self) -> Instant {
        match self {
            Self::Work { started, .. }
            | Self::Preview { started, .. }
            | Self::Display { started, .. }
            | Self::Barrier { started, .. }
            | Self::Snapshot { started, .. } => *started,
        }
    }
}
struct SaveJob {
    receiver: Receiver<std::result::Result<(PathBuf, u32), String>>,
    thread: JoinHandle<()>,
}
struct PreviewJob {
    thread: JoinHandle<std::result::Result<PathBuf, String>>,
}
enum Completion {
    Preview {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        row_bytes: u32,
        format: wgpu::TextureFormat,
        path: PathBuf,
    },
    Work,
    Barrier,
    Display,
    Snapshot {
        film: cloud_pt::film::Film,
        record: Record,
        exposure: f32,
        path: PathBuf,
    },
}

struct View {
    window: Arc<Window>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    occluded: bool,
    focused: bool,
    resize_pending: bool,
    renderer: ProgressiveRenderer,
    sky: SkyBackground,
    ui: ViewerUi,
    sky_dirty: bool,
    ui_repaint_at: Option<Instant>,
    capture_path: Option<PathBuf>,
    display_uniform: wgpu::Buffer,
    display_bind_group: wgpu::BindGroup,
    display_pipeline: wgpu::RenderPipeline,
    pending: Option<Pending>,
    errors: Receiver<String>,
    next_poll: Instant,
    next_work: Instant,
    next_frame: Instant,
    needs_redraw: bool,
    redraw_requested: bool,
    batch_active: bool,
    progress: Option<WorkProgress>,
    last_work_ms: f64,
    frames: u64,
    chunks: u64,
    adapter_name: String,
}

/// Headless rendering explicitly opts into GPU creation; CPU commands never call it.
pub async fn create_device(
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Device, wgpu::Queue, String)> {
    let instance = wgpu::Instance::default();
    let adapter = request_adapter(&instance, surface).await?;
    let name = adapter.get_info().name;
    let (device, queue) = request_device(&adapter).await?;
    Ok((device, queue, name))
}
async fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter> {
    Ok(instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
        })
        .await?)
}
async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let mut limits = adapter.limits();
    limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.min(1 << 30);
    Ok(adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("cloud reference device"),
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?)
}
impl View {
    async fn new(
        window: Arc<Window>,
        volume: &SparseGpuVolume,
        camera: &Camera,
        settings: &TransportSettings,
        config: &RenderConfig,
        capture_path: Option<PathBuf>,
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = request_adapter(&instance, Some(&surface)).await?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = request_device(&adapter).await?;
        let (error_sender, errors) = mpsc::channel();
        let uncaptured = error_sender.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            let _ = uncaptured.send(format!("GPU validation/error: {error}"));
        }));
        device.set_device_lost_callback(move |reason, message| {
            let _ = error_sender.send(format!("GPU device lost ({reason:?}): {message}"));
        });
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| capabilities.formats.first().copied())
            .ok_or("surface has no texture format")?;
        let present_mode = capabilities
            .present_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::PresentMode::Fifo)
            .or_else(|| capabilities.present_modes.first().copied())
            .ok_or("surface has no presentation mode")?;
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .ok_or("surface has no alpha mode")?;
        if capture_path.is_some() && !capabilities.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            return Err("this surface does not support optional --capture-preview COPY_SRC".into());
        }
        if capture_path.is_some()
            && !matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm
                    | wgpu::TextureFormat::Bgra8UnormSrgb
                    | wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Rgba8UnormSrgb
            )
        {
            return Err("optional --capture-preview requires an RGBA8/BGRA8 surface".into());
        }
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | if capture_path.is_some() {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        if size.width > 0 && size.height > 0 {
            surface.configure(&device, &surface_config);
        }
        let sky = SkyBackground::new(&device, &queue, settings)?;
        let renderer =
            ProgressiveRenderer::new_preview(&device, &queue, volume, camera, settings, config)?;
        let ui = ViewerUi::new(&window, &device, format);
        let display_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud display transform"),
            contents: bytemuck::bytes_of(&Display::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud film display"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });
        let display_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud linear sky composite bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud sky composite layout"),
                bind_group_layouts: &[Some(&display_layout)],
                immediate_size: 0,
            });
        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud display"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud display film"),
            layout: &display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: renderer.film_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: display_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(sky.view()),
                },
            ],
        });
        let now = Instant::now();
        let (sender, receiver) = mpsc::channel();
        queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        Ok(Self {
            window,
            _instance: instance,
            surface,
            device,
            queue,
            surface_config,
            size,
            occluded: false,
            focused: true,
            resize_pending: false,
            renderer,
            sky,
            ui,
            sky_dirty: true,
            ui_repaint_at: None,
            capture_path,
            display_uniform,
            display_bind_group,
            display_pipeline,
            pending: Some(Pending::Barrier {
                receiver,
                started: now,
            }),
            errors,
            next_poll: now,
            next_work: now,
            next_frame: now,
            needs_redraw: true,
            redraw_requested: false,
            batch_active: false,
            progress: None,
            last_work_ms: 0.0,
            frames: 0,
            chunks: 0,
            adapter_name,
        })
    }
    fn drawable(&self) -> bool {
        self.size.width > 0 && self.size.height > 0 && !self.occluded
    }
    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.size = size;
        self.resize_pending = true;
        self.needs_redraw = true;
        if size.width > 0 && size.height > 0 {
            self.occluded = false;
        }
    }
    fn check_completion(&mut self, now: Instant) -> Result<Option<Completion>> {
        if now >= self.next_poll {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| e.to_string())?;
            self.next_poll = now + POLL_INTERVAL;
        }
        if let Ok(error) = self.errors.try_recv() {
            return Err(error.into());
        }
        let Some(pending) = self.pending.as_ref() else {
            return Ok(None);
        };
        let ready = match pending {
            Pending::Work { readback, .. }
            | Pending::Snapshot { readback, .. }
            | Pending::Preview { readback, .. } => readback.try_bytes()?,
            Pending::Display { receiver, .. } | Pending::Barrier { receiver, .. } => {
                match receiver.try_recv() {
                    Ok(()) => Some(Vec::new()),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => {
                        return Err("GPU presentation completion channel disconnected".into());
                    }
                }
            }
        };
        let Some(bytes) = ready else {
            if now.duration_since(pending.started()) > RESPONSE_TIMEOUT {
                return Err("GPU work/readback did not respond within 5 seconds; stopped without reusing the device".into());
            }
            return Ok(None);
        };
        let pending = self.pending.take().unwrap();
        match pending {
            Pending::Preview {
                width,
                height,
                row_bytes,
                format,
                path,
                ..
            } => {
                self.frames += 1;
                Ok(Some(Completion::Preview {
                    bytes,
                    width,
                    height,
                    row_bytes,
                    format,
                    path,
                }))
            }
            Pending::Work { started, .. } => {
                let p = self.renderer.complete_work(&bytes)?;
                let elapsed = now.duration_since(started);
                self.last_work_ms = elapsed.as_secs_f64() * 1000.0;
                self.next_work = now + work_rest(elapsed);
                self.batch_active = !p.batch_finished;
                if progress_updates_film(&p)
                    && self.progress.as_ref().is_none_or(|old| {
                        old.tile_start != p.tile_start
                            || old.completed_paths != p.completed_paths
                            || p.batch_finished
                    })
                {
                    self.needs_redraw = true;
                }
                self.chunks += 1;
                self.progress = Some(p);
                Ok(Some(Completion::Work))
            }
            Pending::Display { .. } => {
                self.frames += 1;
                Ok(Some(Completion::Display))
            }
            Pending::Barrier { .. } => Ok(Some(Completion::Barrier)),
            Pending::Snapshot {
                film_bytes,
                record,
                exposure,
                path,
                ..
            } => {
                if bytes.len() != film_bytes * 2 {
                    return Err("cloud snapshot readback length mismatch".into());
                }
                let film = self
                    .renderer
                    .decode_film_readback(&bytes[..film_bytes], &bytes[film_bytes..])?;
                Ok(Some(Completion::Snapshot {
                    film,
                    record,
                    exposure,
                    path,
                }))
            }
        }
    }
    fn submit_work(&mut self) -> Result<()> {
        if self.pending.is_some() {
            return Err("cloud viewer attempted overlapping GPU work".into());
        }
        let count = self
            .renderer
            .sample_batch_capacity()
            .min(self.renderer.target_samples() - self.renderer.sample_count());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cloud bounded work"),
            });
        self.renderer
            .encode_work(&self.device, &self.queue, &mut encoder, count)?;
        let mut readback = AsyncReadback::encode(
            &self.device,
            &mut encoder,
            &[(self.renderer.progress_buffer(), WORK_PROGRESS_BYTES as u64)],
        )?;
        self.queue.submit([encoder.finish()]);
        readback.arm();
        self.pending = Some(Pending::Work {
            readback,
            started: Instant::now(),
        });
        self.batch_active = true;
        Ok(())
    }
    fn snapshot(&mut self, record: Record, exposure: f32) -> Result<()> {
        if self.pending.is_some() || self.batch_active {
            return Err("snapshot needs a completed sample batch".into());
        }
        let size = self.renderer.size();
        let film_bytes = u64::from(size[0]) * u64::from(size[1]) * 16;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cloud snapshot copy"),
            });
        let mut readback = AsyncReadback::encode(
            &self.device,
            &mut encoder,
            &[
                (self.renderer.film_buffer(), film_bytes),
                (self.renderer.film_variance_buffer(), film_bytes),
            ],
        )?;
        let path = PathBuf::from(format!("out/cloud-demo-{}", unix_nanos()));
        self.queue.submit([encoder.finish()]);
        readback.arm();
        self.pending = Some(Pending::Snapshot {
            readback,
            started: Instant::now(),
            film_bytes: film_bytes as usize,
            record,
            exposure,
            path,
        });
        Ok(())
    }
    fn redraw(
        &mut self,
        model: &mut Controls,
        progress: &Progress,
        ignore_film: bool,
    ) -> Result<viewer_ui::Actions> {
        self.redraw_requested = false;
        if !self.drawable() || self.pending.is_some() {
            return Ok(viewer_ui::Actions::default());
        }
        if self.resize_pending {
            self.surface_config.width = self.size.width;
            self.surface_config.height = self.size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.resize_pending = false;
        }
        self.next_frame = Instant::now() + FRAME_INTERVAL;
        let (frame, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => (f, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => (f, true),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize_pending = true;
                return Ok(viewer_ui::Actions::default());
            }
            wgpu::CurrentSurfaceTexture::Timeout => return Ok(viewer_ui::Actions::default()),
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.occluded = true;
                return Ok(viewer_ui::Actions::default());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("cloud surface validation failed".into());
            }
        };
        // Prepare after acquisition succeeds: input/texture deltas are consumed
        // exactly once, and UI rendering shares this tracked queue submission.
        let (actions, prepared) =
            self.ui
                .prepare(&self.window, model, progress, self.renderer.size());
        let target = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cloud sky + controls display"),
            });
        if self.sky_dirty {
            self.sky.encode(
                &self.device,
                &self.queue,
                &mut encoder,
                &model.camera,
                &model.transport,
                model.sky_enabled,
            )?;
            self.renderer.set_environment(
                &self.device,
                &self.queue,
                model
                    .sky_enabled
                    .then(|| (self.sky.view(), self.sky.sunlight_view())),
            )?;
            self.sky_dirty = false;
        }
        let (forward, right, up) = model.camera.basis()?;
        let pack = |v: glam::DVec3, w: f32| [v.x as f32, v.y as f32, v.z as f32, w];
        self.queue.write_buffer(
            &self.display_uniform,
            0,
            bytemuck::bytes_of(&Display {
                image: [
                    self.renderer.size()[0],
                    self.renderer.size()[1],
                    self.size.width,
                    self.size.height,
                ],
                viewport: prepared.viewport.map(|n| n as f32),
                exposure: [
                    model.exposure,
                    if self.surface_config.format.is_srgb() {
                        0.0
                    } else {
                        1.0
                    },
                    f32::from(model.sky_enabled),
                    f32::from(
                        ignore_film
                            || actions.physical_changed
                            || actions.target_changed
                            || actions.reset,
                    ),
                ],
                camera_forward: pack(
                    forward,
                    (model.camera.horizontal_fov_deg.to_radians() * 0.5).tan() as f32,
                ),
                camera_right: pack(right, 0.0),
                camera_up: pack(up, 0.0),
                ambient: pack(model.transport.sky_radiance, 0.0),
            }),
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cloud sky composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let [x, y, w, h] = prepared.viewport;
            if w > 0 && h > 0 {
                pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, w, h);
                pass.set_pipeline(&self.display_pipeline);
                pass.set_bind_group(0, &self.display_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        let commands = self
            .ui
            .encode(&self.device, &self.queue, &mut encoder, &target, &prepared);
        let capture = if self.chunks >= 20
            || self.renderer.sample_count() >= self.renderer.target_samples()
        {
            if let Some(path) = self.capture_path.take() {
                let (readback, row_bytes) = AsyncReadback::texture(
                    &self.device,
                    &mut encoder,
                    &frame.texture,
                    self.size.width,
                    self.size.height,
                )?;
                Some((readback, row_bytes, path))
            } else {
                None
            }
        } else {
            None
        };
        self.queue.submit(
            commands
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        let (sender, receiver) = mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.pending = if let Some((mut readback, row_bytes, path)) = capture {
            readback.arm();
            Some(Pending::Preview {
                readback,
                started: Instant::now(),
                width: self.size.width,
                height: self.size.height,
                row_bytes,
                format: self.surface_config.format,
                path,
            })
        } else {
            Some(Pending::Display {
                receiver,
                started: Instant::now(),
            })
        };
        frame.present();
        self.ui.submitted(&prepared);
        self.ui_repaint_at = Instant::now().checked_add(prepared.repaint_after);
        self.needs_redraw = false;
        if suboptimal {
            self.resize_pending = true;
        }
        Ok(actions)
    }
}
fn work_rest(elapsed: Duration) -> Duration {
    // Conservatively target at most 25% GPU duty using observed completion wall
    // time (including callback polling), and always provide a real idle gap.
    elapsed.saturating_mul(3).max(MIN_WORK_REST)
}
fn progress_updates_film(progress: &WorkProgress) -> bool {
    progress.batch_finished || progress.completed_paths > 0
}

struct App {
    volume: Option<SparseGpuVolume>,
    initial_camera: Camera,
    initial_settings: TransportSettings,
    camera: Camera,
    settings: TransportSettings,
    config: RenderConfig,
    exposure_ev: f32,
    record: Record,
    options: ViewOptions,
    started: Instant,
    view: Option<View>,
    error_window: Option<Arc<Window>>,
    paused: bool,
    dragging: bool,
    looking: bool,
    control_pressed: bool,
    controller: Controller,
    sky_enabled: bool,
    last_motion: Instant,
    next_motion: Instant,
    cursor: Option<PhysicalPosition<f64>>,
    fatal: Option<String>,
    exiting: bool,
    reset_pending: bool,
    save_requested: bool,
    save_job: Option<SaveJob>,
    preview_job: Option<PreviewJob>,
    capture_written: bool,
    message: Option<String>,
}
impl App {
    fn stop(&mut self, error: impl ToString) {
        let error = error.to_string();
        let path = write_error_log(&error, &self.record);
        eprintln!(
            "Cloud viewer stopped: {error}\nDetails: {}\nThe window stays open. Close it or press Esc.",
            path.display()
        );
        self.fatal = Some(format!("{error}; log: {}", path.display()));
        self.paused = true;
        self.save_requested = false;
        // Release a lost/invalid device; never reset or submit to it again.
        self.view = None;
        if let Some(window) = &self.error_window {
            window.set_title(&format!(
                "Cloud STOPPED | {} | log: {} | Esc to close",
                error.chars().take(90).collect::<String>(),
                path.display()
            ));
        }
    }
    fn guard(&mut self, action: impl FnOnce(&mut Self) -> Result<()>) {
        match catch_unwind(AssertUnwindSafe(|| action(self))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => self.stop(error),
            Err(payload) => self.stop(panic_message(payload)),
        }
    }
    fn title(&self) {
        let Some(view) = &self.view else {
            return;
        };
        let status = if self.reset_pending {
            "updating camera/light"
        } else if self.save_requested {
            "waiting for complete batch to save"
        } else if self.paused {
            "paused"
        } else if !view.focused || !view.drawable() {
            "idle (window hidden/unfocused)"
        } else if view.renderer.sample_count() >= view.renderer.target_samples() {
            "complete"
        } else {
            "sampling"
        };
        let progress = view
            .progress
            .as_ref()
            .map(|p| {
                format!(
                    " | current pass pixels completed {} / {}",
                    p.completed_paths, p.total_paths
                )
            })
            .unwrap_or_default();
        let message = self
            .message
            .as_ref()
            .map(|m| format!(" | {m}"))
            .unwrap_or_default();
        view.window.set_title(&format!(
            "Cloud {}x{} | {} / {} spp | {status}{progress} | EV {:+.1}{message}",
            self.config.width,
            self.config.height,
            view.renderer.sample_count(),
            view.renderer.target_samples(),
            self.exposure_ev
        ));
    }
    fn mark_redraw(&mut self) {
        if let Some(view) = &mut self.view {
            view.needs_redraw = true;
        }
    }
    fn reset_film(&mut self) -> Result<()> {
        self.reset_pending = true;
        self.save_requested = false;
        self.message = None;
        self.mark_redraw();
        self.title();
        Ok(())
    }
    fn update_camera(&mut self, camera: Camera) -> Result<()> {
        if camera.basis().is_err()
            || self
                .settings
                .ground
                .is_some_and(|g| camera.origin.y < g.height)
        {
            return Ok(());
        }
        if same_pose(&self.camera, &camera) {
            return Ok(());
        }
        self.camera = camera;
        self.reset_film()
    }
    fn orbit(&mut self, delta: PhysicalPosition<f64>) -> Result<()> {
        if let Some(camera) = Controller::orbit(&self.camera, delta.x, delta.y) {
            self.update_camera(camera)?;
        }
        Ok(())
    }
    fn zoom(&mut self, amount: f64) -> Result<()> {
        if let Some(camera) = Controller::zoom(&self.camera, amount) {
            self.update_camera(camera)?;
        }
        Ok(())
    }
    fn move_sun(&mut self, elevation_delta: f64, azimuth_delta: f64) -> Result<()> {
        let [elevation, azimuth] = sun_angles(self.settings.sun_direction);
        self.settings.sun_direction = sun_direction(
            (elevation + elevation_delta.to_degrees()).clamp(-89.0, 89.0),
            azimuth + azimuth_delta.to_degrees(),
        );
        self.reset_film()
    }
    fn draw(&mut self) -> Result<()> {
        let mut model = Controls {
            camera: self.camera.clone(),
            transport: self.settings.clone(),
            target_spp: self.config.spp,
            paused: self.paused,
            exposure: self.exposure_ev,
            speed: self.controller.speed,
            sky_enabled: self.sky_enabled,
            local_ground: self.initial_settings.ground.unwrap_or(GroundPlane {
                height: -1000.0,
                albedo: glam::DVec3::splat(0.2),
            }),
        };
        let Some(view) = &mut self.view else {
            return Ok(());
        };
        let p = view.progress.as_ref();
        let progress = Progress {
            samples: view.renderer.sample_count(),
            completed_paths: p.map_or(0, |p| p.completed_paths),
            total_paths: p.map_or(0, |p| p.total_paths),
            message: self.message.clone(),
        };
        let actions = view.redraw(&mut model, &progress, self.reset_pending)?;
        self.record.environment = view.sky.metadata();
        self.paused = model.paused;
        self.exposure_ev = model.exposure;
        self.controller.speed = model.speed;
        if actions.physical_changed || actions.target_changed {
            model.camera.basis()?;
            model.transport.validate()?;
            self.camera = model.camera;
            self.settings = model.transport;
            self.sky_enabled = model.sky_enabled;
            self.config.spp = model.target_spp;
            self.reset_film()?;
        }
        if actions.reset {
            self.restore_initial()?;
        }
        if actions.save {
            self.request_save();
        }
        self.title();
        Ok(())
    }
    fn restore_initial(&mut self) -> Result<()> {
        self.camera = self.initial_camera.clone();
        self.settings = self.initial_settings.clone();
        self.sky_enabled = true;
        self.controller.clear();
        self.reset_film()
    }
    fn request_save(&mut self) {
        if self.options.diagnostic() {
            self.message = Some("diagnostic run: reference export disabled".into());
        } else if self.save_requested
            || self.save_job.is_some()
            || self
                .view
                .as_ref()
                .is_some_and(|view| matches!(view.pending, Some(Pending::Snapshot { .. })))
        {
            self.message = Some("snapshot request is already in progress".into());
        } else {
            self.save_requested = true;
            self.message = Some("snapshot waits for a full sample batch".into());
        }
        self.title();
    }
    fn tick(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        if self.exiting {
            return Ok(());
        }
        let now = Instant::now();
        if self
            .options
            .exit_after
            .is_some_and(|d| now.duration_since(self.started) >= d)
        {
            if let Some(view) = &self.view {
                println!(
                    "Cloud diagnostic duration reached: {} completed displays / {} completed bounded work chunks / {} complete spp; no reference exported.",
                    view.frames,
                    view.chunks,
                    view.renderer.sample_count()
                );
            } else {
                println!("Cloud diagnostic duration reached; no reference exported.");
            }
            self.exiting = true;
            event_loop.exit();
            return Ok(());
        }
        if self.fatal.is_some() {
            return Ok(());
        }
        if self
            .preview_job
            .as_ref()
            .is_some_and(|job| job.thread.is_finished())
        {
            match self.preview_job.take().unwrap().thread.join() {
                Ok(Ok(path)) => {
                    self.capture_written = true;
                    println!(
                        "Diagnostic UI preview: {} (not a reference)",
                        path.display()
                    );
                }
                Ok(Err(error)) => return Err(error.into()),
                Err(payload) => return Err(panic_message(payload).into()),
            }
            if self.options.exit_after.is_none() && self.options.smoke_frames.is_none() {
                self.exiting = true;
                event_loop.exit();
                return Ok(());
            }
        }
        let camera_input = self
            .view
            .as_ref()
            .is_some_and(|v| v.focused && v.drawable() && !v.ui.keyboard_captured());
        if camera_input && self.controller.moving() && now >= self.next_motion {
            let dt = now.duration_since(self.last_motion).as_secs_f64();
            self.last_motion = now;
            self.next_motion = now + FRAME_INTERVAL;
            if let Some(camera) =
                self.controller
                    .advance(&self.camera, dt, self.settings.ground.map(|g| g.height))
            {
                self.update_camera(camera)?;
            }
        } else if !camera_input || !self.controller.moving() {
            self.last_motion = now;
            self.next_motion = now + FRAME_INTERVAL;
        }
        if let Some(job) = &self.save_job {
            match job.receiver.try_recv() {
                Ok(Ok((path, n))) => {
                    self.message = Some(format!("saved {n} spp: {}", path.display()));
                    println!("{}", self.message.as_ref().unwrap());
                    self.save_job.take().unwrap().thread.join().ok();
                }
                Ok(Err(error)) => {
                    let log = write_error_log(&format!("snapshot output: {error}"), &self.record);
                    eprintln!("Cloud snapshot failed: {error}; log: {}", log.display());
                    self.message = Some(format!("save failed; log: {}", log.display()));
                    self.save_job.take().unwrap().thread.join().ok();
                }
                Err(TryRecvError::Disconnected) => {
                    self.save_job.take().unwrap().thread.join().ok();
                    let log = write_error_log(
                        "snapshot output completion channel disconnected",
                        &self.record,
                    );
                    self.message = Some(format!("snapshot output stopped; log: {}", log.display()));
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        let completion = if let Some(view) = &mut self.view {
            view.check_completion(now)?
        } else {
            None
        };
        if let Some(Completion::Preview {
            bytes,
            width,
            height,
            row_bytes,
            format,
            path,
        }) = &completion
        {
            let bytes = bytes.clone();
            let width = *width;
            let height = *height;
            let row_bytes = *row_bytes;
            let format = *format;
            let path = path.clone();
            self.preview_job = Some(PreviewJob {
                thread: std::thread::spawn(move || {
                    write_preview_png(&path, &bytes, width, height, row_bytes, format)
                        .map(|()| path)
                        .map_err(|error| error.to_string())
                }),
            });
        }
        if let Some(Completion::Snapshot {
            film,
            record,
            exposure,
            path,
        }) = completion
        {
            let (sender, receiver) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                let samples = film.samples_per_pixel;
                let result = match catch_unwind(AssertUnwindSafe(|| {
                    output::save(&path, &film, &record, exposure)
                })) {
                    Ok(Ok(())) => Ok((path, samples)),
                    Ok(Err(error)) => Err(error.to_string()),
                    Err(payload) => Err(panic_message(payload)),
                };
                let _ = sender.send(result);
            });
            self.save_job = Some(SaveJob { receiver, thread });
            self.message = Some("writing snapshot".into());
        }
        let Some(view) = &mut self.view else {
            return Ok(());
        };
        if view.ui_repaint_at.is_some_and(|at| now >= at) {
            view.ui_repaint_at = None;
            view.needs_redraw = true;
        }
        if self.options.smoke_frames.is_some_and(|n| view.frames >= n) {
            println!(
                "Cloud diagnostic completed {} displays / {} bounded work chunks; no reference exported.",
                view.frames, view.chunks
            );
            self.exiting = true;
            event_loop.exit();
            return Ok(());
        }
        if view.pending.is_none() {
            if self.reset_pending {
                view.renderer
                    .reset(&view.device, &view.queue, &self.camera, &self.settings)?;
                view.renderer.set_target_samples(self.config.spp)?;
                let (sender, receiver) = mpsc::channel();
                view.queue.on_submitted_work_done(move || {
                    let _ = sender.send(());
                });
                view.pending = Some(Pending::Barrier {
                    receiver,
                    started: now,
                });
                view.batch_active = false;
                view.progress = None;
                view.sky_dirty = true;
                view.needs_redraw = true;
                view.next_work = now + MIN_WORK_REST;
                self.reset_pending = false;
                self.record.camera = self.camera.clone();
                self.record.transport = self.settings.clone();
            }
            if view.pending.is_some() {
                self.title();
                return Ok(());
            }
            if self.save_requested
                && self.save_job.is_none()
                && !view.batch_active
                && view.renderer.sample_count() > 0
            {
                let mut record = self.record.clone();
                record.camera = self.camera.clone();
                record.transport = self.settings.clone();
                record.render = self.config.clone();
                record.environment = view.sky.metadata();
                view.snapshot(record, self.exposure_ev)?;
                self.save_requested = false;
            } else if view.needs_redraw && view.drawable() && now >= view.next_frame {
                if !view.redraw_requested {
                    view.window.request_redraw();
                    view.redraw_requested = true;
                }
            } else if view.drawable()
                && view.focused
                && !view.sky_dirty
                && (!self.paused || self.save_requested)
                && view.renderer.sample_count() < view.renderer.target_samples()
                && now >= view.next_work
            {
                view.submit_work()?;
            }
        }
        self.title();
        Ok(())
    }
    fn wait_control_flow(&self, event_loop: &ActiveEventLoop) {
        if self.exiting {
            // Win32 applies the next wait after about_to_wait. Keep it awake
            // until exit is observed; tick no longer submits any GPU work.
            event_loop.set_control_flow(ControlFlow::Poll);
            return;
        }
        let now = Instant::now();
        let mut wake = self.options.exit_after.map(|d| self.started + d);
        let mut add = |deadline: Instant| {
            wake = Some(wake.map_or(deadline, |old| old.min(deadline)));
        };
        if self.save_job.is_some() {
            add(now + POLL_INTERVAL);
        }
        if self.preview_job.is_some() {
            add(now + POLL_INTERVAL);
        }
        if self.fatal.is_none() {
            if let Some(view) = &self.view {
                if view.drawable()
                    && view.focused
                    && self.controller.moving()
                    && !view.ui.keyboard_captured()
                {
                    add(self.next_motion);
                }
                if view.drawable() && !view.redraw_requested {
                    if let Some(at) = view.ui_repaint_at {
                        add(at.max(view.next_frame));
                    }
                }
                if view.pending.is_some() {
                    add(view.next_poll);
                } else {
                    if view.drawable() && view.needs_redraw && !view.redraw_requested {
                        add(view.next_frame);
                    }
                    if view.drawable()
                        && view.focused
                        && !view.sky_dirty
                        && !view.redraw_requested
                        && (!self.paused || self.save_requested)
                        && view.renderer.sample_count() < view.renderer.target_samples()
                    {
                        add(view.next_work);
                    }
                    if self.reset_pending
                        || self.save_requested
                            && !view.batch_active
                            && view.renderer.sample_count() > 0
                    {
                        add(now + POLL_INTERVAL);
                    }
                }
            }
        }
        event_loop.set_control_flow(wake.map_or(ControlFlow::Wait, |at| {
            ControlFlow::WaitUntil(at.max(now + Duration::from_millis(1)))
        }));
    }
    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
    ) -> Result<()> {
        let consumed = if !self.exiting && self.fatal.is_none() {
            if let Some(view) = &mut self.view {
                let response = view.ui.on_event(&view.window, &event);
                if response.repaint {
                    view.needs_redraw = true;
                }
                response.consumed
            } else {
                false
            }
        } else {
            false
        };
        let keyboard_captured = self.view.as_ref().is_some_and(|v| v.ui.keyboard_captured());
        let pointer_ok = !consumed
            && self.view.as_ref().is_some_and(|v| {
                !v.ui.pointer_captured()
                    && self
                        .cursor
                        .is_some_and(|p| v.ui.pointer_in_viewport(p, v.window.scale_factor()))
            });
        if keyboard_captured {
            self.controller.clear();
        }
        match event {
            WindowEvent::CloseRequested => {
                self.exiting = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
            {
                self.exiting = true;
                event_loop.exit();
            }
            _ if self.fatal.is_some() || self.exiting => {}
            WindowEvent::RedrawRequested => self.draw()?,
            WindowEvent::Resized(size) => {
                if let Some(view) = &mut self.view {
                    view.resize(size);
                }
            }
            WindowEvent::Occluded(occluded) => {
                if let Some(view) = &mut self.view {
                    view.occluded = occluded;
                    if !occluded {
                        view.needs_redraw = true;
                    }
                }
                if occluded {
                    self.controller.clear();
                }
            }
            WindowEvent::Focused(focused) => {
                if let Some(view) = &mut self.view {
                    view.focused = focused;
                    if focused {
                        view.needs_redraw = true;
                    }
                }
                if !focused {
                    self.dragging = false;
                    self.looking = false;
                    self.control_pressed = false;
                    self.cursor = None;
                    self.controller.clear();
                }
                self.title();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.control_pressed = modifiers.state().control_key();
                self.controller.set_key(
                    MoveKey::Fast,
                    modifiers.state().shift_key() && !keyboard_captured,
                );
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed && pointer_ok;
                if pressed {
                    if let Some(view) = &self.view {
                        view.ui.release_keyboard_focus();
                    }
                }
                if button == MouseButton::Left {
                    self.dragging = pressed;
                }
                if button == MouseButton::Right {
                    self.looking = pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let inside = self.view.as_ref().is_some_and(|v| {
                    !v.ui.pointer_captured()
                        && v.ui.pointer_in_viewport(position, v.window.scale_factor())
                });
                if !consumed && inside {
                    if let Some(old) = self.cursor {
                        let delta = PhysicalPosition::new(position.x - old.x, position.y - old.y);
                        if self.looking {
                            if let Some(camera) = Controller::look(&self.camera, delta.x, delta.y) {
                                self.update_camera(camera)?;
                            }
                        } else if self.dragging {
                            self.orbit(delta)?;
                        }
                    }
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => {
                self.dragging = false;
                self.looking = false;
                self.cursor = None;
            }
            WindowEvent::MouseWheel { delta, .. } if pointer_ok => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y) * 0.1,
                    MouseScrollDelta::PixelDelta(p) => p.y * 0.002,
                };
                self.zoom(amount)?;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::KeyS && self.control_pressed {
                        self.controller.set_key(MoveKey::Backward, false);
                        if event.state == ElementState::Pressed
                            && !event.repeat
                            && !consumed
                            && !keyboard_captured
                        {
                            self.request_save();
                            self.mark_redraw();
                        }
                        return Ok(());
                    }
                    if let Some(movement) = move_key(key) {
                        if event.state == ElementState::Released {
                            self.controller.set_key(movement, false);
                        } else if !consumed && !keyboard_captured {
                            if !self.controller.moving() {
                                self.last_motion = Instant::now();
                                self.next_motion = self.last_motion;
                            }
                            self.controller.set_key(movement, true);
                        }
                    }
                    if event.state == ElementState::Pressed && !consumed && !keyboard_captured {
                        match key {
                            KeyCode::Space if !event.repeat => {
                                self.paused = !self.paused;
                                self.mark_redraw();
                            }
                            KeyCode::KeyR if !event.repeat => self.restore_initial()?,
                            KeyCode::BracketLeft => {
                                self.exposure_ev = (self.exposure_ev - 0.25).max(-30.0);
                                self.mark_redraw();
                            }
                            KeyCode::BracketRight => {
                                self.exposure_ev = (self.exposure_ev + 0.25).min(30.0);
                                self.mark_redraw();
                            }
                            KeyCode::KeyI => self.move_sun(2.0_f64.to_radians(), 0.0)?,
                            KeyCode::KeyK => self.move_sun(-2.0_f64.to_radians(), 0.0)?,
                            KeyCode::KeyJ => self.move_sun(0.0, -2.0_f64.to_radians())?,
                            KeyCode::KeyL => self.move_sun(0.0, 2.0_f64.to_radians())?,
                            _ => {}
                        }
                        self.title();
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.view.is_some() || self.fatal.is_some() {
            return;
        }
        self.guard(|app| {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("Cloud viewer | initializing")
                        .with_inner_size(LogicalSize::new(
                            app.config.width.max(960),
                            app.config.height.max(540),
                        )),
                )?,
            );
            app.error_window = Some(window.clone());
            let volume = app.volume.take().ok_or("cloud scene already consumed")?;
            app.view = Some(pollster::block_on(View::new(
                window,
                &volume,
                &app.camera,
                &app.settings,
                &app.config,
                app.options.capture_preview.clone(),
            ))?);
            app.record.adapter = app.view.as_ref().map(|view| view.adapter_name.clone());
            app.title();
            Ok(())
        });
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self
            .error_window
            .as_ref()
            .is_none_or(|window| window.id() != id)
        {
            return;
        }
        self.guard(|app| app.handle_window_event(event_loop, event));
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.guard(|app| app.tick(event_loop));
        self.wait_control_flow(event_loop);
    }
}
fn move_key(key: KeyCode) -> Option<MoveKey> {
    match key {
        KeyCode::KeyW => Some(MoveKey::Forward),
        KeyCode::KeyS => Some(MoveKey::Backward),
        KeyCode::KeyA => Some(MoveKey::Left),
        KeyCode::KeyD => Some(MoveKey::Right),
        KeyCode::KeyQ => Some(MoveKey::Down),
        KeyCode::KeyE => Some(MoveKey::Up),
        _ => None,
    }
}
fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}
fn write_preview_png(
    path: &std::path::Path,
    bytes: &[u8],
    width: u32,
    height: u32,
    row_bytes: u32,
    format: wgpu::TextureFormat,
) -> Result<()> {
    if path.exists() {
        return Err("diagnostic preview path already exists".into());
    }
    let rgba = preview_rgba(bytes, width, height, row_bytes, format)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    image::codecs::png::PngEncoder::new(file).write_image(
        &rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(())
}
fn preview_rgba(
    bytes: &[u8],
    width: u32,
    height: u32,
    row_bytes: u32,
    format: wgpu::TextureFormat,
) -> Result<Vec<u8>> {
    let packed_row = width.checked_mul(4).ok_or("preview width overflow")?;
    if width == 0
        || height == 0
        || row_bytes < packed_row
        || u64::from(row_bytes) * u64::from(height) != bytes.len() as u64
    {
        return Err("diagnostic preview readback size mismatch".into());
    }
    let bgra = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    if !bgra
        && !matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        )
    {
        return Err("unsupported preview surface format".into());
    }
    let mut rgba = Vec::with_capacity(packed_row as usize * height as usize);
    for row in bytes.chunks_exact(row_bytes as usize) {
        for pixel in row[..packed_row as usize].chunks_exact(4) {
            if bgra {
                rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
            } else {
                rgba.extend_from_slice(pixel);
            }
        }
    }
    Ok(rgba)
}
fn write_error_log(error: &str, record: &Record) -> PathBuf {
    let path = PathBuf::from(format!("out/cloud-demo-view-error-{}.log", unix_nanos()));
    let contents = serde_json::to_string_pretty(&serde_json::json!({
        "kind":"cloud_viewer_error","error":error,"scene":record,
        "reference_exported_by_error":false,"device_reused_after_error":false,
    }))
    .unwrap_or_else(|_| error.to_owned());
    if let Err(log_error) = fs::create_dir_all("out").and_then(|()| fs::write(&path, contents)) {
        eprintln!("Could not write {}: {log_error}", path.display());
    }
    path
}
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "unexpected window or GPU panic".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_preview_unpads_rows_and_preserves_rgba_colors() {
        let mut bytes = vec![99; 512];
        bytes[..8].copy_from_slice(&[3, 2, 1, 255, 6, 5, 4, 255]);
        bytes[256..264].copy_from_slice(&[9, 8, 7, 255, 12, 11, 10, 255]);
        let rgba = preview_rgba(&bytes, 2, 2, 256, wgpu::TextureFormat::Bgra8UnormSrgb).unwrap();
        assert_eq!(
            rgba,
            vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]
        );
        assert_eq!(
            preview_rgba(&rgba, 2, 2, 8, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap(),
            rgba
        );
        assert!(preview_rgba(&bytes, 2, 2, 4, wgpu::TextureFormat::Bgra8UnormSrgb).is_err());
        assert!(
            preview_rgba(
                &bytes[..511],
                2,
                2,
                256,
                wgpu::TextureFormat::Bgra8UnormSrgb
            )
            .is_err()
        );
        assert!(preview_rgba(&bytes, 2, 2, 256, wgpu::TextureFormat::Rgba16Float).is_err());
        let path = std::env::temp_dir().join(format!("cloud-ui-preview-cpu-{}.png", unix_nanos()));
        write_preview_png(
            &path,
            &bytes,
            2,
            2,
            256,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        )
        .unwrap();
        let decoded = image::open(&path).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.into_raw(), rgba);
        assert!(
            write_preview_png(
                &path,
                &bytes,
                2,
                2,
                256,
                wgpu::TextureFormat::Bgra8UnormSrgb
            )
            .is_err()
        );
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn idle_gap_is_positive_and_scales_with_observed_work() {
        assert_eq!(work_rest(Duration::ZERO), Duration::from_millis(1));
        assert_eq!(
            work_rest(Duration::from_millis(1)),
            Duration::from_millis(3)
        );
        assert_eq!(
            work_rest(Duration::from_millis(10)),
            Duration::from_millis(30)
        );
        assert_eq!(
            work_rest(Duration::from_millis(100)),
            Duration::from_millis(300)
        );
    }
    #[test]
    fn automatic_exit_is_explicit_diagnostic_mode() {
        assert!(!ViewOptions::default().diagnostic());
        assert!(
            ViewOptions {
                exit_after: Some(Duration::from_secs(1)),
                smoke_frames: None,
                capture_preview: None,
            }
            .diagnostic()
        );
        assert!(
            ViewOptions {
                exit_after: None,
                smoke_frames: Some(2),
                capture_preview: None,
            }
            .diagnostic()
        );
    }
    #[test]
    fn completed_paths_update_the_display_before_the_whole_batch_finishes() {
        let mut p = WorkProgress {
            samples_per_pixel: 0,
            completed_paths: 0,
            total_paths: 12,
            tile_start: 0,
            total_pixels: 24,
            batch_samples: 4,
            batch_finished: false,
        };
        assert!(!progress_updates_film(&p));
        p.completed_paths = 11;
        assert!(progress_updates_film(&p));
        p.completed_paths = 12;
        assert!(progress_updates_film(&p));
        p.completed_paths = 0;
        p.batch_finished = true;
        assert!(progress_updates_film(&p));
    }
}
