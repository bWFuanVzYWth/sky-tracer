//! Explicit GPU viewer with bounded work, asynchronous readbacks and idle time.
use crate::output::{self, Record};
use bytemuck::{Pod, Zeroable};
use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    gpu::{PRESENT_SHADER, ProgressiveRenderer, WORK_PROGRESS_BYTES, WorkProgress},
    transport::TransportSettings,
    volume::SparseGpuVolume,
};
use glam::DQuat;
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

const POLL_INTERVAL: Duration = Duration::from_millis(5);
const MIN_WORK_REST: Duration = Duration::from_millis(16);
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewOptions {
    pub exit_after: Option<Duration>,
    pub smoke_frames: Option<u64>,
}
impl ViewOptions {
    fn diagnostic(self) -> bool {
        self.exit_after.is_some() || self.smoke_frames.is_some()
    }
}

pub fn run(
    volume: SparseGpuVolume,
    camera: Camera,
    settings: TransportSettings,
    config: RenderConfig,
    exposure_ev: f32,
    record: Record,
    options: ViewOptions,
) -> Result<()> {
    camera.basis()?;
    settings.validate()?;
    config.validate()?;
    if !exposure_ev.is_finite() {
        return Err("display exposure must be finite".into());
    }
    println!(
        "Cloud viewer: {}x{}, target {} spp. Space pause; left drag orbit; wheel zoom; R reset; I/K sun elevation; J/L azimuth; [ / ] exposure; S snapshot; Esc exit.",
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
        cursor: None,
        fatal: None,
        exiting: false,
        reset_pending: false,
        save_requested: false,
        save_job: None,
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
    if let Some(error) = app.fatal {
        return Err(error.into());
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Display {
    image: [u32; 4],
    exposure: [f32; 4],
}

/// Readback copies are encoded with the corresponding GPU work. Mapping starts
/// only after submission, and the event loop checks completion without waiting.
struct AsyncReadback {
    buffer: wgpu::Buffer,
    receiver: Receiver<std::result::Result<(), String>>,
    sender: Option<Sender<std::result::Result<(), String>>>,
}
impl AsyncReadback {
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
enum Completion {
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
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
        let renderer = ProgressiveRenderer::new(&device, &queue, volume, camera, settings, config)?;
        let display_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud display transform"),
            contents: bytemuck::bytes_of(&Display::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud film display"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_SHADER.into()),
        });
        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cloud display"),
            layout: None,
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
            layout: &display_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: renderer.film_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: display_uniform.as_entire_binding(),
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
            Pending::Work { readback, .. } | Pending::Snapshot { readback, .. } => {
                readback.try_bytes()?
            }
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
    fn redraw(&mut self, exposure: f32) -> Result<()> {
        self.redraw_requested = false;
        if !self.drawable() || self.pending.is_some() {
            return Ok(());
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
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout => return Ok(()),
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.occluded = true;
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("cloud surface validation failed".into());
            }
        };
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
                exposure: [
                    exposure,
                    if self.surface_config.format.is_srgb() {
                        0.0
                    } else {
                        1.0
                    },
                    0.0,
                    0.0,
                ],
            }),
        );
        let target = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cloud display"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cloud display"),
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
            pass.set_pipeline(&self.display_pipeline);
            pass.set_bind_group(0, &self.display_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        self.pending = Some(Pending::Display {
            receiver,
            started: Instant::now(),
        });
        frame.present();
        self.needs_redraw = false;
        if suboptimal {
            self.resize_pending = true;
        }
        Ok(())
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
    cursor: Option<PhysicalPosition<f64>>,
    fatal: Option<String>,
    exiting: bool,
    reset_pending: bool,
    save_requested: bool,
    save_job: Option<SaveJob>,
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
                    " | tile {} / {} | paths {} / {}",
                    p.tile_start, p.total_pixels, p.completed_paths, p.total_paths
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
        self.camera = camera;
        self.reset_film()
    }
    fn orbit(&mut self, delta: PhysicalPosition<f64>) -> Result<()> {
        if !delta.x.is_finite() || !delta.y.is_finite() {
            return Ok(());
        }
        let offset = self.camera.origin - self.camera.target;
        let up = self.camera.up.normalize();
        let yawed = DQuat::from_axis_angle(up, -delta.x * 0.005) * offset;
        let right = (-yawed.normalize()).cross(up).normalize();
        let pitched = DQuat::from_axis_angle(right, -delta.y * 0.005) * yawed;
        let offset = if pitched.normalize().dot(up).abs() < 0.995 {
            pitched
        } else {
            yawed
        };
        let mut camera = self.camera.clone();
        camera.origin = camera.target + offset;
        self.update_camera(camera)
    }
    fn zoom(&mut self, amount: f64) -> Result<()> {
        if !amount.is_finite() {
            return Ok(());
        }
        let offset = self.camera.origin - self.camera.target;
        let distance = (offset.length() * (-amount).exp().clamp(0.2, 5.0)).clamp(0.1, 1e9);
        let mut camera = self.camera.clone();
        camera.origin = camera.target + offset.normalize() * distance;
        self.update_camera(camera)
    }
    fn move_sun(&mut self, elevation_delta: f64, azimuth_delta: f64) -> Result<()> {
        let dir = self.settings.sun_direction.normalize();
        let elevation =
            (dir.y.asin() + elevation_delta).clamp(-89.0_f64.to_radians(), 89.0_f64.to_radians());
        let azimuth = dir.z.atan2(dir.x) + azimuth_delta;
        self.settings.sun_direction = glam::DVec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        );
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
                view.snapshot(record, self.exposure_ev)?;
                self.save_requested = false;
            } else if view.needs_redraw && view.drawable() && now >= view.next_frame {
                if !view.redraw_requested {
                    view.window.request_redraw();
                    view.redraw_requested = true;
                }
            } else if view.drawable()
                && view.focused
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
        if self.fatal.is_none() {
            if let Some(view) = &self.view {
                if view.pending.is_some() {
                    add(view.next_poll);
                } else {
                    if view.drawable() && view.needs_redraw && !view.redraw_requested {
                        add(view.next_frame);
                    }
                    if view.drawable()
                        && view.focused
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
                event_loop.exit()
            }
            _ if self.fatal.is_some() || self.exiting => {}
            WindowEvent::RedrawRequested => {
                if let Some(view) = &mut self.view {
                    view.redraw(self.exposure_ev)?;
                }
                self.title();
            }
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
                    self.cursor = None;
                }
                self.title();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.dragging = state == ElementState::Pressed,
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some(old) = self.cursor {
                        self.orbit(PhysicalPosition::new(
                            position.x - old.x,
                            position.y - old.y,
                        ))?;
                    }
                }
                self.cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => {
                self.dragging = false;
                self.cursor = None;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y) * 0.1,
                    MouseScrollDelta::PixelDelta(p) => p.y * 0.002,
                };
                self.zoom(amount)?;
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match key {
                        KeyCode::Space if !event.repeat => self.paused = !self.paused,
                        KeyCode::KeyR if !event.repeat => {
                            self.camera = self.initial_camera.clone();
                            self.settings = self.initial_settings.clone();
                            self.reset_film()?;
                        }
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
                        KeyCode::KeyS if !event.repeat => self.request_save(),
                        _ => {}
                    }
                    self.title();
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
                            app.config.width.max(640),
                            app.config.height.max(360),
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
fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
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
    fn idle_gap_is_positive_and_scales_with_observed_work() {
        assert_eq!(work_rest(Duration::ZERO), Duration::from_millis(16));
        assert_eq!(
            work_rest(Duration::from_millis(1)),
            Duration::from_millis(16)
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
                smoke_frames: None
            }
            .diagnostic()
        );
        assert!(
            ViewOptions {
                exit_after: None,
                smoke_frames: Some(2)
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
