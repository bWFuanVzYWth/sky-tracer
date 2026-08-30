use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{WindowAttributes, WindowId};

use crate::assets::RealtimeAsset;
use crate::catalog::{CatalogScan, scan_asset_root};
use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, SurfaceViewport, UpdateContext,
};
use crate::gpu::{GpuContext, SurfaceFrameStatus};
use crate::passes::unreal_atmosphere_8wave::UnrealAtmosphere8WaveExperiment;
use crate::ui_renderer::UiRenderer;
use crate::workbench::{WorkbenchAction, WorkbenchState};

pub struct RunConfig {
    pub asset_path: PathBuf,
    pub experiment: ExperimentKind,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ExperimentKind {
    #[value(name = "unreal-8wave", alias = "unreal8-wave")]
    Unreal8Wave,
}

#[derive(Debug)]
enum UserEvent {
    RepaintAfter(Duration),
    CatalogScanned {
        generation: u64,
        result: Result<CatalogScan, String>,
    },
}

pub fn run(config: RunConfig) -> Result<(), Box<dyn Error>> {
    let asset = RealtimeAsset::load(&config.asset_path)?;
    println!("loaded asset: {}", asset.summary_line());
    warn_missing_files(&asset);

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let workbench = WorkbenchState::new(&asset);
    let mut app = DemoApp {
        asset,
        gpu: None,
        experiment: None,
        experiment_kind: config.experiment,
        ui: None,
        workbench,
        proxy,
        modifiers: ModifiersState::default(),
        cursor_position: None,
        last_viewport: SurfaceViewport {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        pending_actions: Vec::new(),
        next_repaint: None,
        init_error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.init_error {
        return Err(io::Error::other(error).into());
    }
    Ok(())
}

struct DemoApp {
    asset: RealtimeAsset,
    gpu: Option<GpuContext>,
    experiment: Option<Box<dyn RealtimeExperiment>>,
    experiment_kind: ExperimentKind,
    ui: Option<UiRenderer>,
    workbench: WorkbenchState,
    proxy: EventLoopProxy<UserEvent>,
    modifiers: ModifiersState,
    cursor_position: Option<PhysicalPosition<f64>>,
    last_viewport: SurfaceViewport,
    pending_actions: Vec<WorkbenchAction>,
    next_repaint: Option<Instant>,
    init_error: Option<String>,
}

impl DemoApp {
    fn create_experiment(
        kind: ExperimentKind,
        gpu: &GpuContext,
        asset: &RealtimeAsset,
    ) -> Result<Box<dyn RealtimeExperiment>, String> {
        let init = ExperimentInit {
            device: gpu.device(),
            queue: gpu.queue(),
            surface_format: gpu.surface_format(),
            asset,
            display: DisplayTransform::default(),
        };
        match kind {
            ExperimentKind::Unreal8Wave => UnrealAtmosphere8WaveExperiment::new(init)
                .map(|experiment| Box::new(experiment) as Box<dyn RealtimeExperiment>),
        }
    }

    fn window_id(&self) -> Option<WindowId> {
        self.gpu.as_ref().map(|gpu| gpu.window().id())
    }

    fn request_redraw(&self) {
        if let Some(gpu) = &self.gpu {
            gpu.window().request_redraw();
        }
    }

    fn create_ui_renderer(&self, gpu: &GpuContext) -> UiRenderer {
        let proxy = self.proxy.clone();
        UiRenderer::new(
            gpu.window(),
            gpu.device(),
            gpu.surface_format(),
            move |delay| {
                let _ = proxy.send_event(UserEvent::RepaintAfter(delay));
            },
        )
    }

    fn synchronize_display_mode(&mut self) -> bool {
        let desired_hdr = self.workbench.controls.hdr_enabled;
        let Some(gpu) = &self.gpu else {
            return false;
        };
        if desired_hdr == gpu.hdr_enabled() {
            return false;
        }
        if desired_hdr && !gpu.hdr_supported() {
            self.workbench.controls.hdr_enabled = false;
            self.workbench
                .set_error("HDR unavailable: this surface has no scRGB Rgba16Float format.");
            self.request_redraw();
            return false;
        }

        let previous_hdr = gpu.hdr_enabled();
        if let Err(error) = self
            .gpu
            .as_mut()
            .expect("GPU checked above")
            .set_hdr_enabled(desired_hdr)
        {
            self.workbench.controls.hdr_enabled = previous_hdr;
            self.workbench
                .set_error(format!("HDR switch failed: {error}"));
            self.request_redraw();
            return false;
        }

        let next_experiment = {
            let gpu = self.gpu.as_ref().expect("GPU checked above");
            Self::create_experiment(self.experiment_kind, gpu, &self.asset)
        };
        let next_experiment = match next_experiment {
            Ok(experiment) => experiment,
            Err(error) => {
                let _ = self
                    .gpu
                    .as_mut()
                    .expect("GPU checked above")
                    .set_hdr_enabled(previous_hdr);
                self.workbench.controls.hdr_enabled = previous_hdr;
                self.workbench
                    .set_error(format!("HDR pipeline rebuild failed: {error}"));
                self.request_redraw();
                return false;
            }
        };

        let next_ui = {
            let gpu = self.gpu.as_ref().expect("GPU checked above");
            self.create_ui_renderer(gpu)
        };
        self.experiment = Some(next_experiment);
        self.ui = Some(next_ui);
        if let Some(gpu) = &self.gpu {
            gpu.window().set_title(&format!(
                "{}{}",
                self.asset.title(),
                if desired_hdr { " [HDR scRGB]" } else { "" }
            ));
        }
        true
    }

    fn start_catalog_scan(&mut self) {
        let (generation, root) = self.workbench.begin_scan();
        let proxy = self.proxy.clone();
        std::thread::spawn(move || {
            let result = scan_asset_root(root);
            let _ = proxy.send_event(UserEvent::CatalogScanned { generation, result });
        });
    }

    fn load_asset(&mut self, path: PathBuf) {
        let Some(gpu) = &self.gpu else {
            self.workbench.set_error("GPU is not initialized.");
            return;
        };
        let candidate = prepare_asset_change(path.clone(), |asset| {
            Self::create_experiment(self.experiment_kind, gpu, asset)
        });
        let (next_asset, next_experiment) = match candidate {
            Ok(candidate) => candidate,
            Err(error) => {
                self.workbench.set_error(error);
                return;
            }
        };

        warn_missing_files(&next_asset);
        self.asset = next_asset;
        self.experiment = Some(next_experiment);
        self.workbench.set_current_asset(&self.asset);
        if !self.reference_available()
            && self.workbench.controls.compare_mode != CompareMode::Realtime
        {
            self.workbench.controls.compare_mode = CompareMode::Realtime;
        }
        if let Some(gpu) = &self.gpu {
            gpu.window().set_title(&format!(
                "{}{}",
                self.asset.title(),
                if gpu.hdr_enabled() {
                    " [HDR scRGB]"
                } else {
                    ""
                }
            ));
        }
        println!("loaded asset: {}", self.asset.summary_line());
    }

    fn reference_available(&self) -> bool {
        self.experiment
            .as_ref()
            .is_some_and(|experiment| experiment.reference_available())
    }

    fn process_actions(&mut self, actions: impl IntoIterator<Item = WorkbenchAction>) {
        for action in actions {
            match action {
                WorkbenchAction::OpenAsset => {
                    let mut dialog = rfd::FileDialog::new()
                        .add_filter("Sky asset manifest", &["json"])
                        .set_title("Open Sky Asset");
                    if self.workbench.asset_root().is_dir() {
                        dialog = dialog.set_directory(self.workbench.asset_root());
                    }
                    if let Some(path) = dialog.pick_file() {
                        self.load_asset(path);
                    }
                }
                WorkbenchAction::ChooseAssetRoot => {
                    let mut dialog = rfd::FileDialog::new().set_title("Choose Asset Root");
                    if self.workbench.asset_root().is_dir() {
                        dialog = dialog.set_directory(self.workbench.asset_root());
                    }
                    if let Some(root) = dialog.pick_folder() {
                        self.workbench.set_asset_root(root);
                        self.start_catalog_scan();
                    }
                }
                WorkbenchAction::RefreshCatalog => self.start_catalog_scan(),
                WorkbenchAction::LoadAsset(path) => self.load_asset(path),
            }
        }
    }

    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let reference_available = self.reference_available();
        let (hdr_supported, hdr_active) = self.gpu.as_ref().map_or((false, false), |gpu| {
            (gpu.hdr_supported(), gpu.hdr_enabled())
        });
        let (workbench_frame, prepared_ui) = {
            let Some(gpu) = &self.gpu else {
                return;
            };
            let Some(ui) = &mut self.ui else {
                return;
            };
            ui.run(gpu.window(), |root| {
                self.workbench.show(
                    root,
                    &self.asset,
                    reference_available,
                    hdr_supported,
                    hdr_active,
                )
            })
        };

        let mut actions = std::mem::take(&mut self.pending_actions);
        actions.extend(workbench_frame.actions);
        self.process_actions(actions);
        if self.synchronize_display_mode() {
            self.request_redraw();
            return;
        }

        let Some(gpu) = &self.gpu else {
            return;
        };
        let viewport = physical_viewport(
            workbench_frame.viewport_points,
            prepared_ui.pixels_per_point(),
            gpu.size(),
        );
        self.last_viewport = viewport;
        let controls = self.workbench.controls;

        let frame = match gpu.acquire_frame() {
            Ok(frame) => frame,
            Err(SurfaceFrameStatus::Reconfigure) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(gpu.size());
                }
                self.request_redraw();
                return;
            }
            Err(SurfaceFrameStatus::Skip) => return,
            Err(SurfaceFrameStatus::Exit) => {
                event_loop.exit();
                return;
            }
        };
        let reconfigure_after_present = frame.reconfigure_after_present;
        let texture = frame.texture;
        let target = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sky_realtime_demo_frame_encoder"),
            });

        let user_command_buffers = self
            .ui
            .as_mut()
            .expect("UI initialized with GPU")
            .prepare_gpu(gpu.device(), gpu.queue(), &mut encoder, &prepared_ui);

        let experiment = self
            .experiment
            .as_mut()
            .expect("experiment initialized with GPU");
        experiment.update(UpdateContext {
            controls: &controls,
        });
        experiment.render(FrameContext {
            device: gpu.device(),
            queue: gpu.queue(),
            encoder: &mut encoder,
            target: &target,
            viewport,
        });
        self.ui.as_mut().expect("UI initialized with GPU").render(
            &mut encoder,
            &target,
            &prepared_ui,
        );

        gpu.window().pre_present_notify();
        gpu.queue()
            .submit(user_command_buffers.into_iter().chain([encoder.finish()]));
        texture.present();
        self.ui
            .as_mut()
            .expect("UI initialized with GPU")
            .free_textures(&prepared_ui);

        if reconfigure_after_present {
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.resize(gpu.size());
            }
            self.request_redraw();
        }
    }

    fn handle_shortcut(&mut self, event: &KeyEvent) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }
        let PhysicalKey::Code(key) = event.physical_key else {
            return false;
        };
        if !self.cursor_inside_viewport() {
            return false;
        }

        if self.modifiers.control_key() && key == KeyCode::KeyO {
            self.pending_actions.push(WorkbenchAction::OpenAsset);
            return true;
        }
        match key {
            KeyCode::F5 => {
                self.pending_actions.push(WorkbenchAction::RefreshCatalog);
                return true;
            }
            KeyCode::PageUp => {
                if let Some(path) = self.workbench.step_asset(-1) {
                    self.pending_actions.push(WorkbenchAction::LoadAsset(path));
                }
                return true;
            }
            KeyCode::PageDown => {
                if let Some(path) = self.workbench.step_asset(1) {
                    self.pending_actions.push(WorkbenchAction::LoadAsset(path));
                }
                return true;
            }
            KeyCode::F6 if self.gpu.as_ref().is_some_and(GpuContext::hdr_supported) => {
                self.workbench.controls.hdr_enabled = !self.workbench.controls.hdr_enabled;
                return true;
            }
            _ => {}
        }

        let reference_available = self.reference_available();
        match key {
            KeyCode::Digit1 => self.workbench.controls.compare_mode = CompareMode::Realtime,
            KeyCode::Digit2 if reference_available => {
                self.workbench.controls.compare_mode = CompareMode::Reference;
            }
            KeyCode::Digit3 if reference_available => {
                self.workbench.controls.compare_mode = CompareMode::AbsoluteDifference;
            }
            KeyCode::Digit4 if reference_available => {
                self.workbench.controls.compare_mode = CompareMode::SignedDifference;
            }
            KeyCode::KeyD if reference_available => {
                self.workbench.controls.compare_mode = self.workbench.controls.compare_mode.next();
            }
            KeyCode::BracketLeft => self.workbench.controls.sun_elevation_deg -= 1.0,
            KeyCode::BracketRight => self.workbench.controls.sun_elevation_deg += 1.0,
            KeyCode::KeyR => self.workbench.controls.reset_view(),
            _ => return false,
        }
        self.workbench.controls = self.workbench.controls.normalized();
        true
    }

    fn cursor_inside_viewport(&self) -> bool {
        let Some(position) = self.cursor_position else {
            return false;
        };
        let viewport = self.last_viewport;
        position.x >= viewport.x as f64
            && position.y >= viewport.y as f64
            && position.x < (viewport.x + viewport.width) as f64
            && position.y < (viewport.y + viewport.height) as f64
    }
}

fn prepare_asset_change<T>(
    path: PathBuf,
    initialize: impl FnOnce(&RealtimeAsset) -> Result<T, String>,
) -> Result<(RealtimeAsset, T), String> {
    let asset = RealtimeAsset::load(&path)
        .map_err(|error| format!("Failed to load {}: {error}", path.display()))?;
    let prepared = initialize(&asset)
        .map_err(|error| format!("Failed to initialize {}: {error}", path.display()))?;
    Ok((asset, prepared))
}

impl ApplicationHandler<UserEvent> for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let initial_size =
            event_loop
                .primary_monitor()
                .map_or(LogicalSize::new(1440.0, 900.0), |monitor| {
                    let scale = monitor.scale_factor();
                    let size = monitor.size();
                    LogicalSize::new(
                        1440.0_f64.min(size.width as f64 / scale * 0.9),
                        900.0_f64.min(size.height as f64 / scale * 0.9),
                    )
                });
        let window = match event_loop.create_window(
            WindowAttributes::default()
                .with_title(self.asset.title())
                .with_inner_size(initial_size)
                .with_min_inner_size(LogicalSize::new(1100.0, 680.0)),
        ) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.init_error = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };

        let required_features = match self.experiment_kind {
            ExperimentKind::Unreal8Wave => sky_unreal_atmosphere_8wave::REQUIRED_FEATURES,
        };
        let required_limits = wgpu::Limits::default();
        let display = DisplayTransform::default();
        println!("display transform: {}", display.output_space.label());
        let gpu = match pollster::block_on(GpuContext::new(
            window.clone(),
            required_features,
            required_limits,
        )) {
            Ok(gpu) => gpu,
            Err(error) => {
                self.init_error = Some(error);
                event_loop.exit();
                return;
            }
        };
        let experiment = match Self::create_experiment(self.experiment_kind, &gpu, &self.asset) {
            Ok(experiment) => experiment,
            Err(error) => {
                self.init_error = Some(error);
                event_loop.exit();
                return;
            }
        };

        let ui = self.create_ui_renderer(&gpu);
        println!("selected realtime experiment: {}", experiment.name());
        println!("view controls: viewport drag = yaw/pitch, viewport wheel = fov, R = reset");
        println!("asset controls: Ctrl+O open, F5 rescan, Page Up/Down navigate");
        println!("display controls: F6 toggles HDR scRGB when supported");

        self.experiment = Some(experiment);
        self.ui = Some(ui);
        self.gpu = Some(gpu);
        self.start_catalog_scan();
        self.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id() {
            return;
        }

        let egui_response = if let (Some(gpu), Some(ui)) = (&self.gpu, &mut self.ui) {
            ui.on_window_event(gpu.window(), &event)
        } else {
            egui_winit::EventResponse::default()
        };
        if egui_response.repaint {
            self.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                }
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = Some(position);
                self.request_redraw();
            }
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                self.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !egui_response.consumed && self.handle_shortcut(&event) {
                    self.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.render_frame(event_loop),
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::RepaintAfter(delay) if delay.is_zero() => self.request_redraw(),
            UserEvent::RepaintAfter(delay) => {
                if let Some(deadline) = Instant::now().checked_add(delay) {
                    self.next_repaint = Some(
                        self.next_repaint
                            .map_or(deadline, |current| current.min(deadline)),
                    );
                }
            }
            UserEvent::CatalogScanned { generation, result } => {
                self.workbench.apply_scan(generation, result);
                self.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(deadline) = self.next_repaint {
            if deadline <= Instant::now() {
                self.next_repaint = None;
                self.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if self.init_error.is_none() {
            println!(
                "closed realtime demo for {}",
                self.asset.manifest_path().display()
            );
        }
    }
}

fn physical_viewport(
    rect: egui::Rect,
    pixels_per_point: f32,
    surface_size: PhysicalSize<u32>,
) -> SurfaceViewport {
    let surface_width = surface_size.width.max(1);
    let surface_height = surface_size.height.max(1);
    let x0 = (rect.min.x * pixels_per_point)
        .floor()
        .clamp(0.0, (surface_width - 1) as f32) as u32;
    let y0 = (rect.min.y * pixels_per_point)
        .floor()
        .clamp(0.0, (surface_height - 1) as f32) as u32;
    let x1 = (rect.max.x * pixels_per_point)
        .ceil()
        .clamp((x0 + 1) as f32, surface_width as f32) as u32;
    let y1 = (rect.max.y * pixels_per_point)
        .ceil()
        .clamp((y0 + 1) as f32, surface_height as f32) as u32;
    SurfaceViewport {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

fn warn_missing_files(asset: &RealtimeAsset) {
    for missing_file in asset.missing_referenced_files() {
        eprintln!(
            "warning: referenced asset file does not exist: {}",
            missing_file.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    use egui::{Rect, pos2};
    use sky_core::asset::{SpectralAssetFiles, SpectralAssetManifest};
    use winit::dpi::PhysicalSize;

    use super::{physical_viewport, prepare_asset_change};

    static NEXT_TEST_ROOT: AtomicU64 = AtomicU64::new(0);

    fn write_manifest(path: &Path) {
        let manifest = SpectralAssetManifest::spectral_panorama(
            [4, 2],
            8,
            1,
            12.0,
            34.0,
            0.2,
            vec![500.0],
            SpectralAssetFiles {
                rgb_exr: "sky.exr".to_owned(),
                rgb_png: "sky.png".to_owned(),
                band_exrs: vec!["band.exr".to_owned()],
            },
        );
        std::fs::write(
            path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    #[test]
    fn viewport_points_are_clamped_and_scaled_to_surface_pixels() {
        let viewport = physical_viewport(
            Rect::from_min_max(pos2(100.25, 40.0), pos2(700.0, 440.25)),
            1.5,
            PhysicalSize::new(1200, 800),
        );
        assert_eq!(viewport.x, 150);
        assert_eq!(viewport.y, 60);
        assert_eq!(viewport.width, 900);
        assert_eq!(viewport.height, 601);
    }

    #[test]
    fn failed_experiment_initialization_produces_no_load_candidate() {
        let root = std::env::temp_dir().join(format!(
            "sky-realtime-transaction-{}-{}",
            std::process::id(),
            NEXT_TEST_ROOT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create test root");
        let path = root.join("asset.json");
        write_manifest(&path);

        let result: Result<(_, ()), _> =
            prepare_asset_change(path, |_| Err("synthetic GPU failure".to_owned()));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("synthetic GPU failure"));

        std::fs::remove_dir_all(root).expect("remove test root");
    }
}
