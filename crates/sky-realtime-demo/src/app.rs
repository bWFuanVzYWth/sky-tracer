use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{WindowAttributes, WindowId};

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext,
};
use crate::gpu::{GpuContext, SurfaceFrameStatus};
use crate::passes::hillaire_atmosphere::HillaireAtmosphereExperiment;
use crate::passes::precomputed_atmosphere::PrecomputedAtmosphereExperiment;
use crate::passes::unreal_atmosphere::UnrealAtmosphereExperiment;
use crate::view::ViewController;

pub struct RunConfig {
    pub asset_path: PathBuf,
    pub experiment: ExperimentKind,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ExperimentKind {
    Hillaire,
    Precomputed,
    Unreal,
}

pub fn run(config: RunConfig) -> Result<(), Box<dyn Error>> {
    let asset = RealtimeAsset::load(&config.asset_path)?;
    println!("loaded asset: {}", asset.summary_line());
    for missing_file in asset.missing_referenced_files() {
        eprintln!(
            "warning: referenced asset file does not exist: {}",
            missing_file.display()
        );
    }

    let event_loop = EventLoop::new()?;
    let mut app = DemoApp {
        asset,
        gpu: None,
        experiment: None,
        experiment_kind: config.experiment,
        view: ViewController::default(),
        compare_mode: CompareMode::default(),
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
    view: ViewController,
    compare_mode: CompareMode,
    init_error: Option<String>,
}

impl DemoApp {
    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let Some(experiment) = self.experiment.as_mut() else {
            return;
        };

        let frame = match gpu.acquire_frame() {
            Ok(frame) => frame,
            Err(SurfaceFrameStatus::Reconfigure) => {
                gpu.resize(gpu.size());
                experiment.resize(gpu.size());
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

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sky_realtime_demo_frame_encoder"),
            });

        {
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            experiment.update(UpdateContext {
                asset: &self.asset,
                view: self.view.state(),
                compare_mode: self.compare_mode,
            });
            experiment.render(FrameContext {
                device: gpu.device(),
                queue: gpu.queue(),
                encoder: &mut encoder,
                target: &view,
                surface_size: gpu.size(),
            });
        }

        gpu.queue().submit([encoder.finish()]);
        texture.present();

        if reconfigure_after_present {
            gpu.resize(gpu.size());
            experiment.resize(gpu.size());
        }
    }

    fn window_id(&self) -> Option<WindowId> {
        self.gpu.as_ref().map(|gpu| gpu.window().id())
    }

    fn handle_compare_key(&mut self, event: &KeyEvent) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }
        let mode = match event.physical_key {
            PhysicalKey::Code(KeyCode::Digit1) => Some(CompareMode::Realtime),
            PhysicalKey::Code(KeyCode::Digit2) => Some(CompareMode::Reference),
            PhysicalKey::Code(KeyCode::Digit3) => Some(CompareMode::AbsoluteDifference),
            PhysicalKey::Code(KeyCode::Digit4) => Some(CompareMode::SignedDifference),
            PhysicalKey::Code(KeyCode::KeyD) => Some(self.compare_mode.next()),
            _ => None,
        };
        let Some(mode) = mode else {
            return false;
        };
        self.compare_mode = mode;
        println!("comparison mode: {}", self.compare_mode.label());
        true
    }
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let window = match event_loop
            .create_window(WindowAttributes::default().with_title(self.asset.title()))
        {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.init_error = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };

        let gpu = match pollster::block_on(GpuContext::new(window)) {
            Ok(gpu) => gpu,
            Err(error) => {
                self.init_error = Some(error);
                event_loop.exit();
                return;
            }
        };
        let display = DisplayTransform::default();
        println!("display transform: {}", display.output_space.label());
        println!("view controls: left drag = yaw/pitch, mouse wheel = fov, R = reset");
        println!(
            "comparison controls: 1 realtime, 2 reference, 3 abs diff, 4 signed diff, D cycle"
        );

        let init = ExperimentInit {
            device: gpu.device(),
            queue: gpu.queue(),
            surface_format: gpu.surface_format(),
            asset: &self.asset,
            display,
        };
        let mut experiment: Box<dyn RealtimeExperiment> = match self.experiment_kind {
            ExperimentKind::Hillaire => match HillaireAtmosphereExperiment::new(init) {
                Ok(experiment) => Box::new(experiment),
                Err(error) => {
                    self.init_error = Some(error);
                    event_loop.exit();
                    return;
                }
            },
            ExperimentKind::Precomputed => match PrecomputedAtmosphereExperiment::new(init) {
                Ok(experiment) => Box::new(experiment),
                Err(error) => {
                    self.init_error = Some(error);
                    event_loop.exit();
                    return;
                }
            },
            ExperimentKind::Unreal => match UnrealAtmosphereExperiment::new(init) {
                Ok(experiment) => Box::new(experiment),
                Err(error) => {
                    self.init_error = Some(error);
                    event_loop.exit();
                    return;
                }
            },
        };
        experiment.resize(gpu.size());

        println!("selected realtime experiment: {}", experiment.name());
        self.experiment = Some(experiment);
        self.gpu = Some(gpu);
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

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                    if let Some(experiment) = self.experiment.as_mut() {
                        experiment.resize(gpu.size());
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.view.cursor_moved(position);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.view.mouse_input(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.view.mouse_wheel(delta);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_compare_key(&event);
                self.view.keyboard_input(&event);
            }
            WindowEvent::RedrawRequested => self.render_frame(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            gpu.window().request_redraw();
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
