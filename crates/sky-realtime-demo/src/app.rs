use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::experiment::{ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext};
use crate::gpu::{GpuContext, SurfaceFrameStatus};
use crate::passes::fullscreen_debug::FullscreenDebugExperiment;
use crate::view::ViewController;

pub struct RunConfig {
    pub asset_path: PathBuf,
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
        view: ViewController::default(),
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
    view: ViewController,
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

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sky_realtime_demo_frame_encoder"),
            });

        experiment.update(UpdateContext {
            queue: gpu.queue(),
            asset: &self.asset,
            view: self.view.state(),
        });
        experiment.render(FrameContext {
            encoder: &mut encoder,
            target: &frame.view,
        });

        gpu.queue().submit([encoder.finish()]);
        frame.texture.present();

        if frame.reconfigure_after_present {
            gpu.resize(gpu.size());
            experiment.resize(gpu.size());
        }
    }

    fn window_id(&self) -> Option<WindowId> {
        self.gpu.as_ref().map(|gpu| gpu.window().id())
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

        let mut experiment: Box<dyn RealtimeExperiment> =
            Box::new(FullscreenDebugExperiment::new(ExperimentInit {
                device: gpu.device(),
                surface_format: gpu.surface_format(),
                asset: &self.asset,
                display,
            }));
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
