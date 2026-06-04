use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use sky_core::asset::SpectralAssetManifest;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

#[derive(Parser, Debug)]
#[command(version, about = "Realtime atmosphere experiment demo")]
struct Cli {
    #[arg(long, default_value = "out/asset.json")]
    asset: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let manifest: SpectralAssetManifest = serde_json::from_reader(File::open(&cli.asset)?)?;
    println!(
        "loaded asset {}: {}x{}, {} bands",
        cli.asset.display(),
        manifest.dimensions[0],
        manifest.dimensions[1],
        manifest.band_centers_nm.len()
    );

    let event_loop = EventLoop::new()?;
    let mut app = DemoApp {
        asset_path: cli.asset,
        manifest,
        state: None,
        init_error: None,
    };
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.init_error {
        return Err(error.into());
    }
    Ok(())
}

struct DemoApp {
    asset_path: PathBuf,
    manifest: SpectralAssetManifest,
    state: Option<GpuState>,
    init_error: Option<String>,
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let title = format!(
            "sky realtime demo - {}x{} {} bands",
            self.manifest.dimensions[0],
            self.manifest.dimensions[1],
            self.manifest.band_centers_nm.len()
        );
        let window = match event_loop.create_window(WindowAttributes::default().with_title(title)) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.init_error = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(GpuState::new(window, &self.manifest)) {
            Ok(state) => self.state = Some(state),
            Err(error) => {
                self.init_error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if window_id != state.window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => match state.render() {
                RenderStatus::Presented => {}
                RenderStatus::Reconfigure => state.resize(state.size),
                RenderStatus::Skip => {}
                RenderStatus::Exit => event_loop.exit(),
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if self.init_error.is_none() {
            println!("closed realtime demo for {}", self.asset_path.display());
        }
    }
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
}

impl GpuState {
    async fn new(window: Arc<Window>, manifest: &SpectralAssetManifest) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| error.to_string())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|error| error.to_string())?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sky_realtime_demo_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(caps.present_modes[0]);
        let alpha_mode = caps.alpha_modes[0];
        let width = size.width.max(1);
        let height = size.height.max(1);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size: PhysicalSize::new(width, height),
            clear_color: clear_color_from_manifest(manifest),
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) -> RenderStatus {
        let (frame, reconfigure_after_present) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return RenderStatus::Reconfigure;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return RenderStatus::Skip;
            }
            wgpu::CurrentSurfaceTexture::Validation => return RenderStatus::Exit,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sky_realtime_demo_encoder"),
            });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky_realtime_demo_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
        }

        self.queue.submit([encoder.finish()]);
        frame.present();
        if reconfigure_after_present {
            RenderStatus::Reconfigure
        } else {
            RenderStatus::Presented
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderStatus {
    Presented,
    Reconfigure,
    Skip,
    Exit,
}

fn clear_color_from_manifest(manifest: &SpectralAssetManifest) -> wgpu::Color {
    let sun = (manifest.sun_elevation_deg / 90.0).clamp(-1.0, 1.0) as f64;
    let band_count = manifest.band_centers_nm.len().max(1) as f64;
    wgpu::Color {
        r: 0.03 + 0.08 * sun.max(0.0),
        g: 0.04 + 0.015 * band_count.min(32.0),
        b: 0.10 + 0.10 * (1.0 - sun.max(0.0)),
        a: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use sky_core::asset::{SpectralAssetFiles, SpectralAssetManifest};

    #[test]
    fn parses_manifest_json() {
        let manifest = SpectralAssetManifest::spectral_panorama(
            [4, 2],
            8,
            1,
            0.0,
            0.0,
            0.2,
            vec![500.0],
            SpectralAssetFiles {
                rgb_exr: "sky_rgb.exr".to_owned(),
                rgb_png: "sky_rgb.png".to_owned(),
                band_exrs: vec!["bands/sky_500nm.exr".to_owned()],
            },
        );
        let json = serde_json::to_string(&manifest).expect("json");
        let parsed: SpectralAssetManifest = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.kind, sky_core::asset::SPECTRAL_PANORAMA_KIND);
        assert_eq!(parsed.band_centers_nm, vec![500.0]);
    }
}
