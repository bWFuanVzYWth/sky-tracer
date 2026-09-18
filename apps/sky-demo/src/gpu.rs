use std::sync::Arc;

use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct GpuContext {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sdr_format: wgpu::TextureFormat,
    hdr_format: Option<wgpu::TextureFormat>,
    size: PhysicalSize<u32>,
}

impl GpuContext {
    pub async fn new(
        window: Arc<Window>,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self, String> {
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
        if !adapter.features().contains(required_features) {
            return Err(format!(
                "adapter does not support required realtime atmosphere features: {required_features:?}"
            ));
        }
        let adapter_limits = adapter.limits();
        if adapter_limits.max_texture_dimension_3d < required_limits.max_texture_dimension_3d {
            return Err(format!(
                "adapter max_texture_dimension_3d is {}, but this experiment requires {}",
                adapter_limits.max_texture_dimension_3d, required_limits.max_texture_dimension_3d
            ));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sky_realtime_demo_device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| error.to_string())?;

        let caps = surface.get_capabilities(&adapter);
        let sdr_format = preferred_sdr_format(&caps.formats)
            .ok_or_else(|| "surface reports no compatible formats".to_owned())?;
        let hdr_format = preferred_hdr_format(&caps.formats);
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
            format: sdr_format,
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
            sdr_format,
            hdr_format,
            size: PhysicalSize::new(width, height),
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn hdr_supported(&self) -> bool {
        self.hdr_format.is_some()
    }

    pub fn hdr_enabled(&self) -> bool {
        self.hdr_format == Some(self.config.format)
    }

    pub fn set_hdr_enabled(&mut self, enabled: bool) -> Result<bool, String> {
        let format = if enabled {
            self.hdr_format.ok_or_else(|| {
                "the active adapter/surface does not expose an scRGB Rgba16Float format".to_owned()
            })?
        } else {
            self.sdr_format
        };
        if format == self.config.format {
            return Ok(false);
        }
        self.config.format = format;
        self.surface.configure(&self.device, &self.config);
        Ok(true)
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn acquire_frame(&self) -> Result<SurfaceFrame, SurfaceFrameStatus> {
        let (texture, reconfigure_after_present) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return Err(SurfaceFrameStatus::Reconfigure);
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Err(SurfaceFrameStatus::Skip);
            }
            wgpu::CurrentSurfaceTexture::Validation => return Err(SurfaceFrameStatus::Exit),
        };
        Ok(SurfaceFrame {
            texture,
            reconfigure_after_present,
        })
    }
}

fn preferred_sdr_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .or_else(|| formats.first().copied())
}

fn preferred_hdr_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats
        .iter()
        .copied()
        .find(|format| *format == wgpu::TextureFormat::Rgba16Float)
}

pub struct SurfaceFrame {
    pub texture: wgpu::SurfaceTexture,
    pub reconfigure_after_present: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceFrameStatus {
    Reconfigure,
    Skip,
    Exit,
}

#[cfg(test)]
mod tests {
    use super::{preferred_hdr_format, preferred_sdr_format};

    #[test]
    fn surface_format_selection_prefers_srgb_sdr_and_float_scrgb_hdr() {
        let formats = [
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ];
        assert_eq!(
            preferred_sdr_format(&formats),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            preferred_hdr_format(&formats),
            Some(wgpu::TextureFormat::Rgba16Float)
        );
    }

    #[test]
    fn hdr_is_unavailable_without_float_surface_format() {
        let formats = [wgpu::TextureFormat::Bgra8UnormSrgb];
        assert_eq!(preferred_hdr_format(&formats), None);
    }
}
