use winit::dpi::PhysicalSize;

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::controls::RealtimeControls;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CompareMode {
    #[default]
    Realtime,
    Reference,
    AbsoluteDifference,
    SignedDifference,
}

impl CompareMode {
    pub const fn shader_id(self) -> f32 {
        match self {
            Self::Realtime => 0.0,
            Self::Reference => 1.0,
            Self::AbsoluteDifference => 2.0,
            Self::SignedDifference => 3.0,
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Realtime => Self::Reference,
            Self::Reference => Self::AbsoluteDifference,
            Self::AbsoluteDifference => Self::SignedDifference,
            Self::SignedDifference => Self::Realtime,
        }
    }
}

pub struct ExperimentInit<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
    pub asset: &'a RealtimeAsset,
    pub display: DisplayTransform,
}

pub struct UpdateContext<'a> {
    pub controls: &'a RealtimeControls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl SurfaceViewport {
    pub const fn size(self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.width, self.height)
    }
}

pub struct FrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
    pub viewport: SurfaceViewport,
}

pub trait RealtimeExperiment {
    fn name(&self) -> &'static str;

    /// Headless photometric validation bypasses exposure and display mapping.
    fn set_linear_output(&mut self, _enabled: bool) {}

    fn update(&mut self, _context: UpdateContext<'_>) {}

    fn reference_available(&self) -> bool {
        false
    }

    fn render(&mut self, context: FrameContext<'_>);
}
