use winit::dpi::PhysicalSize;

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::view::ViewState;

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

    pub const fn label(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Reference => "offline-reference",
            Self::AbsoluteDifference => "absolute-difference",
            Self::SignedDifference => "signed-difference",
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
    pub asset: &'a RealtimeAsset,
    pub view: ViewState,
    pub compare_mode: CompareMode,
    pub sun_elevation_deg: f32,
}

pub struct FrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
    pub surface_size: PhysicalSize<u32>,
}

pub trait RealtimeExperiment {
    fn name(&self) -> &'static str;

    fn resize(&mut self, _size: PhysicalSize<u32>) {}

    fn update(&mut self, _context: UpdateContext<'_>) {}

    fn render(&mut self, context: FrameContext<'_>);
}
