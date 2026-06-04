use winit::dpi::PhysicalSize;

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::view::ViewState;

pub struct ExperimentInit<'a> {
    pub device: &'a wgpu::Device,
    pub surface_format: wgpu::TextureFormat,
    pub asset: &'a RealtimeAsset,
    pub display: DisplayTransform,
}

pub struct UpdateContext<'a> {
    pub queue: &'a wgpu::Queue,
    pub asset: &'a RealtimeAsset,
    pub view: ViewState,
}

pub struct FrameContext<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
}

pub trait RealtimeExperiment {
    fn name(&self) -> &'static str;

    fn resize(&mut self, _size: PhysicalSize<u32>) {}

    fn update(&mut self, _context: UpdateContext<'_>) {}

    fn render(&mut self, context: FrameContext<'_>);
}
