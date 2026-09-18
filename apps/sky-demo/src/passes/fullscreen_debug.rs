use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::experiment::{ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext};
use crate::view::ViewState;

const SHADER: &str = include_str!("../shaders/fullscreen_debug.wgsl");

pub struct FullscreenDebugExperiment {
    pass: FullscreenDebugPass,
    uniform: DebugUniform,
    display: DisplayTransform,
    surface_size: PhysicalSize<u32>,
}

impl FullscreenDebugExperiment {
    pub fn new(context: ExperimentInit<'_>) -> Self {
        let surface_size = PhysicalSize::new(1, 1);
        let uniform = DebugUniform::from_asset(
            context.asset,
            surface_size,
            context.display,
            ViewState::default(),
        );
        Self {
            pass: FullscreenDebugPass::new(context.device, context.surface_format, uniform),
            uniform,
            display: context.display,
            surface_size,
        }
    }
}

impl RealtimeExperiment for FullscreenDebugExperiment {
    fn name(&self) -> &'static str {
        "fullscreen-debug-panorama"
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.surface_size = PhysicalSize::new(size.width.max(1), size.height.max(1));
    }

    fn update(&mut self, context: UpdateContext<'_>) {
        self.uniform =
            DebugUniform::from_asset(context.asset, self.surface_size, self.display, context.view);
        self.pass.write_uniform(context.queue, self.uniform);
    }

    fn render(&mut self, context: FrameContext<'_>) {
        self.pass.render(context);
    }
}

struct FullscreenDebugPass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl FullscreenDebugPass {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        uniform: DebugUniform,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen_debug_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen_debug_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fullscreen_debug_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fullscreen_debug_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fullscreen_debug_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen_debug_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            bind_group,
            uniform_buffer,
        }
    }

    fn write_uniform(&self, queue: &wgpu::Queue, uniform: DebugUniform) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn render(&self, context: FrameContext<'_>) {
        let mut pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fullscreen_debug_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: context.target,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::SHADER;

    #[test]
    fn shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("parse fullscreen debug wgsl");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("validate fullscreen debug wgsl");
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DebugUniform {
    viewport_spp_band_count: [f32; 4],
    sun_observer_exposure_output: [f32; 4],
    asset_dimensions_padding: [f32; 4],
    view_yaw_pitch_fov_aspect: [f32; 4],
}

impl DebugUniform {
    fn from_asset(
        asset: &RealtimeAsset,
        viewport_size: PhysicalSize<u32>,
        display: DisplayTransform,
        view: ViewState,
    ) -> Self {
        let manifest = asset.manifest();
        let width = viewport_size.width.max(1) as f32;
        let height = viewport_size.height.max(1) as f32;
        Self {
            viewport_spp_band_count: [
                width,
                height,
                manifest.spp as f32,
                manifest.band_centers_nm.len() as f32,
            ],
            sun_observer_exposure_output: [
                manifest.sun_elevation_deg,
                manifest.sun_azimuth_deg,
                manifest.observer_altitude_km,
                display.exposure,
            ],
            asset_dimensions_padding: [
                manifest.dimensions[0] as f32,
                manifest.dimensions[1] as f32,
                0.0,
                display.output_space.shader_id(),
            ],
            view_yaw_pitch_fov_aspect: [
                view.yaw_deg,
                view.pitch_deg,
                view.fov_y_deg,
                width / height,
            ],
        }
    }
}
