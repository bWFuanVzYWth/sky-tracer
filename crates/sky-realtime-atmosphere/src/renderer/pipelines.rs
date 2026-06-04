use std::borrow::Cow;

use ca_render::gpu::{DEPTH_FORMAT, SCENE_RADIANCE_FORMAT};

use super::bindings::RendererLayouts;

const TRANSMITTANCE_WGSL: &str = include_str!("../wgsl/transmittance.comp.wgsl");
const SKY_VIEW_WGSL: &str = include_str!("../wgsl/sky_view.comp.wgsl");
const AERIAL_PERSPECTIVE_WGSL: &str = include_str!("../wgsl/aerial_perspective.comp.wgsl");
const AP_APPLY_WGSL: &str = include_str!("../wgsl/aerial_perspective_apply.wgsl");
const SKY_WGSL: &str = include_str!("../wgsl/raster_sky_hillaire.wgsl");

pub(super) struct RendererPipelines {
    pub transmittance: wgpu::ComputePipeline,
    pub sky_view: wgpu::ComputePipeline,
    pub aerial_perspective: wgpu::ComputePipeline,
    pub ap_apply: wgpu::RenderPipeline,
    pub sky: wgpu::RenderPipeline,
}

impl RendererPipelines {
    pub fn new(device: &wgpu::Device, layouts: &RendererLayouts) -> Self {
        let transmittance = compute_pipeline(
            device,
            &layouts.transmittance,
            "hillaire.transmittance.pipeline",
            &format!("{}\n\n{}", crate::COMMON_WGSL, TRANSMITTANCE_WGSL),
        );
        let sky_view = compute_pipeline(
            device,
            &layouts.sky_view,
            "hillaire.sky_view.pipeline",
            &format!(
                "{}\n\n{}\n\n{}",
                crate::COMMON_WGSL,
                crate::INSCATTER_WGSL,
                SKY_VIEW_WGSL
            ),
        );
        let aerial_perspective = compute_pipeline(
            device,
            &layouts.aerial_perspective,
            "hillaire.aerial_perspective.pipeline",
            &format!(
                "{}\n\n{}\n\n{}",
                crate::COMMON_WGSL,
                crate::INSCATTER_WGSL,
                AERIAL_PERSPECTIVE_WGSL
            ),
        );
        let ap_apply = render_pipeline(
            device,
            &[Some(&layouts.view), Some(&layouts.ap_apply)],
            None,
            "hillaire.ap_apply.pipeline",
            AP_APPLY_WGSL,
        );
        let sky = render_pipeline(
            device,
            &[Some(&layouts.view), Some(&layouts.sky)],
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            "hillaire.sky.pipeline",
            &format!(
                "{}\n\n{}\n\n{}\n\n{}",
                ca_render::atmo::SKY_VIEW_WGSL,
                ca_render::atmo::SUN_WGSL,
                ca_render::atmo::VOXEL_ATMOSPHERE_LIGHTING_WGSL,
                SKY_WGSL,
            ),
        );
        Self {
            transmittance,
            sky_view,
            aerial_perspective,
            ap_apply,
            sky,
        }
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    source: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source.to_owned())),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn render_pipeline(
    device: &wgpu::Device,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    depth_stencil: Option<wgpu::DepthStencilState>,
    label: &'static str,
    source: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source.to_owned())),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_RADIANCE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
