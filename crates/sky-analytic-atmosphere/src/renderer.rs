use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub const SCENE_RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSun {
    pub sun_to_scene: Vec3,
}

impl Default for AnalyticSun {
    fn default() -> Self {
        Self {
            sun_to_scene: Vec3::new(-0.431_934, -0.863_868, -0.259_161),
        }
    }
}

impl AnalyticSun {
    #[must_use]
    pub fn to_sun(self) -> Vec3 {
        -self.sun_to_scene
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AnalyticView {
    pub relative_world_from_clip: [[f32; 4]; 4],
    pub world_position: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct AnalyticFrameParams {
    pub view: AnalyticView,
    pub sun: AnalyticSun,
}

pub struct AnalyticAtmosphereContext {
    params_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl AnalyticAtmosphereContext {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let params_buffer = uniform_buffer(
            device,
            "analytic.params.uniform",
            AnalyticParamsGpu::from_frame(&AnalyticFrameParams {
                view: AnalyticView::zeroed(),
                sun: AnalyticSun::default(),
            }),
        );
        let view_buffer = uniform_buffer(device, "analytic.view.uniform", AnalyticView::zeroed());
        let layout = bind_group_layout(device);
        let bind_group = bind_group(
            device,
            &layout,
            BindGroupInput {
                params: &params_buffer,
                view: &view_buffer,
            },
        );
        let pipeline = render_pipeline(device, &layout);
        Self {
            params_buffer,
            view_buffer,
            bind_group,
            pipeline,
        }
    }

    pub fn prepare(&self, queue: &wgpu::Queue, params: &AnalyticFrameParams) {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&AnalyticParamsGpu::from_frame(params)),
        );
        queue.write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&params.view));
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("analytic.render.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AnalyticParamsGpu {
    sun_dir: [f32; 4],
}

impl AnalyticParamsGpu {
    fn from_frame(params: &AnalyticFrameParams) -> Self {
        Self {
            sun_dir: params
                .sun
                .to_sun()
                .normalize_or_zero()
                .extend(0.0)
                .to_array(),
        }
    }
}

fn uniform_buffer<T: Pod>(device: &wgpu::Device, label: &'static str, value: T) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("analytic.render.bgl"),
        entries: &[uniform_entry(0), uniform_entry(1)],
    })
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct BindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    view: &'a wgpu::Buffer,
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: BindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("analytic.render.bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.view.as_entire_binding(),
            },
        ],
    })
}

fn render_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("analytic.render.pipeline"),
        source: wgpu::ShaderSource::Wgsl(crate::ANALYTIC_SKY_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("analytic.render.pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("analytic.render.pipeline"),
        layout: Some(&pipeline_layout),
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
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

const _: () = assert!(core::mem::size_of::<AnalyticView>() == 80);
const _: () = assert!(core::mem::size_of::<AnalyticParamsGpu>() == 16);
