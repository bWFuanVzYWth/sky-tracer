use std::time::Duration;

use egui_wgpu::ScreenDescriptor;
use winit::window::Window;

use crate::workbench::configure_context;

const HDR_UI_COMPOSITE_SHADER: &str = include_str!("shaders/hdr_ui_composite.wgsl");

pub struct UiRenderer {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    hdr_overlay: Option<HdrUiOverlay>,
}

pub struct PreparedUi {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen: ScreenDescriptor,
}

impl PreparedUi {
    pub fn pixels_per_point(&self) -> f32 {
        self.screen.pixels_per_point
    }
}

impl UiRenderer {
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        request_repaint: impl Fn(Duration) + Send + Sync + 'static,
    ) -> Self {
        let context = egui::Context::default();
        configure_context(&context);
        context.set_request_repaint_callback(move |info| request_repaint(info.delay));
        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let hdr_overlay = (surface_format == wgpu::TextureFormat::Rgba16Float)
            .then(|| HdrUiOverlay::new(device, surface_format));
        let ui_format = if hdr_overlay.is_some() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            surface_format
        };
        let renderer =
            egui_wgpu::Renderer::new(device, ui_format, egui_wgpu::RendererOptions::default());
        Self {
            context,
            state,
            renderer,
            hdr_overlay,
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn run<R>(
        &mut self,
        window: &Window,
        mut build: impl FnMut(&mut egui::Ui) -> R,
    ) -> (R, PreparedUi) {
        let input = self.state.take_egui_input(window);
        let mut result = None;
        let output = self.context.run_ui(input, |ui| {
            result = Some(build(ui));
        });
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = output;
        self.state.handle_platform_output(window, platform_output);
        let paint_jobs = self.context.tessellate(shapes, pixels_per_point);
        let size = window.inner_size();
        (
            result.expect("egui frame closure must run"),
            PreparedUi {
                paint_jobs,
                textures_delta,
                screen: ScreenDescriptor {
                    size_in_pixels: [size.width.max(1), size.height.max(1)],
                    pixels_per_point,
                },
            },
        )
    }

    pub fn prepare_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedUi,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(overlay) = &mut self.hdr_overlay {
            overlay.ensure_size(device, prepared.screen.size_in_pixels);
        }
        for (id, delta) in &prepared.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &prepared.paint_jobs,
            &prepared.screen,
        )
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        prepared: &PreparedUi,
    ) {
        let ui_target = self.hdr_overlay.as_ref().map_or(target, HdrUiOverlay::view);
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky_workbench_egui_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ui_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if self.hdr_overlay.is_some() {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer.render(
            &mut pass.forget_lifetime(),
            &prepared.paint_jobs,
            &prepared.screen,
        );
        if let Some(overlay) = &self.hdr_overlay {
            overlay.composite(encoder, target);
        }
    }

    pub fn free_textures(&mut self, prepared: &PreparedUi) {
        for id in &prepared.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

struct HdrUiOverlay {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    bind_group: Option<wgpu::BindGroup>,
    size: [u32; 2],
}

impl HdrUiOverlay {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_workbench_hdr_ui_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_workbench_hdr_ui_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(HDR_UI_COMPOSITE_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_workbench_hdr_ui_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_workbench_hdr_ui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky_workbench_hdr_ui_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            texture: None,
            view: None,
            bind_group: None,
            size: [0, 0],
        }
    }

    fn ensure_size(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if self.size == size {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky_workbench_hdr_ui_texture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_workbench_hdr_ui_bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.texture = Some(texture);
        self.view = Some(view);
        self.bind_group = Some(bind_group);
        self.size = size;
    }

    fn view(&self) -> &wgpu::TextureView {
        self.view
            .as_ref()
            .expect("HDR UI texture prepared before rendering")
    }

    fn composite(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky_workbench_hdr_ui_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(
            0,
            self.bind_group
                .as_ref()
                .expect("HDR UI bind group prepared before rendering"),
            &[],
        );
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hdr_ui_composite_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(super::HDR_UI_COMPOSITE_SHADER)
            .expect("parse HDR UI composite WGSL");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .expect("validate HDR UI composite WGSL");
    }
}
