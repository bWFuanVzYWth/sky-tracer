use std::time::Duration;

use egui_wgpu::ScreenDescriptor;
use winit::window::Window;

use crate::workbench::configure_context;

pub struct UiRenderer {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
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
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );
        Self {
            context,
            state,
            renderer,
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
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        prepared: &PreparedUi,
    ) {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky_workbench_egui_pass"),
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
        self.renderer.render(
            &mut pass.forget_lifetime(),
            &prepared.paint_jobs,
            &prepared.screen,
        );
    }

    pub fn free_textures(&mut self, prepared: &PreparedUi) {
        for id in &prepared.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
