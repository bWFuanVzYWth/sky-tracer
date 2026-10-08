//! Lightweight egui panel; CPU state and intent remain separate from GPU work.
use crate::controller::{sun_angles, sun_direction};
use cloud_pt::{
    config::Camera,
    transport::{GroundPlane, TransportSettings},
};
use egui_wgpu::ScreenDescriptor;
use glam::DVec3;
use std::time::Duration;
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

#[derive(Clone)]
pub struct Controls {
    pub camera: Camera,
    pub transport: TransportSettings,
    pub target_spp: u32,
    pub paused: bool,
    pub exposure: f32,
    pub speed: f64,
    pub sky_enabled: bool,
    pub local_ground: GroundPlane,
}
#[derive(Default)]
pub struct Actions {
    pub reset: bool,
    pub save: bool,
    pub physical_changed: bool,
    pub target_changed: bool,
    pub display_changed: bool,
}
#[derive(Default)]
pub struct Progress {
    pub samples: u32,
    pub completed_paths: u32,
    pub total_paths: u32,
    pub message: Option<String>,
}
pub struct Prepared {
    jobs: Vec<egui::ClippedPrimitive>,
    textures: egui::TexturesDelta,
    screen: ScreenDescriptor,
    pub viewport: [u32; 4],
    pub repaint_after: Duration,
}
pub struct ViewerUi {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    viewport: egui::Rect,
}
impl ViewerUi {
    pub fn new(window: &Window, device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let context = egui::Context::default();
        context.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer =
            egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        Self {
            context,
            state,
            renderer,
            viewport: egui::Rect::NOTHING,
        }
    }
    pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }
    pub fn keyboard_captured(&self) -> bool {
        self.context.egui_wants_keyboard_input()
    }
    pub fn pointer_captured(&self) -> bool {
        self.context.egui_is_using_pointer()
    }
    pub fn release_keyboard_focus(&self) {
        self.context.memory_mut(|memory| {
            if let Some(id) = memory.focused() {
                memory.surrender_focus(id);
            }
        });
    }
    pub fn pointer_in_viewport(
        &self,
        physical: winit::dpi::PhysicalPosition<f64>,
        scale: f64,
    ) -> bool {
        self.viewport.contains(egui::pos2(
            (physical.x / scale) as f32,
            (physical.y / scale) as f32,
        ))
    }
    pub fn prepare(
        &mut self,
        window: &Window,
        model: &mut Controls,
        progress: &Progress,
        film: [u32; 2],
    ) -> (Actions, Prepared) {
        let input = self.state.take_egui_input(window);
        let mut actions = Actions::default();
        let mut viewport = egui::Rect::NOTHING;
        let output = self.context.run_ui(input, |root| {
            viewport = build_panel(root, model, progress, &mut actions);
        });
        self.viewport = viewport;
        let repaint_after = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        self.state
            .handle_platform_output(window, output.platform_output);
        let jobs = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let size = window.inner_size();
        let available = physical_rect(viewport, output.pixels_per_point, size);
        let prepared = Prepared {
            jobs,
            textures: output.textures_delta,
            screen: ScreenDescriptor {
                size_in_pixels: [size.width.max(1), size.height.max(1)],
                pixels_per_point: output.pixels_per_point,
            },
            viewport: letterbox(available, film),
            repaint_after,
        };
        (actions, prepared)
    }
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        prepared: &Prepared,
    ) -> Vec<wgpu::CommandBuffer> {
        for (id, delta) in &prepared.textures.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        let commands =
            self.renderer
                .update_buffers(device, queue, encoder, &prepared.jobs, &prepared.screen);
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud controls"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer.render(
            &mut pass.forget_lifetime(),
            &prepared.jobs,
            &prepared.screen,
        );
        commands
    }
    pub fn submitted(&mut self, prepared: &Prepared) {
        for id in &prepared.textures.free {
            self.renderer.free_texture(id);
        }
    }
}
fn build_panel(
    root: &mut egui::Ui,
    model: &mut Controls,
    progress: &Progress,
    actions: &mut Actions,
) -> egui::Rect {
    egui::Panel::left("cloud_controls").default_size(280.0).resizable(false).show(root,|ui| {
                egui::ScrollArea::vertical().show(ui,|ui| {
                    ui.heading("Cloud + realtime sky");
                    ui.label(format!("{} / {} spp",progress.samples,model.target_spp));
                    ui.add(egui::ProgressBar::new((progress.samples as f32/model.target_spp.max(1) as f32).min(1.0)).show_percentage());
                    if progress.total_paths>0 {
                        ui.small(format!("Current pass: {} / {} pixels completed",progress.completed_paths,progress.total_paths));
                    }
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut model.paused,"Pause");
                        actions.reset|=ui.button("Reset").clicked();
                        actions.save|=ui.button("Save").clicked();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Target spp");
                        actions.target_changed|=ui.add(egui::DragValue::new(&mut model.target_spp).range(1..=16_777_216).speed(16)).changed();
                    }).response.on_hover_text("Changing the target starts a fresh fixed-spp accumulation.");
                    actions.display_changed|=ui.add(egui::Slider::new(&mut model.exposure,-30.0..=30.0).text("Exposure EV")).changed();
                    if let Some(message)=&progress.message {ui.small(message);}
                    ui.separator();
                    ui.collapsing("Lighting and medium",|ui| {
                        actions.physical_changed|=ui.checkbox(&mut model.sky_enabled,"Realtime sky illumination").changed();
                        let mut ground=model.transport.ground.is_some();
                        if ui.checkbox(&mut ground,"Local ground plane").on_hover_text("Optional flat cloud ground. It can cover the atmospheric horizon.").changed() {
                            model.transport.ground=ground.then(||model.local_ground.clone());
                            if let Some(g)=&model.transport.ground {
                                if model.camera.origin.y<g.height {
                                    let dy=g.height-model.camera.origin.y;
                                    model.camera.origin.y+=dy;model.camera.target.y+=dy;
                                }
                            }
                            actions.physical_changed=true;
                        }
                        let [mut elevation,mut azimuth]=sun_angles(model.transport.sun_direction);
                        let changed=ui.add(egui::Slider::new(&mut elevation,-90.0..=90.0).text("Sun elevation")).changed()
                            |ui.add(egui::Slider::new(&mut azimuth,-180.0..=180.0).text("Sun azimuth")).changed();
                        if changed {model.transport.sun_direction=sun_direction(elevation,azimuth);actions.physical_changed=true;}
                        actions.physical_changed|=ui.add(egui::Slider::new(&mut model.transport.extinction_scale,0.0..=16.0).text("Cloud density")).changed();
                        let mut albedo=model.transport.scattering_albedo.max_element();
                        if ui.add(egui::Slider::new(&mut albedo,0.0..=1.0).text("Cloud albedo")).changed() {
                            model.transport.scattering_albedo=DVec3::splat(albedo);actions.physical_changed=true;
                        }
                        actions.physical_changed|=ui.add(egui::Slider::new(&mut model.transport.phase_g,-0.99..=0.99).text("HG g")).changed();
                    });
                    ui.separator();
                    ui.collapsing("Camera",|ui| {
                        let old=model.camera.origin;
                        for (axis,label) in ["X","Y","Z"].into_iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!("{label} (m)"));
                                let low=if axis==1 {model.transport.ground.as_ref().map_or(-100000.0,|g|g.height)}else{-100000.0};
                                actions.physical_changed|=ui.add(egui::DragValue::new(&mut model.camera.origin[axis]).range(low..=100000.0).speed(1.0)).changed();
                            });
                        }
                        model.camera.target+=model.camera.origin-old;
                        actions.physical_changed|=ui.add(egui::Slider::new(&mut model.camera.horizontal_fov_deg,10.0..=150.0).text("Horizontal FOV")).changed();
                        ui.add(egui::Slider::new(&mut model.speed,0.1..=5000.0).logarithmic(true).text("Speed m/s"));
                        if let Ok((f,_,_))=model.camera.basis() {ui.small(format!("Forward {:.3}, {:.3}, {:.3}",f.x,f.y,f.z));}
                    });
                    ui.separator();
                    ui.small("WASD horizontal | Q/E down/up\nShift faster | RMB look\nLMB orbit | wheel zoom\nSpace pause | R reset | Ctrl+S save");
                    ui.small("Moving/light changes restart cloud samples. Background remains visible.");
                });
            });
    // Keep the canvas outside egui interaction areas so camera input reaches
    // winit only when it is not captured by a panel widget.
    root.available_rect_before_wrap()
}
fn physical_rect(rect: egui::Rect, scale: f32, size: PhysicalSize<u32>) -> [u32; 4] {
    let x = (rect.min.x * scale).floor().max(0.0) as u32;
    let y = (rect.min.y * scale).floor().max(0.0) as u32;
    let x = x.min(size.width);
    let y = y.min(size.height);
    let right = ((rect.max.x * scale).ceil().max(0.0) as u32).min(size.width);
    let bottom = ((rect.max.y * scale).ceil().max(0.0) as u32).min(size.height);
    [x, y, right.saturating_sub(x), bottom.saturating_sub(y)]
}
pub fn letterbox(available: [u32; 4], film: [u32; 2]) -> [u32; 4] {
    let [x, y, w, h] = available;
    if w == 0 || h == 0 || film[0] == 0 || film[1] == 0 {
        return [x, y, 0, 0];
    }
    let (vw, vh) = if u64::from(w) * u64::from(film[1]) <= u64::from(h) * u64::from(film[0]) {
        (
            w,
            ((u64::from(w) * u64::from(film[1]) / u64::from(film[0])) as u32).max(1),
        )
    } else {
        (
            ((u64::from(h) * u64::from(film[0]) / u64::from(film[1])) as u32).max(1),
            h,
        )
    };
    [x + (w - vw) / 2, y + (h - vh) / 2, vw, vh]
}
#[cfg(test)]
mod tests {
    use super::*;
    fn controls() -> Controls {
        Controls {
            camera: Camera::default(),
            transport: TransportSettings::default(),
            target_spp: 1024,
            paused: false,
            exposure: 0.0,
            speed: 100.0,
            sky_enabled: true,
            local_ground: GroundPlane {
                height: -1000.0,
                albedo: DVec3::splat(0.2),
            },
        }
    }
    fn cpu_panel(
        context: &egui::Context,
        model: &mut Controls,
        pointer: egui::Pos2,
        time: f64,
    ) -> (Actions, egui::Rect) {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 540.0),
            )),
            events: vec![egui::Event::PointerMoved(pointer)],
            time: Some(time),
            ..Default::default()
        };
        let mut actions = Actions::default();
        let mut viewport = egui::Rect::NOTHING;
        let _ = context.run_ui(input, |root| {
            viewport = build_panel(root, model, &Progress::default(), &mut actions);
        });
        (actions, viewport)
    }
    #[test]
    fn cpu_panel_leaves_camera_canvas_outside_egui_capture() {
        let context = egui::Context::default();
        let mut model = controls();
        let original = model.clone();
        let _ = cpu_panel(&context, &mut model, egui::pos2(700.0, 250.0), 0.0);
        let (actions, viewport) = cpu_panel(&context, &mut model, egui::pos2(700.0, 250.0), 0.02);
        assert!(viewport.min.x >= 250.0 && viewport.contains(egui::pos2(700.0, 250.0)));
        assert!(!context.egui_wants_pointer_input());
        assert!(!context.egui_wants_keyboard_input());
        assert!(!actions.physical_changed && !actions.target_changed && !actions.reset);
        assert_eq!(model.camera.origin, original.camera.origin);
        assert_eq!(model.camera.target, original.camera.target);
        assert_eq!(
            model.transport.sun_direction,
            original.transport.sun_direction
        );
        let _ = cpu_panel(&context, &mut model, egui::pos2(30.0, 20.0), 0.04);
        assert!(context.egui_wants_pointer_input());
    }
    #[test]
    fn viewport_preserves_film_aspect_and_side_panel_offset() {
        assert_eq!(
            letterbox([280, 0, 640, 540], [128, 72]),
            [280, 90, 640, 360]
        );
        assert_eq!(
            letterbox([300, 10, 200, 100], [128, 72]),
            [311, 10, 177, 100]
        );
        assert_eq!(letterbox([10, 10, 0, 100], [128, 72]), [10, 10, 0, 0]);
    }
}
