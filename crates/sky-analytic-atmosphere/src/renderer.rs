use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub const SCENE_RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

const M_TO_KM: f32 = 1.0e-3;
const SUN_IRRADIANCE_REC2020_W_PER_M2: [f32; 3] = [205.0, 205.0, 205.0];
const SUN_SPECTRAL_IRRADIANCE: [f32; 4] = [1.679, 1.828, 1.986, 1.307];
const RAYLEIGH_SCATTERING_BASE_KM_INV: [f32; 4] = [6.605e-3, 1.067e-2, 1.842e-2, 3.156e-2];
const MIE_SCATTERING_BASE_KM_INV: [f32; 4] = [1.25e-3, 1.69e-3, 2.53e-3, 3.40e-3];
const MIE_CONCENTRATION_SCALE: f32 = 10.0;
const MIE_EXTINCTION_SCALE: f32 = 1.11;
// 630/560/490/430 nm ozone peak absorption for a 0..44 km triangular layer
// centered at 22 km, matched to the vertical ozone column in data/atmosphere_profile.csv.
const OZONE_ABSORPTION_PEAK_KM_INV: [f32; 4] = [1.373_34e-3, 1.390_23e-3, 5.150_03e-4, 1.684_31e-5];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticAtmosphere {
    pub bottom_radius_m: f32,
    pub top_radius_m: f32,
    pub world_y0_radius_m: f32,
    pub scene_units_to_m: f32,
}

impl Default for AnalyticAtmosphere {
    fn default() -> Self {
        Self {
            bottom_radius_m: 6_360_000.0,
            top_radius_m: 6_460_000.0,
            world_y0_radius_m: 6_360_500.0,
            scene_units_to_m: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSun {
    pub sun_to_scene: Vec3,
    pub angular_radius_rad: f32,
}

impl Default for AnalyticSun {
    fn default() -> Self {
        Self {
            sun_to_scene: Vec3::new(-0.431_934, -0.863_868, -0.259_161),
            angular_radius_rad: 0.004_71,
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
    pub atmosphere: AnalyticAtmosphere,
    pub sun: AnalyticSun,
}

pub struct AnalyticAtmosphereContext {
    params_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    sun_buffer: wgpu::Buffer,
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
                atmosphere: AnalyticAtmosphere::default(),
                sun: AnalyticSun::default(),
            }),
        );
        let view_buffer = uniform_buffer(device, "analytic.view.uniform", AnalyticView::zeroed());
        let sun_buffer = uniform_buffer(
            device,
            "analytic.sun.uniform",
            AnalyticSunGpu::from_sun(AnalyticSun::default()),
        );
        let layout = bind_group_layout(device);
        let bind_group = bind_group(
            device,
            &layout,
            BindGroupInput {
                params: &params_buffer,
                view: &view_buffer,
                sun: &sun_buffer,
            },
        );
        let pipeline = render_pipeline(device, &layout);
        Self {
            params_buffer,
            view_buffer,
            sun_buffer,
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
        queue.write_buffer(
            &self.sun_buffer,
            0,
            bytemuck::bytes_of(&AnalyticSunGpu::from_sun(params.sun)),
        );
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
    planet: [f32; 4],
    sun_dir: [f32; 4],
    sun_spectral_irradiance: [f32; 4],
    rayleigh_scattering_base: [f32; 4],
    mie_scattering_base: [f32; 4],
    mie_extinction_base: [f32; 4],
    ozone_absorption_base: [f32; 4],
}

impl AnalyticParamsGpu {
    fn from_frame(params: &AnalyticFrameParams) -> Self {
        let atmosphere = params.atmosphere;
        let bottom_radius_km = atmosphere.bottom_radius_m * M_TO_KM;
        let top_radius_km = atmosphere.top_radius_m * M_TO_KM;
        let eye_radius_km = view_radius_from_position(params.view.world_position, atmosphere);
        let eye_altitude_km = eye_radius_km - bottom_radius_km;
        let mut mie_scattering = [0.0; 4];
        let mut mie_extinction = [0.0; 4];
        for ((scattering_out, extinction_out), scattering) in mie_scattering
            .iter_mut()
            .zip(mie_extinction.iter_mut())
            .zip(MIE_SCATTERING_BASE_KM_INV.iter())
        {
            let scaled_scattering = scattering * MIE_CONCENTRATION_SCALE;
            *scattering_out = scaled_scattering;
            *extinction_out = scaled_scattering * MIE_EXTINCTION_SCALE;
        }

        Self {
            planet: [
                bottom_radius_km,
                top_radius_km,
                eye_radius_km,
                eye_altitude_km,
            ],
            sun_dir: params
                .sun
                .to_sun()
                .normalize_or_zero()
                .extend(0.0)
                .to_array(),
            sun_spectral_irradiance: SUN_SPECTRAL_IRRADIANCE,
            rayleigh_scattering_base: RAYLEIGH_SCATTERING_BASE_KM_INV,
            mie_scattering_base: mie_scattering,
            mie_extinction_base: mie_extinction,
            ozone_absorption_base: OZONE_ABSORPTION_PEAK_KM_INV,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AnalyticSunGpu {
    sun_to_scene: [f32; 3],
    angular_radius_rad: f32,
    irradiance_rec2020_w_m2: [f32; 3],
    cos_angular_radius: f32,
}

impl AnalyticSunGpu {
    fn from_sun(sun: AnalyticSun) -> Self {
        Self {
            sun_to_scene: sun.sun_to_scene.to_array(),
            angular_radius_rad: sun.angular_radius_rad,
            irradiance_rec2020_w_m2: SUN_IRRADIANCE_REC2020_W_PER_M2,
            cos_angular_radius: sun.angular_radius_rad.cos(),
        }
    }
}

fn view_radius_from_position(world_position: [f32; 4], atmosphere: AnalyticAtmosphere) -> f32 {
    let radius_m =
        world_position[1].mul_add(atmosphere.scene_units_to_m, atmosphere.world_y0_radius_m);
    radius_m.clamp(
        atmosphere.bottom_radius_m + 1.0,
        atmosphere.top_radius_m - 1.0,
    ) * M_TO_KM
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
        entries: &[uniform_entry(0), uniform_entry(1), uniform_entry(2)],
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
    sun: &'a wgpu::Buffer,
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input.sun.as_entire_binding(),
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
const _: () = assert!(core::mem::size_of::<AnalyticParamsGpu>() == 112);
const _: () = assert!(core::mem::size_of::<AnalyticSunGpu>() == 32);
