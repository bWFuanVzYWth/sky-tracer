use std::error::Error as StdError;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use glam::UVec2;
use sky_realtime_atmosphere::HillaireAtmosphere;
use sky_realtime_atmosphere::atmo::{Sun, SunGpu};
use sky_realtime_atmosphere::gpu::{Gpu, RenderTargets, ViewFrame};
use sky_realtime_atmosphere::params::{
    AEROSOL_SPECIES, AerosolPreset, HillaireParamsGpu, HillairePhaseMode, HillaireSettings,
    HillaireSpeciesGpu, MOLECULAR_SCATTERING_BASE, OZONE_ABSORPTION_CROSS_SECTION,
    SUN_SPECTRAL_IRRADIANCE, aerosol_preset_defaults, ozone_monthly_dobson,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const M_TO_KM: f32 = 1.0e-3;
const TRANSMITTANCE_SIZE: UVec2 = UVec2::new(256, 64);
const MULTI_SCATTERING_SIZE: UVec2 = UVec2::new(32, 32);
const SKY_VIEW_SIZE: UVec2 = UVec2::new(256, 512);
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy, Debug)]
pub struct UnrealFrameParams {
    pub view: ViewFrame,
    pub atmosphere: HillaireAtmosphere,
    pub settings: HillaireSettings,
    pub aerosol: AerosolPreset,
    pub phase_mode: HillairePhaseMode,
    pub sun: Sun,
}

impl UnrealFrameParams {
    #[must_use]
    pub fn new(view: ViewFrame) -> Self {
        Self {
            view,
            atmosphere: HillaireAtmosphere::default(),
            settings: HillaireSettings::default(),
            aerosol: AerosolPreset::default(),
            phase_mode: HillairePhaseMode::default(),
            sun: Sun::default(),
        }
    }
}

pub struct UnrealAtmosphereContext {
    params_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    sun_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    resources: Resources,
    layouts: Layouts,
    pipelines: Pipelines,
    render_bind_group: wgpu::BindGroup,
    precompute_key: Option<PrecomputeKey>,
}

impl UnrealAtmosphereContext {
    pub fn new(gpu: &Gpu<'_>) -> Result<Self, UnrealRendererError> {
        let device = gpu.device();
        let resources = Resources::new(device, gpu.queue());
        let layouts = Layouts::new(device);
        let pipelines = Pipelines::new(device, &layouts);
        let params_buffer =
            uniform_buffer(device, "unreal.params.uniform", HillaireParamsGpu::zeroed());
        let view_buffer = uniform_buffer(
            device,
            "unreal.view.uniform",
            RuntimeViewGpu::from_view(&ViewFrame::zeroed()),
        );
        let sun_buffer = uniform_buffer(
            device,
            "unreal.sun.uniform",
            SunGpu::from_sun(Sun::default()),
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("unreal.lut.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let render_bind_group = render_bind_group(
            device,
            &layouts.render,
            RenderBindGroupInput {
                params: &params_buffer,
                view: &view_buffer,
                sun: &sun_buffer,
                transmittance: &resources.transmittance.view,
                sampler: &sampler,
                sky_view: &resources.sky_view.view,
            },
        );
        Ok(Self {
            params_buffer,
            view_buffer,
            sun_buffer,
            sampler,
            resources,
            layouts,
            pipelines,
            render_bind_group,
            precompute_key: None,
        })
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &UnrealFrameParams,
    ) {
        let gpu_params = unreal_params(params);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&gpu_params));
        queue.write_buffer(
            &self.view_buffer,
            0,
            bytemuck::bytes_of(&RuntimeViewGpu::from_view(&params.view)),
        );
        queue.write_buffer(
            &self.sun_buffer,
            0,
            bytemuck::bytes_of(&SunGpu::from_sun(params.sun)),
        );

        let key = PrecomputeKey::from_params(params, gpu_params);
        if self.precompute_key != Some(key) {
            self.dispatch_precompute(device, encoder);
            self.precompute_key = Some(key);
        }
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("unreal.render.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: targets.post_view(),
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
        pass.set_pipeline(&self.pipelines.render);
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn dispatch_precompute(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        dispatch_compute_2d(
            encoder,
            &self.pipelines.transmittance,
            &transmittance_bind_group(
                device,
                &self.layouts.transmittance,
                &self.params_buffer,
                &self.resources.transmittance.view,
            ),
            TRANSMITTANCE_SIZE,
            "unreal.transmittance.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.multi_scattering,
            &multi_scattering_bind_group(
                device,
                &self.layouts.multi_scattering,
                MultiScatteringBindGroupInput {
                    params: &self.params_buffer,
                    transmittance: &self.resources.transmittance.view,
                    sampler: &self.sampler,
                    multi_scattering: &self.resources.multi_scattering.view,
                    phase_lut: &self.resources.phase_lut.view,
                },
            ),
            MULTI_SCATTERING_SIZE,
            "unreal.multi_scattering.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.sky_view,
            &sky_view_bind_group(
                device,
                &self.layouts.sky_view,
                SkyViewBindGroupInput {
                    params: &self.params_buffer,
                    transmittance: &self.resources.transmittance.view,
                    sampler: &self.sampler,
                    multi_scattering: &self.resources.multi_scattering.view,
                    sky_view: &self.resources.sky_view.view,
                    phase_lut: &self.resources.phase_lut.view,
                },
            ),
            SKY_VIEW_SIZE,
            "unreal.sky_view.pass",
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RuntimeViewGpu {
    relative_world_from_clip: [[f32; 4]; 4],
    world_position: [f32; 4],
}

impl RuntimeViewGpu {
    const fn from_view(view: &ViewFrame) -> Self {
        Self {
            relative_world_from_clip: view.relative_world_from_clip,
            world_position: view.world_position,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrecomputeKey {
    atmosphere: [u32; 4],
    settings: [u32; 6],
    sun_dir: [u32; 3],
    sky_view_height: u32,
    sun_angular_radius: u32,
    phase_mode: HillairePhaseMode,
    aerosol: AerosolPreset,
}

impl PrecomputeKey {
    fn from_params(params: &UnrealFrameParams, gpu_params: HillaireParamsGpu) -> Self {
        let atmosphere = params.atmosphere;
        Self {
            atmosphere: [
                atmosphere.bottom_radius_m.to_bits(),
                atmosphere.top_radius_m.to_bits(),
                atmosphere.world_y0_radius_m.to_bits(),
                atmosphere.scene_units_to_m.to_bits(),
            ],
            settings: [
                params.settings.month,
                params.settings.aerosol_turbidity.to_bits(),
                params.settings.ground_albedo_spectral[0].to_bits(),
                params.settings.ground_albedo_spectral[1].to_bits(),
                params.settings.ground_albedo_spectral[2].to_bits(),
                params.settings.ground_albedo_spectral[3].to_bits(),
            ],
            sun_dir: [
                gpu_params.sun_dir[0].to_bits(),
                gpu_params.sun_dir[1].to_bits(),
                gpu_params.sun_dir[2].to_bits(),
            ],
            sky_view_height: gpu_params.sky_view_height_km.to_bits(),
            sun_angular_radius: params.sun.angular_radius_rad.to_bits(),
            phase_mode: params.phase_mode,
            aerosol: params.aerosol,
        }
    }
}

struct Resources {
    transmittance: Texture2d,
    multi_scattering: Texture2d,
    sky_view: Texture2d,
    phase_lut: TextureArray,
}

impl Resources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            transmittance: Texture2d::new(device, TRANSMITTANCE_SIZE, "unreal.transmittance"),
            multi_scattering: Texture2d::new(
                device,
                MULTI_SCATTERING_SIZE,
                "unreal.multi_scattering",
            ),
            sky_view: Texture2d::new(device, SKY_VIEW_SIZE, "unreal.sky_view"),
            phase_lut: TextureArray::phase_lut(device, queue),
        }
    }
}

struct Texture2d {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Texture2d {
    fn new(device: &wgpu::Device, size: UVec2, label: &'static str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });
        Self {
            _texture: texture,
            view,
        }
    }
}

struct TextureArray {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl TextureArray {
    fn phase_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let width = sky_realtime_atmosphere::aerosol::PHASE_LUT_COS_BINS_U32;
        let layers = sky_realtime_atmosphere::aerosol::PHASE_LUT_SPECIES_U32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("unreal.aerosol_phase.lut"),
            size: wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (z, species_lut) in (0..layers).zip(sky_realtime_atmosphere::aerosol::PHASE_LUTS.iter())
        {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(*species_lut),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4 * 4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("unreal.aerosol_phase.lut.view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            _texture: texture,
            view,
        }
    }
}

struct Layouts {
    transmittance: wgpu::BindGroupLayout,
    multi_scattering: wgpu::BindGroupLayout,
    sky_view: wgpu::BindGroupLayout,
    render: wgpu::BindGroupLayout,
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            transmittance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unreal.transmittance.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(1),
                ],
            }),
            multi_scattering: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unreal.multi_scattering.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(3),
                    texture_2d_array_entry(4, wgpu::ShaderStages::COMPUTE),
                ],
            }),
            sky_view: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unreal.sky_view.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(3, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(4),
                    texture_2d_array_entry(5, wgpu::ShaderStages::COMPUTE),
                ],
            }),
            render: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unreal.render.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(3, wgpu::ShaderStages::FRAGMENT),
                    sampler_entry(4, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(5, wgpu::ShaderStages::FRAGMENT),
                ],
            }),
        }
    }
}

struct Pipelines {
    transmittance: wgpu::ComputePipeline,
    multi_scattering: wgpu::ComputePipeline,
    sky_view: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device, layouts: &Layouts) -> Self {
        Self {
            transmittance: compute_pipeline(
                device,
                &layouts.transmittance,
                "unreal.transmittance.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
                    include_str!("wgsl/transmittance.comp.wgsl")
                ),
            ),
            multi_scattering: compute_pipeline(
                device,
                &layouts.multi_scattering,
                "unreal.multi_scattering.pipeline",
                &format!(
                    "{}\n\n{}\n\n{}",
                    crate::COMMON_WGSL,
                    crate::INSCATTER_WGSL,
                    include_str!("wgsl/multi_scattering.comp.wgsl")
                ),
            ),
            sky_view: compute_pipeline(
                device,
                &layouts.sky_view,
                "unreal.sky_view.pipeline",
                &format!(
                    "{}\n\n{}\n\n{}",
                    crate::COMMON_WGSL,
                    crate::INSCATTER_WGSL,
                    include_str!("wgsl/sky_view.comp.wgsl")
                ),
            ),
            render: render_pipeline(device, &layouts.render),
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
        source: wgpu::ShaderSource::Wgsl(source.into()),
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

fn render_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let source = format!(
        "{}\n\n{}\n\n{}",
        crate::COMMON_WGSL,
        sky_realtime_atmosphere::atmo::SUN_WGSL,
        include_str!("wgsl/render_sky.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("unreal.render.pipeline"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("unreal.render.pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("unreal.render.pipeline"),
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
                format: sky_realtime_atmosphere::gpu::SCENE_RADIANCE_FORMAT,
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

fn unreal_params(params: &UnrealFrameParams) -> HillaireParamsGpu {
    let atmosphere = params.atmosphere;
    let view_radius_m = view_radius_from_position(params.view.world_position, atmosphere);
    let earth_radius_km = atmosphere.bottom_radius_m * M_TO_KM;
    let atmosphere_thickness_km = (atmosphere.top_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let eye_distance_to_earth_center_km = view_radius_m * M_TO_KM;
    let eye_altitude_km = (view_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let sun_dir = params.sun.to_sun().normalize_or_zero();
    let species_defaults = aerosol_preset_defaults(params.aerosol);

    let mut p = HillaireParamsGpu::zeroed();
    p.earth_radius_km = earth_radius_km;
    p.atmosphere_thickness_km = atmosphere_thickness_km;
    p.eye_distance_to_earth_center_km = eye_distance_to_earth_center_km;
    p.eye_altitude_km = eye_altitude_km;
    p.sun_dir = sun_dir.to_array();
    p.sky_view_height_km = eye_distance_to_earth_center_km.clamp(
        earth_radius_km + 1.0e-3,
        earth_radius_km + atmosphere_thickness_km - 1.0e-3,
    );
    p.sun_spectral_irradiance = SUN_SPECTRAL_IRRADIANCE;
    p.molecular_scattering_base = MOLECULAR_SCATTERING_BASE;
    p.ozone_absorption_cross_section = OZONE_ABSORPTION_CROSS_SECTION;

    for (slot, ((base, bg, scale), (sigma_sca, sigma_abs))) in p.species.iter_mut().zip(
        species_defaults.iter().zip(
            sky_realtime_atmosphere::aerosol::SIGMA_SCA
                .iter()
                .zip(sky_realtime_atmosphere::aerosol::SIGMA_ABS.iter()),
        ),
    ) {
        *slot = HillaireSpeciesGpu {
            sigma_sca: *sigma_sca,
            sigma_abs: *sigma_abs,
            base_density: *base,
            bg_density: *bg,
            height_scale: *scale,
            pad: 0.0,
        };
    }

    p.turbidity = params.settings.aerosol_turbidity;
    p.ozone_mean_dobson = ozone_monthly_dobson(params.settings.month);
    p.mie_phase_mode = params.phase_mode.as_gpu_u32();
    p.ground_albedo_spectral = params.settings.ground_albedo_spectral;
    p
}

fn view_radius_from_position(world_position: [f32; 4], atmosphere: HillaireAtmosphere) -> f32 {
    let radius_m =
        world_position[1].mul_add(atmosphere.scene_units_to_m, atmosphere.world_y0_radius_m);
    radius_m.clamp(
        atmosphere.bottom_radius_m + 1.0,
        atmosphere.top_radius_m - 1.0,
    )
}

fn dispatch_compute_2d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    size: UVec2,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
}

fn uniform_buffer<T: Pod>(device: &wgpu::Device, label: &'static str, value: T) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

const fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn texture_2d_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

const fn texture_2d_array_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

const fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

const fn storage_2d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: LUT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

const fn sampler_binding(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

fn transmittance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    out: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unreal.transmittance.bg"),
        layout,
        entries: &[buffer_binding(0, params), texture_binding(1, out)],
    })
}

struct MultiScatteringBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    multi_scattering: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
}

fn multi_scattering_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: MultiScatteringBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unreal.multi_scattering.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            texture_binding(3, input.multi_scattering),
            texture_binding(4, input.phase_lut),
        ],
    })
}

struct SkyViewBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    multi_scattering: &'a wgpu::TextureView,
    sky_view: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
}

fn sky_view_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: SkyViewBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unreal.sky_view.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            texture_binding(3, input.multi_scattering),
            texture_binding(4, input.sky_view),
            texture_binding(5, input.phase_lut),
        ],
    })
}

struct RenderBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    view: &'a wgpu::Buffer,
    sun: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    sky_view: &'a wgpu::TextureView,
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: RenderBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unreal.render.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.view),
            buffer_binding(2, input.sun),
            texture_binding(3, input.transmittance),
            sampler_binding(4, input.sampler),
            texture_binding(5, input.sky_view),
        ],
    })
}

#[derive(Debug)]
pub enum UnrealRendererError {
    ResourceSizeOverflow,
}

impl fmt::Display for UnrealRendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceSizeOverflow => f.write_str("unreal atmosphere resource size overflow"),
        }
    }
}

impl StdError for UnrealRendererError {}

const _: () = assert!(core::mem::size_of::<RuntimeViewGpu>() == 80);
const _: () = assert!(core::mem::size_of::<HillaireParamsGpu>() == 256);
const _: () = assert!(AEROSOL_SPECIES == 3);
