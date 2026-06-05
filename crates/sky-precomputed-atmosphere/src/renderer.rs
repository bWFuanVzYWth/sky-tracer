use std::error::Error as StdError;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use sky_realtime_atmosphere::HillaireAtmosphere;
use sky_realtime_atmosphere::atmo::{Sun, SunGpu};
use sky_realtime_atmosphere::gpu::{Gpu, RenderTargets, ViewFrame};
use sky_realtime_atmosphere::params::{
    AEROSOL_SPECIES, AerosolPreset, HillaireSettings, HillaireSpeciesGpu,
    MOLECULAR_SCATTERING_BASE, OZONE_ABSORPTION_CROSS_SECTION, SUN_SPECTRAL_IRRADIANCE,
    aerosol_preset_defaults, ozone_monthly_dobson,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const M_TO_KM: f32 = 1.0e-3;
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
const SCATTERING_SIZE: (u32, u32, u32) = (256, 128, 32);
const IRRADIANCE_SIZE: (u32, u32) = (64, 16);
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SCATTERING_ORDERS: [u32; 3] = [2, 3, 4];

#[derive(Clone, Copy, Debug)]
pub struct PrecomputedFrameParams {
    pub view: ViewFrame,
    pub atmosphere: HillaireAtmosphere,
    pub settings: HillaireSettings,
    pub aerosol: AerosolPreset,
    pub sun: Sun,
}

impl PrecomputedFrameParams {
    #[must_use]
    pub fn new(view: ViewFrame) -> Self {
        Self {
            view,
            atmosphere: HillaireAtmosphere::default(),
            settings: HillaireSettings::default(),
            aerosol: AerosolPreset::default(),
            sun: Sun::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PrecomputedParamsGpu {
    earth_radius_km: f32,
    atmosphere_thickness_km: f32,
    eye_distance_to_earth_center_km: f32,
    eye_altitude_km: f32,
    sun_dir: [f32; 3],
    sun_angular_radius_rad: f32,
    sun_spectral_irradiance: [f32; 4],
    molecular_scattering_base: [f32; 4],
    ozone_absorption_cross_section: [f32; 4],
    species: [HillaireSpeciesGpu; AEROSOL_SPECIES],
    turbidity: f32,
    ozone_mean_dobson: f32,
    mu_s_min: f32,
    mie_phase_mode: u32,
    ground_albedo_spectral: [f32; 4],
}

impl PrecomputedParamsGpu {
    fn zeroed() -> Self {
        Zeroable::zeroed()
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OrderGpu {
    scattering_order: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrecomputeKey {
    atmosphere: [u32; 4],
    settings: [u32; 6],
    sun_angular_radius: u32,
    aerosol: AerosolPreset,
}

pub struct PrecomputedAtmosphereContext {
    params_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    sun_buffer: wgpu::Buffer,
    order_buffers: [wgpu::Buffer; 3],
    sampler: wgpu::Sampler,
    resources: Resources,
    layouts: Layouts,
    pipelines: Pipelines,
    render_bind_group: wgpu::BindGroup,
    precomputed_key: Option<PrecomputeKey>,
}

impl PrecomputedAtmosphereContext {
    pub fn new(gpu: &Gpu<'_>) -> Result<Self, PrecomputedRendererError> {
        let device = gpu.device();
        let resources = Resources::new(device, gpu.queue());
        let layouts = Layouts::new(device);
        let pipelines = Pipelines::new(device, &layouts);
        let params_buffer = uniform_buffer(
            device,
            "precomputed.params.uniform",
            PrecomputedParamsGpu::zeroed(),
        );
        let view_buffer = uniform_buffer(
            device,
            "precomputed.view.uniform",
            RuntimeViewGpu::from_view(&ViewFrame::zeroed()),
        );
        let sun_buffer = uniform_buffer(
            device,
            "precomputed.sun.uniform",
            SunGpu::from_sun(Sun::default()),
        );
        let order_buffers = SCATTERING_ORDERS.map(|order| {
            uniform_buffer(
                device,
                "precomputed.order.uniform",
                OrderGpu {
                    scattering_order: order,
                    _pad: [0; 3],
                },
            )
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("precomputed.lut.sampler"),
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
                scattering: &resources.scattering_b.view,
                irradiance: &resources.irradiance_b.view,
                phase_lut: &resources.phase_lut.view,
                single_molecular: &resources.single_molecular.view,
                single_aerosol0: &resources.single_aerosol0.view,
                single_aerosol1: &resources.single_aerosol1.view,
                single_aerosol2: &resources.single_aerosol2.view,
                sampler: &sampler,
            },
        );
        Ok(Self {
            params_buffer,
            view_buffer,
            sun_buffer,
            order_buffers,
            sampler,
            resources,
            layouts,
            pipelines,
            render_bind_group,
            precomputed_key: None,
        })
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &PrecomputedFrameParams,
    ) {
        let gpu_params = precomputed_params(params);
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

        let key = PrecomputeKey::from_params(params);
        if self.precomputed_key != Some(key) {
            self.dispatch_precompute(device, encoder);
            self.precomputed_key = Some(key);
        }
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("precomputed.render.pass"),
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
            "precomputed.transmittance.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.direct_irradiance,
            &direct_irradiance_bind_group(
                device,
                &self.layouts.direct_irradiance,
                &self.params_buffer,
                &self.resources.transmittance.view,
                &self.resources.irradiance_a.view,
                &self.sampler,
            ),
            IRRADIANCE_SIZE,
            "precomputed.direct_irradiance.pass",
        );
        dispatch_compute_3d(
            encoder,
            &self.pipelines.single_scattering,
            &single_scattering_bind_group(
                device,
                &self.layouts.single_scattering,
                &self.params_buffer,
                &self.resources.transmittance.view,
                SingleScatteringOutputs {
                    molecular: &self.resources.single_molecular.view,
                    aerosol0: &self.resources.single_aerosol0.view,
                    aerosol1: &self.resources.single_aerosol1.view,
                    aerosol2: &self.resources.single_aerosol2.view,
                },
                &self.sampler,
            ),
            SCATTERING_SIZE,
            "precomputed.single_scattering.pass",
        );

        self.dispatch_order(
            device,
            encoder,
            0,
            OrderTextures {
                scattering_in: &self.resources.scattering_a,
                scattering_out: &self.resources.scattering_b,
                irradiance_in: &self.resources.irradiance_a,
                irradiance_out: &self.resources.irradiance_b,
            },
        );
        self.dispatch_order(
            device,
            encoder,
            1,
            OrderTextures {
                scattering_in: &self.resources.scattering_b,
                scattering_out: &self.resources.scattering_a,
                irradiance_in: &self.resources.irradiance_b,
                irradiance_out: &self.resources.irradiance_a,
            },
        );
        self.dispatch_order(
            device,
            encoder,
            2,
            OrderTextures {
                scattering_in: &self.resources.scattering_a,
                scattering_out: &self.resources.scattering_b,
                irradiance_in: &self.resources.irradiance_a,
                irradiance_out: &self.resources.irradiance_b,
            },
        );
    }

    fn dispatch_order(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        order_index: usize,
        textures: OrderTextures<'_>,
    ) {
        let order = &self.order_buffers[order_index];
        dispatch_compute_3d(
            encoder,
            &self.pipelines.scattering_density,
            &scattering_density_bind_group(
                device,
                &self.layouts.scattering_density,
                DensityBindGroupInput {
                    params: &self.params_buffer,
                    order,
                    transmittance: &self.resources.transmittance.view,
                    phase_lut: &self.resources.phase_lut.view,
                    single_molecular: &self.resources.single_molecular.view,
                    single_aerosol0: &self.resources.single_aerosol0.view,
                    single_aerosol1: &self.resources.single_aerosol1.view,
                    single_aerosol2: &self.resources.single_aerosol2.view,
                    multiple_scattering_in: &textures.scattering_in.view,
                    irradiance_in: &textures.irradiance_in.view,
                    density_out: &self.resources.scattering_density.view,
                    sampler: &self.sampler,
                },
            ),
            SCATTERING_SIZE,
            "precomputed.scattering_density.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.indirect_irradiance,
            &indirect_irradiance_bind_group(
                device,
                &self.layouts.indirect_irradiance,
                &self.params_buffer,
                &self.resources.scattering_density.view,
                &self.resources.delta_irradiance.view,
                &self.sampler,
            ),
            IRRADIANCE_SIZE,
            "precomputed.indirect_irradiance.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.accumulate_2d,
            &accumulate_2d_bind_group(
                device,
                &self.layouts.accumulate_2d,
                &textures.irradiance_in.view,
                &self.resources.delta_irradiance.view,
                &textures.irradiance_out.view,
                &self.sampler,
            ),
            IRRADIANCE_SIZE,
            "precomputed.accumulate_irradiance.pass",
        );
        dispatch_compute_3d(
            encoder,
            &self.pipelines.multiple_scattering,
            &multiple_scattering_bind_group(
                device,
                &self.layouts.multiple_scattering,
                &self.params_buffer,
                &self.resources.transmittance.view,
                &self.resources.scattering_density.view,
                &self.resources.delta_multiple.view,
                &self.sampler,
            ),
            SCATTERING_SIZE,
            "precomputed.multiple_scattering.pass",
        );
        if order_index == 0 {
            copy_3d_texture(
                encoder,
                &self.resources.delta_multiple,
                textures.scattering_out,
                SCATTERING_SIZE,
            );
        } else {
            dispatch_compute_3d(
                encoder,
                &self.pipelines.accumulate_3d,
                &accumulate_3d_bind_group(
                    device,
                    &self.layouts.accumulate_3d,
                    &textures.scattering_in.view,
                    &self.resources.delta_multiple.view,
                    &textures.scattering_out.view,
                    &self.sampler,
                ),
                SCATTERING_SIZE,
                "precomputed.accumulate_scattering.pass",
            );
        }
    }
}

struct OrderTextures<'a> {
    scattering_in: &'a Texture,
    scattering_out: &'a Texture,
    irradiance_in: &'a Texture,
    irradiance_out: &'a Texture,
}

struct Resources {
    transmittance: Texture,
    single_molecular: Texture,
    single_aerosol0: Texture,
    single_aerosol1: Texture,
    single_aerosol2: Texture,
    scattering_a: Texture,
    scattering_b: Texture,
    scattering_density: Texture,
    delta_multiple: Texture,
    irradiance_a: Texture,
    irradiance_b: Texture,
    delta_irradiance: Texture,
    phase_lut: Texture,
}

impl Resources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            transmittance: Texture::new_2d(
                device,
                TRANSMITTANCE_SIZE,
                "precomputed.transmittance",
                LUT_FORMAT,
            ),
            single_molecular: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.single_molecular",
                LUT_FORMAT,
            ),
            single_aerosol0: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.single_aerosol0",
                LUT_FORMAT,
            ),
            single_aerosol1: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.single_aerosol1",
                LUT_FORMAT,
            ),
            single_aerosol2: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.single_aerosol2",
                LUT_FORMAT,
            ),
            scattering_a: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.scattering_a",
                LUT_FORMAT,
            ),
            scattering_b: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.scattering_b",
                LUT_FORMAT,
            ),
            scattering_density: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.scattering_density",
                LUT_FORMAT,
            ),
            delta_multiple: Texture::new_3d(
                device,
                SCATTERING_SIZE,
                "precomputed.delta_multiple",
                LUT_FORMAT,
            ),
            irradiance_a: Texture::new_2d(
                device,
                IRRADIANCE_SIZE,
                "precomputed.irradiance_a",
                LUT_FORMAT,
            ),
            irradiance_b: Texture::new_2d(
                device,
                IRRADIANCE_SIZE,
                "precomputed.irradiance_b",
                LUT_FORMAT,
            ),
            delta_irradiance: Texture::new_2d(
                device,
                IRRADIANCE_SIZE,
                "precomputed.delta_irradiance",
                LUT_FORMAT,
            ),
            phase_lut: Texture::phase_lut(device, queue),
        }
    }
}

struct Texture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Texture {
    fn new_2d(
        device: &wgpu::Device,
        size: (u32, u32),
        label: &'static str,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
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

    fn new_3d(
        device: &wgpu::Device,
        size: (u32, u32, u32),
        label: &'static str,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: size.2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        Self {
            _texture: texture,
            view,
        }
    }

    fn phase_lut(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let width = sky_realtime_atmosphere::aerosol::PHASE_LUT_COS_BINS_U32;
        let layers = sky_realtime_atmosphere::aerosol::PHASE_LUT_SPECIES_U32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("precomputed.aerosol_phase.lut"),
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
            label: Some("precomputed.aerosol_phase.lut.view"),
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
    direct_irradiance: wgpu::BindGroupLayout,
    single_scattering: wgpu::BindGroupLayout,
    scattering_density: wgpu::BindGroupLayout,
    indirect_irradiance: wgpu::BindGroupLayout,
    multiple_scattering: wgpu::BindGroupLayout,
    accumulate_2d: wgpu::BindGroupLayout,
    accumulate_3d: wgpu::BindGroupLayout,
    render: wgpu::BindGroupLayout,
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            transmittance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.transmittance.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(1),
                ],
            }),
            direct_irradiance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.direct_irradiance.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(3),
                ],
            }),
            single_scattering: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.single_scattering.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_3d_entry(3),
                    storage_3d_entry(4),
                    storage_3d_entry(5),
                    storage_3d_entry(6),
                ],
            }),
            scattering_density: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.scattering_density.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    uniform_entry(1, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(2, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(3, wgpu::ShaderStages::COMPUTE),
                    texture_2d_array_entry(4, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(5, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(6, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(7, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(8, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(9, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(10, wgpu::ShaderStages::COMPUTE),
                    storage_3d_entry(11),
                ],
            }),
            indirect_irradiance: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("precomputed.indirect_irradiance.bgl"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(1, wgpu::ShaderStages::COMPUTE),
                        sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                        storage_2d_entry(3),
                    ],
                },
            ),
            multiple_scattering: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("precomputed.multiple_scattering.bgl"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(2, wgpu::ShaderStages::COMPUTE),
                        sampler_entry(3, wgpu::ShaderStages::COMPUTE),
                        storage_3d_entry(4),
                    ],
                },
            ),
            accumulate_2d: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.accumulate_2d.bgl"),
                entries: &[
                    texture_2d_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(3),
                ],
            }),
            accumulate_3d: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.accumulate_3d.bgl"),
                entries: &[
                    texture_3d_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_3d_entry(3),
                ],
            }),
            render: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precomputed.render.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(1, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(3, wgpu::ShaderStages::FRAGMENT),
                    texture_3d_entry(4, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(5, wgpu::ShaderStages::FRAGMENT),
                    sampler_entry(6, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_array_entry(7, wgpu::ShaderStages::FRAGMENT),
                    texture_3d_entry(8, wgpu::ShaderStages::FRAGMENT),
                    texture_3d_entry(9, wgpu::ShaderStages::FRAGMENT),
                    texture_3d_entry(10, wgpu::ShaderStages::FRAGMENT),
                    texture_3d_entry(11, wgpu::ShaderStages::FRAGMENT),
                ],
            }),
        }
    }
}

struct Pipelines {
    transmittance: wgpu::ComputePipeline,
    direct_irradiance: wgpu::ComputePipeline,
    single_scattering: wgpu::ComputePipeline,
    scattering_density: wgpu::ComputePipeline,
    indirect_irradiance: wgpu::ComputePipeline,
    multiple_scattering: wgpu::ComputePipeline,
    accumulate_2d: wgpu::ComputePipeline,
    accumulate_3d: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device, layouts: &Layouts) -> Self {
        Self {
            transmittance: compute_pipeline(
                device,
                &layouts.transmittance,
                "precomputed.transmittance.pipeline",
                include_str!("wgsl/transmittance.comp.wgsl"),
            ),
            direct_irradiance: compute_pipeline(
                device,
                &layouts.direct_irradiance,
                "precomputed.direct_irradiance.pipeline",
                include_str!("wgsl/direct_irradiance.comp.wgsl"),
            ),
            single_scattering: compute_pipeline(
                device,
                &layouts.single_scattering,
                "precomputed.single_scattering.pipeline",
                include_str!("wgsl/single_scattering.comp.wgsl"),
            ),
            scattering_density: compute_pipeline(
                device,
                &layouts.scattering_density,
                "precomputed.scattering_density.pipeline",
                include_str!("wgsl/scattering_density.comp.wgsl"),
            ),
            indirect_irradiance: compute_pipeline(
                device,
                &layouts.indirect_irradiance,
                "precomputed.indirect_irradiance.pipeline",
                include_str!("wgsl/indirect_irradiance.comp.wgsl"),
            ),
            multiple_scattering: compute_pipeline(
                device,
                &layouts.multiple_scattering,
                "precomputed.multiple_scattering.pipeline",
                include_str!("wgsl/multiple_scattering.comp.wgsl"),
            ),
            accumulate_2d: compute_pipeline_raw(
                device,
                &layouts.accumulate_2d,
                "precomputed.accumulate_2d.pipeline",
                include_str!("wgsl/accumulate_2d.comp.wgsl"),
            ),
            accumulate_3d: compute_pipeline_raw(
                device,
                &layouts.accumulate_3d,
                "precomputed.accumulate_3d.pipeline",
                include_str!("wgsl/accumulate_3d.comp.wgsl"),
            ),
            render: render_pipeline(device, &layouts.render),
        }
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    shader_source: &str,
) -> wgpu::ComputePipeline {
    let source = format!("{}\n\n{}", crate::COMMON_WGSL, shader_source);
    compute_pipeline_raw(device, layout, label, &source)
}

fn compute_pipeline_raw(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    shader_source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
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
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("precomputed.render.pipeline"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("precomputed.render.pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("precomputed.render.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
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

fn precomputed_params(params: &PrecomputedFrameParams) -> PrecomputedParamsGpu {
    let atmosphere = params.atmosphere;
    let view_radius_m = view_radius_from_position(params.view.world_position, atmosphere);
    let earth_radius_km = atmosphere.bottom_radius_m * M_TO_KM;
    let atmosphere_thickness_km = (atmosphere.top_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let eye_distance_to_earth_center_km = view_radius_m * M_TO_KM;
    let eye_altitude_km = (view_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let sun_dir = params.sun.to_sun().normalize_or_zero();
    let species_defaults = aerosol_preset_defaults(params.aerosol);

    let mut p = PrecomputedParamsGpu::zeroed();
    p.earth_radius_km = earth_radius_km;
    p.atmosphere_thickness_km = atmosphere_thickness_km;
    p.eye_distance_to_earth_center_km = eye_distance_to_earth_center_km;
    p.eye_altitude_km = eye_altitude_km;
    p.sun_dir = sun_dir.to_array();
    p.sun_angular_radius_rad = params.sun.angular_radius_rad;
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
    p.mu_s_min = (-102.0_f32).to_radians().cos();
    p.ground_albedo_spectral = params.settings.ground_albedo_spectral;
    p
}

impl PrecomputeKey {
    fn from_params(params: &PrecomputedFrameParams) -> Self {
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
            sun_angular_radius: params.sun.angular_radius_rad.to_bits(),
            aerosol: params.aerosol,
        }
    }
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
    size: (u32, u32),
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
}

fn dispatch_compute_3d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    size: (u32, u32, u32),
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(size.0.div_ceil(4), size.1.div_ceil(4), size.2);
}

fn copy_3d_texture(
    encoder: &mut wgpu::CommandEncoder,
    src: &Texture,
    dst: &Texture,
    size: (u32, u32, u32),
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &dst._texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: size.2,
        },
    );
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

const fn texture_3d_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
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

const fn storage_3d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: LUT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn sampler_binding(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

fn buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn transmittance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    out: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.transmittance.bg"),
        layout,
        entries: &[buffer_binding(0, params), texture_binding(1, out)],
    })
}

fn direct_irradiance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    out: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.direct_irradiance.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, transmittance),
            sampler_binding(2, sampler),
            texture_binding(3, out),
        ],
    })
}

fn single_scattering_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    outputs: SingleScatteringOutputs<'_>,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.single_scattering.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, transmittance),
            sampler_binding(2, sampler),
            texture_binding(3, outputs.molecular),
            texture_binding(4, outputs.aerosol0),
            texture_binding(5, outputs.aerosol1),
            texture_binding(6, outputs.aerosol2),
        ],
    })
}

struct SingleScatteringOutputs<'a> {
    molecular: &'a wgpu::TextureView,
    aerosol0: &'a wgpu::TextureView,
    aerosol1: &'a wgpu::TextureView,
    aerosol2: &'a wgpu::TextureView,
}

struct DensityBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    order: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
    single_molecular: &'a wgpu::TextureView,
    single_aerosol0: &'a wgpu::TextureView,
    single_aerosol1: &'a wgpu::TextureView,
    single_aerosol2: &'a wgpu::TextureView,
    multiple_scattering_in: &'a wgpu::TextureView,
    irradiance_in: &'a wgpu::TextureView,
    density_out: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
}

fn scattering_density_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: DensityBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.scattering_density.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.order),
            texture_binding(2, input.transmittance),
            sampler_binding(3, input.sampler),
            texture_binding(4, input.phase_lut),
            texture_binding(5, input.single_molecular),
            texture_binding(6, input.single_aerosol0),
            texture_binding(7, input.single_aerosol1),
            texture_binding(8, input.single_aerosol2),
            texture_binding(9, input.multiple_scattering_in),
            texture_binding(10, input.irradiance_in),
            texture_binding(11, input.density_out),
        ],
    })
}

fn indirect_irradiance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    scattering_density: &wgpu::TextureView,
    out: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.indirect_irradiance.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, scattering_density),
            sampler_binding(2, sampler),
            texture_binding(3, out),
        ],
    })
}

fn multiple_scattering_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    scattering_density: &wgpu::TextureView,
    out: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.multiple_scattering.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, transmittance),
            texture_binding(2, scattering_density),
            sampler_binding(3, sampler),
            texture_binding(4, out),
        ],
    })
}

fn accumulate_2d_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    current: &wgpu::TextureView,
    delta: &wgpu::TextureView,
    out: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.accumulate_2d.bg"),
        layout,
        entries: &[
            texture_binding(0, current),
            texture_binding(1, delta),
            sampler_binding(2, sampler),
            texture_binding(3, out),
        ],
    })
}

fn accumulate_3d_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    current: &wgpu::TextureView,
    delta: &wgpu::TextureView,
    out: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.accumulate_3d.bg"),
        layout,
        entries: &[
            texture_binding(0, current),
            texture_binding(1, delta),
            sampler_binding(2, sampler),
            texture_binding(3, out),
        ],
    })
}

struct RenderBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    view: &'a wgpu::Buffer,
    sun: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    scattering: &'a wgpu::TextureView,
    irradiance: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
    single_molecular: &'a wgpu::TextureView,
    single_aerosol0: &'a wgpu::TextureView,
    single_aerosol1: &'a wgpu::TextureView,
    single_aerosol2: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: RenderBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("precomputed.render.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.view),
            buffer_binding(2, input.sun),
            texture_binding(3, input.transmittance),
            texture_binding(4, input.scattering),
            texture_binding(5, input.irradiance),
            sampler_binding(6, input.sampler),
            texture_binding(7, input.phase_lut),
            texture_binding(8, input.single_molecular),
            texture_binding(9, input.single_aerosol0),
            texture_binding(10, input.single_aerosol1),
            texture_binding(11, input.single_aerosol2),
        ],
    })
}

#[derive(Debug)]
pub enum PrecomputedRendererError {
    ResourceSizeOverflow,
}

impl fmt::Display for PrecomputedRendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceSizeOverflow => {
                f.write_str("precomputed atmosphere resource size overflow")
            }
        }
    }
}

impl StdError for PrecomputedRendererError {}

const _: () = assert!(core::mem::size_of::<PrecomputedParamsGpu>() == 256);
const _: () = assert!(core::mem::size_of::<RuntimeViewGpu>() == 80);
const _: () = assert!(core::mem::size_of::<OrderGpu>() == 16);
