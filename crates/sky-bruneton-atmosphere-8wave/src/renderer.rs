use std::error::Error as StdError;
use std::fmt;

use bytemuck::{Pod, Zeroable};
use glam::{UVec2, UVec3};

use crate::atmosphere::HillaireAtmosphere;
use crate::gpu::{Gpu, RenderTargets, SCENE_RADIANCE_FORMAT, ViewFrame};
use crate::params::{
    AEROSOL_SPECIES, AerosolPreset, HillaireParamsGpu, HillairePhaseMode, HillaireSettings,
    HillaireSpeciesGpu, MOLECULAR_SCATTERING_BASE, OZONE_ABSORPTION_CROSS_SECTION,
    SPECTRAL_GROUP_COUNT, SUN_SPECTRAL_IRRADIANCE, aerosol_preset_defaults, ozone_monthly_dobson,
};
use crate::sun::{SUN_WGSL, Sun, SunGpu};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const M_TO_KM: f32 = 1.0e-3;
const TRANSMITTANCE_SIZE: UVec2 = UVec2::new(512, 128);
const IRRADIANCE_SIZE: UVec2 = UVec2::new(128, 32);
const SCATTERING_R_SIZE: u32 = 64;
const SCATTERING_MU_SIZE: u32 = 128;
const SCATTERING_MU_S_SIZE: u32 = 64;
const SCATTERING_NU_SIZE: u32 = 64;
const SCATTERING_SIZE: UVec3 = UVec3::new(
    SCATTERING_MU_S_SIZE * SCATTERING_NU_SIZE,
    SCATTERING_MU_SIZE,
    SCATTERING_R_SIZE,
);
const SKY_VIEW_SIZE: UVec2 = UVec2::new(256, 256);
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SCATTERING_ORDER_COUNT: u32 = 4;

#[derive(Clone, Copy, Debug)]
enum SpectralGroup {
    Low,
    High,
}

impl SpectralGroup {
    const ALL: [Self; SPECTRAL_GROUP_COUNT] = [Self::Low, Self::High];

    const fn index(self) -> usize {
        match self {
            Self::Low => 0,
            Self::High => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BrunetonFrameParams {
    pub view: ViewFrame,
    pub atmosphere: HillaireAtmosphere,
    pub settings: HillaireSettings,
    pub aerosol: AerosolPreset,
    pub phase_mode: HillairePhaseMode,
    pub sun: Sun,
}

impl BrunetonFrameParams {
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

pub struct BrunetonAtmosphereContext {
    params_buffers: [wgpu::Buffer; SPECTRAL_GROUP_COUNT],
    view_buffer: wgpu::Buffer,
    sun_buffer: wgpu::Buffer,
    order_buffers: Vec<wgpu::Buffer>,
    sampler: wgpu::Sampler,
    resources: [Resources; SPECTRAL_GROUP_COUNT],
    layouts: Layouts,
    pipelines: Pipelines,
    render_bind_group: wgpu::BindGroup,
    precompute_key: Option<PrecomputeKey>,
}

impl BrunetonAtmosphereContext {
    pub fn new(gpu: &Gpu<'_>) -> Result<Self, BrunetonRendererError> {
        let device = gpu.device();
        let resources = [
            Resources::new(device, gpu.queue(), SpectralGroup::Low),
            Resources::new(device, gpu.queue(), SpectralGroup::High),
        ];
        let layouts = Layouts::new(device);
        let pipelines = Pipelines::new(device, &layouts);
        let params_buffers = [
            uniform_buffer(
                device,
                "bruneton.params.low.uniform",
                HillaireParamsGpu::zeroed(),
            ),
            uniform_buffer(
                device,
                "bruneton.params.high.uniform",
                HillaireParamsGpu::zeroed(),
            ),
        ];
        let view_buffer = uniform_buffer(
            device,
            "bruneton.view.uniform",
            RuntimeViewGpu::from_view(&ViewFrame::zeroed()),
        );
        let sun_buffer = uniform_buffer(
            device,
            "bruneton.sun.uniform",
            SunGpu::from_sun(Sun::default()),
        );
        let order_buffers = (2..=SCATTERING_ORDER_COUNT)
            .map(|order| {
                uniform_buffer(
                    device,
                    "bruneton.scattering_order.uniform",
                    ScatteringOrderGpu {
                        order,
                        _pad: [0; 3],
                    },
                )
            })
            .collect();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bruneton.lut.sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let render_bind_group = render_bind_group(
            device,
            &layouts.render,
            RenderBindGroupInput {
                params: &params_buffers[SpectralGroup::Low.index()],
                view: &view_buffer,
                sun: &sun_buffer,
                transmittance_low: &resources[SpectralGroup::Low.index()].transmittance.view,
                transmittance_high: &resources[SpectralGroup::High.index()].transmittance.view,
                sampler: &sampler,
                sky_view_low: &resources[SpectralGroup::Low.index()].sky_view.view,
                sky_view_high: &resources[SpectralGroup::High.index()].sky_view.view,
            },
        );
        Ok(Self {
            params_buffers,
            view_buffer,
            sun_buffer,
            order_buffers,
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
        params: &BrunetonFrameParams,
    ) {
        for group in SpectralGroup::ALL {
            let gpu_params = bruneton_params(params, group);
            queue.write_buffer(
                &self.params_buffers[group.index()],
                0,
                bytemuck::bytes_of(&gpu_params),
            );
        }
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
        if self.precompute_key != Some(key) {
            for group in SpectralGroup::ALL {
                self.dispatch_precompute(device, encoder, group);
            }
            self.precompute_key = Some(key);
        }
        for group in SpectralGroup::ALL {
            self.dispatch_sky_view(device, encoder, group);
        }
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bruneton.render.pass"),
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

    fn dispatch_precompute(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        group: SpectralGroup,
    ) {
        let params_buffer = &self.params_buffers[group.index()];
        let resources = &self.resources[group.index()];
        dispatch_compute_2d(
            encoder,
            &self.pipelines.transmittance,
            &transmittance_bind_group(
                device,
                &self.layouts.transmittance,
                params_buffer,
                &resources.transmittance.view,
            ),
            TRANSMITTANCE_SIZE,
            "bruneton.transmittance.pass",
        );
        dispatch_compute_2d(
            encoder,
            &self.pipelines.direct_irradiance,
            &direct_irradiance_bind_group(
                device,
                &self.layouts.direct_irradiance,
                DirectIrradianceBindGroupInput {
                    params: params_buffer,
                    transmittance: &resources.transmittance.view,
                    sampler: &self.sampler,
                    irradiance: &resources.irradiance.view,
                    delta_irradiance: &resources.delta_irradiance.view,
                },
            ),
            IRRADIANCE_SIZE,
            "bruneton.direct_irradiance.pass",
        );
        dispatch_compute_3d(
            encoder,
            &self.pipelines.single_scattering,
            &single_scattering_bind_group(
                device,
                &self.layouts.single_scattering,
                SingleScatteringBindGroupInput {
                    params: params_buffer,
                    transmittance: &resources.transmittance.view,
                    sampler: &self.sampler,
                    scattering: &resources.scattering.view,
                    single_mie: &resources.single_mie.view,
                    single_rayleigh: &resources.single_rayleigh.view,
                    phase_lut: &resources.phase_lut.view,
                },
            ),
            SCATTERING_SIZE,
            "bruneton.single_scattering.pass",
        );

        for order_buffer in &self.order_buffers {
            dispatch_compute_3d(
                encoder,
                &self.pipelines.scattering_density,
                &scattering_density_bind_group(
                    device,
                    &self.layouts.scattering_density,
                    ScatteringDensityBindGroupInput {
                        params: params_buffer,
                        order: order_buffer,
                        irradiance: &resources.delta_irradiance.view,
                        scattering: &resources.scattering.view,
                        single_mie: &resources.single_mie.view,
                        delta_scattering: &resources.delta_scattering.view,
                        sampler: &self.sampler,
                        phase_lut: &resources.phase_lut.view,
                        density: &resources.scattering_density.view,
                    },
                ),
                SCATTERING_SIZE,
                "bruneton.scattering_density.pass",
            );
            dispatch_compute_2d(
                encoder,
                &self.pipelines.indirect_irradiance,
                &indirect_irradiance_bind_group(
                    device,
                    &self.layouts.indirect_irradiance,
                    IndirectIrradianceBindGroupInput {
                        params: params_buffer,
                        order: order_buffer,
                        scattering: &resources.scattering.view,
                        single_mie: &resources.single_mie.view,
                        delta_scattering: &resources.delta_scattering.view,
                        irradiance: &resources.irradiance.view,
                        sampler: &self.sampler,
                        irradiance_out: &resources.irradiance_next.view,
                        delta_irradiance_out: &resources.delta_irradiance.view,
                        phase_lut: &resources.phase_lut.view,
                    },
                ),
                IRRADIANCE_SIZE,
                "bruneton.indirect_irradiance.pass",
            );
            copy_texture_2d(
                encoder,
                &resources.irradiance_next.texture,
                &resources.irradiance.texture,
                IRRADIANCE_SIZE,
            );
            dispatch_compute_3d(
                encoder,
                &self.pipelines.multiple_scattering,
                &multiple_scattering_bind_group(
                    device,
                    &self.layouts.multiple_scattering,
                    MultipleScatteringBindGroupInput {
                        params: params_buffer,
                        transmittance: &resources.transmittance.view,
                        density: &resources.scattering_density.view,
                        scattering: &resources.scattering.view,
                        sampler: &self.sampler,
                        delta_scattering: &resources.delta_scattering.view,
                        scattering_accum: &resources.scattering_next.view,
                    },
                ),
                SCATTERING_SIZE,
                "bruneton.multiple_scattering.pass",
            );
            copy_texture_3d(
                encoder,
                &resources.scattering_next.texture,
                &resources.scattering.texture,
                SCATTERING_SIZE,
            );
        }
    }

    fn dispatch_sky_view(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        group: SpectralGroup,
    ) {
        let params_buffer = &self.params_buffers[group.index()];
        let resources = &self.resources[group.index()];
        dispatch_compute_2d(
            encoder,
            &self.pipelines.sky_view,
            &sky_view_bind_group(
                device,
                &self.layouts.sky_view,
                SkyViewBindGroupInput {
                    params: params_buffer,
                    transmittance: &resources.transmittance.view,
                    irradiance: &resources.irradiance.view,
                    scattering: &resources.scattering.view,
                    single_rayleigh: &resources.single_rayleigh.view,
                    sampler: &self.sampler,
                    phase_lut: &resources.phase_lut.view,
                    sky_view: &resources.sky_view.view,
                },
            ),
            SKY_VIEW_SIZE,
            "bruneton.sky_view.pass",
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ScatteringOrderGpu {
    order: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrecomputeKey {
    atmosphere: [u32; 4],
    settings: [u32; 6],
    phase_mode: HillairePhaseMode,
    aerosol: AerosolPreset,
}

impl PrecomputeKey {
    fn from_params(params: &BrunetonFrameParams) -> Self {
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
            phase_mode: params.phase_mode,
            aerosol: params.aerosol,
        }
    }
}

struct Resources {
    transmittance: Texture2d,
    irradiance: Texture2d,
    delta_irradiance: Texture2d,
    irradiance_next: Texture2d,
    scattering: Texture3d,
    scattering_next: Texture3d,
    single_rayleigh: Texture3d,
    single_mie: Texture3d,
    delta_scattering: Texture3d,
    scattering_density: Texture3d,
    sky_view: Texture2d,
    phase_lut: TextureArray,
}

impl Resources {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, group: SpectralGroup) -> Self {
        Self {
            transmittance: Texture2d::new(device, TRANSMITTANCE_SIZE, "bruneton.transmittance"),
            irradiance: Texture2d::new(device, IRRADIANCE_SIZE, "bruneton.irradiance"),
            delta_irradiance: Texture2d::new(device, IRRADIANCE_SIZE, "bruneton.delta_irradiance"),
            irradiance_next: Texture2d::new(device, IRRADIANCE_SIZE, "bruneton.irradiance_next"),
            scattering: Texture3d::new(device, SCATTERING_SIZE, "bruneton.scattering"),
            scattering_next: Texture3d::new(device, SCATTERING_SIZE, "bruneton.scattering_next"),
            single_rayleigh: Texture3d::new(device, SCATTERING_SIZE, "bruneton.single_rayleigh"),
            single_mie: Texture3d::new(device, SCATTERING_SIZE, "bruneton.single_mie"),
            delta_scattering: Texture3d::new(device, SCATTERING_SIZE, "bruneton.delta_scattering"),
            scattering_density: Texture3d::new(
                device,
                SCATTERING_SIZE,
                "bruneton.scattering_density",
            ),
            sky_view: Texture2d::new(device, SKY_VIEW_SIZE, "bruneton.sky_view"),
            phase_lut: TextureArray::phase_lut(device, queue, group),
        }
    }
}

struct Texture2d {
    texture: wgpu::Texture,
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });
        Self { texture, view }
    }
}

struct Texture3d {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Texture3d {
    fn new(device: &wgpu::Device, size: UVec3, label: &'static str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: size.z.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LUT_FORMAT,
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
        Self { texture, view }
    }
}

struct TextureArray {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl TextureArray {
    fn phase_lut(device: &wgpu::Device, queue: &wgpu::Queue, group: SpectralGroup) -> Self {
        let width = crate::aerosol::PHASE_LUT_COS_BINS_U32;
        let layers = crate::aerosol::PHASE_LUT_SPECIES_U32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bruneton.aerosol_phase.lut"),
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

        for (z, species_lut) in (0..layers).zip(crate::aerosol::PHASE_LUTS[group.index()].iter()) {
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
            label: Some("bruneton.aerosol_phase.lut.view"),
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
    indirect_irradiance: wgpu::BindGroupLayout,
    scattering_density: wgpu::BindGroupLayout,
    multiple_scattering: wgpu::BindGroupLayout,
    sky_view: wgpu::BindGroupLayout,
    render: wgpu::BindGroupLayout,
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            transmittance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.transmittance.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(1),
                ],
            }),
            direct_irradiance: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.direct_irradiance.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(3),
                    storage_2d_entry(4),
                ],
            }),
            single_scattering: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.single_scattering.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(2, wgpu::ShaderStages::COMPUTE),
                    storage_3d_entry(3),
                    storage_3d_entry(4),
                    storage_3d_entry(5),
                    texture_2d_array_entry(6, wgpu::ShaderStages::COMPUTE),
                ],
            }),
            indirect_irradiance: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("bruneton.indirect_irradiance.bgl"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        uniform_entry(1, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(2, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(3, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(4, wgpu::ShaderStages::COMPUTE),
                        texture_2d_entry(5, wgpu::ShaderStages::COMPUTE),
                        sampler_entry(6, wgpu::ShaderStages::COMPUTE),
                        storage_2d_entry(7),
                        storage_2d_entry(8),
                        texture_2d_array_entry(9, wgpu::ShaderStages::COMPUTE),
                    ],
                },
            ),
            scattering_density: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.scattering_density.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    uniform_entry(1, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(2, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(3, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(4, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(5, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(6, wgpu::ShaderStages::COMPUTE),
                    texture_2d_array_entry(7, wgpu::ShaderStages::COMPUTE),
                    storage_3d_entry(8),
                ],
            }),
            multiple_scattering: device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("bruneton.multiple_scattering.bgl"),
                    entries: &[
                        uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                        texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(2, wgpu::ShaderStages::COMPUTE),
                        texture_3d_entry(3, wgpu::ShaderStages::COMPUTE),
                        sampler_entry(4, wgpu::ShaderStages::COMPUTE),
                        storage_3d_entry(5),
                        storage_3d_entry(6),
                    ],
                },
            ),
            sky_view: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.sky_view.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
                    texture_2d_entry(2, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(3, wgpu::ShaderStages::COMPUTE),
                    texture_3d_entry(4, wgpu::ShaderStages::COMPUTE),
                    sampler_entry(5, wgpu::ShaderStages::COMPUTE),
                    texture_2d_array_entry(6, wgpu::ShaderStages::COMPUTE),
                    storage_2d_entry(7),
                ],
            }),
            render: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bruneton.render.bgl"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(3, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(4, wgpu::ShaderStages::FRAGMENT),
                    sampler_entry(5, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(6, wgpu::ShaderStages::FRAGMENT),
                    texture_2d_entry(7, wgpu::ShaderStages::FRAGMENT),
                ],
            }),
        }
    }
}

struct Pipelines {
    transmittance: wgpu::ComputePipeline,
    direct_irradiance: wgpu::ComputePipeline,
    single_scattering: wgpu::ComputePipeline,
    indirect_irradiance: wgpu::ComputePipeline,
    scattering_density: wgpu::ComputePipeline,
    multiple_scattering: wgpu::ComputePipeline,
    sky_view: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device, layouts: &Layouts) -> Self {
        Self {
            transmittance: compute_pipeline(
                device,
                &layouts.transmittance,
                "bruneton.transmittance.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
                    include_str!("wgsl/transmittance.comp.wgsl")
                ),
            ),
            direct_irradiance: compute_pipeline(
                device,
                &layouts.direct_irradiance,
                "bruneton.direct_irradiance.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
                    include_str!("wgsl/direct_irradiance.comp.wgsl")
                ),
            ),
            single_scattering: compute_pipeline(
                device,
                &layouts.single_scattering,
                "bruneton.single_scattering.pipeline",
                &format!(
                    "{}\n\n{}\n\n{}",
                    crate::COMMON_WGSL,
                    crate::INSCATTER_WGSL,
                    include_str!("wgsl/single_scattering.comp.wgsl")
                ),
            ),
            indirect_irradiance: compute_pipeline(
                device,
                &layouts.indirect_irradiance,
                "bruneton.indirect_irradiance.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
                    include_str!("wgsl/indirect_irradiance.comp.wgsl")
                ),
            ),
            scattering_density: compute_pipeline(
                device,
                &layouts.scattering_density,
                "bruneton.scattering_density.pipeline",
                &format!(
                    "{}\n\n{}\n\n{}",
                    crate::COMMON_WGSL,
                    crate::INSCATTER_WGSL,
                    include_str!("wgsl/scattering_density.comp.wgsl")
                ),
            ),
            multiple_scattering: compute_pipeline(
                device,
                &layouts.multiple_scattering,
                "bruneton.multiple_scattering.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
                    include_str!("wgsl/multiple_scattering_4d.comp.wgsl")
                ),
            ),
            sky_view: compute_pipeline(
                device,
                &layouts.sky_view,
                "bruneton.sky_view.pipeline",
                &format!(
                    "{}\n\n{}",
                    crate::COMMON_WGSL,
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
        SUN_WGSL,
        include_str!("wgsl/render_sky_bruneton.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bruneton.render.pipeline"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bruneton.render.pipeline_layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("bruneton.render.pipeline"),
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

fn bruneton_params(params: &BrunetonFrameParams, group: SpectralGroup) -> HillaireParamsGpu {
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
    p.sun_spectral_irradiance = SUN_SPECTRAL_IRRADIANCE[group.index()];
    p.molecular_scattering_base = MOLECULAR_SCATTERING_BASE[group.index()];
    p.ozone_absorption_cross_section = OZONE_ABSORPTION_CROSS_SECTION[group.index()];

    for (slot, ((base, bg, scale), (sigma_sca, sigma_abs))) in p.species.iter_mut().zip(
        species_defaults.iter().zip(
            crate::aerosol::SIGMA_SCA[group.index()]
                .iter()
                .zip(crate::aerosol::SIGMA_ABS[group.index()].iter()),
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

fn dispatch_compute_3d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    size: UVec3,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(size.x.div_ceil(4), size.y.div_ceil(4), size.z.div_ceil(4));
}

fn copy_texture_2d(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    size: UVec2,
) {
    encoder.copy_texture_to_texture(
        texture_copy_view(src),
        texture_copy_view(dst),
        wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}

fn copy_texture_3d(
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Texture,
    dst: &wgpu::Texture,
    size: UVec3,
) {
    encoder.copy_texture_to_texture(
        texture_copy_view(src),
        texture_copy_view(dst),
        wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: size.z,
        },
    );
}

const fn texture_copy_view(texture: &wgpu::Texture) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
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
        label: Some("bruneton.transmittance.bg"),
        layout,
        entries: &[buffer_binding(0, params), texture_binding(1, out)],
    })
}

struct DirectIrradianceBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    irradiance: &'a wgpu::TextureView,
    delta_irradiance: &'a wgpu::TextureView,
}

fn direct_irradiance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: DirectIrradianceBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.direct_irradiance.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            texture_binding(3, input.irradiance),
            texture_binding(4, input.delta_irradiance),
        ],
    })
}

struct SingleScatteringBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    scattering: &'a wgpu::TextureView,
    single_mie: &'a wgpu::TextureView,
    single_rayleigh: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
}

fn single_scattering_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: SingleScatteringBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.single_scattering.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            texture_binding(3, input.scattering),
            texture_binding(4, input.single_mie),
            texture_binding(5, input.single_rayleigh),
            texture_binding(6, input.phase_lut),
        ],
    })
}

struct IndirectIrradianceBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    order: &'a wgpu::Buffer,
    scattering: &'a wgpu::TextureView,
    single_mie: &'a wgpu::TextureView,
    delta_scattering: &'a wgpu::TextureView,
    irradiance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    irradiance_out: &'a wgpu::TextureView,
    delta_irradiance_out: &'a wgpu::TextureView,
    phase_lut: &'a wgpu::TextureView,
}

fn indirect_irradiance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: IndirectIrradianceBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.indirect_irradiance.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.order),
            texture_binding(2, input.scattering),
            texture_binding(3, input.single_mie),
            texture_binding(4, input.delta_scattering),
            texture_binding(5, input.irradiance),
            sampler_binding(6, input.sampler),
            texture_binding(7, input.irradiance_out),
            texture_binding(8, input.delta_irradiance_out),
            texture_binding(9, input.phase_lut),
        ],
    })
}

struct ScatteringDensityBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    order: &'a wgpu::Buffer,
    irradiance: &'a wgpu::TextureView,
    scattering: &'a wgpu::TextureView,
    single_mie: &'a wgpu::TextureView,
    delta_scattering: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    phase_lut: &'a wgpu::TextureView,
    density: &'a wgpu::TextureView,
}

fn scattering_density_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: ScatteringDensityBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.scattering_density.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.order),
            texture_binding(2, input.irradiance),
            texture_binding(3, input.scattering),
            texture_binding(4, input.single_mie),
            texture_binding(5, input.delta_scattering),
            sampler_binding(6, input.sampler),
            texture_binding(7, input.phase_lut),
            texture_binding(8, input.density),
        ],
    })
}

struct MultipleScatteringBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    density: &'a wgpu::TextureView,
    scattering: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    delta_scattering: &'a wgpu::TextureView,
    scattering_accum: &'a wgpu::TextureView,
}

fn multiple_scattering_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: MultipleScatteringBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.multiple_scattering.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            texture_binding(2, input.density),
            texture_binding(3, input.scattering),
            sampler_binding(4, input.sampler),
            texture_binding(5, input.delta_scattering),
            texture_binding(6, input.scattering_accum),
        ],
    })
}

struct SkyViewBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    irradiance: &'a wgpu::TextureView,
    scattering: &'a wgpu::TextureView,
    single_rayleigh: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    phase_lut: &'a wgpu::TextureView,
    sky_view: &'a wgpu::TextureView,
}

fn sky_view_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: SkyViewBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.sky_view.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            texture_binding(2, input.irradiance),
            texture_binding(3, input.scattering),
            texture_binding(4, input.single_rayleigh),
            sampler_binding(5, input.sampler),
            texture_binding(6, input.phase_lut),
            texture_binding(7, input.sky_view),
        ],
    })
}

struct RenderBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    view: &'a wgpu::Buffer,
    sun: &'a wgpu::Buffer,
    transmittance_low: &'a wgpu::TextureView,
    transmittance_high: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    sky_view_low: &'a wgpu::TextureView,
    sky_view_high: &'a wgpu::TextureView,
}

fn render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: RenderBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bruneton.render.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            buffer_binding(1, input.view),
            buffer_binding(2, input.sun),
            texture_binding(3, input.transmittance_low),
            texture_binding(4, input.transmittance_high),
            sampler_binding(5, input.sampler),
            texture_binding(6, input.sky_view_low),
            texture_binding(7, input.sky_view_high),
        ],
    })
}

#[derive(Debug)]
pub enum BrunetonRendererError {
    ResourceSizeOverflow,
}

impl fmt::Display for BrunetonRendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceSizeOverflow => f.write_str("bruneton atmosphere resource size overflow"),
        }
    }
}

impl StdError for BrunetonRendererError {}

const _: () = assert!(core::mem::size_of::<RuntimeViewGpu>() == 80);
const _: () = assert!(core::mem::size_of::<ScatteringOrderGpu>() == 16);
const _: () = assert!(core::mem::size_of::<HillaireParamsGpu>() == 256);
const _: () = assert!(AEROSOL_SPECIES == 3);
