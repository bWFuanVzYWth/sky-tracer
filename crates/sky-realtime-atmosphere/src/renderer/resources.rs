use bytemuck::{Pod, Zeroable};
use ca_render::atmo::{SkyViewParamsGpu, Sun, SunGpu, VoxelAtmosphereLightingGpu};
use ca_render::gpu::ViewFrame;
use glam::{UVec2, UVec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::params::AP_LUT_DIM;

use super::HillaireRendererError;

const DEFAULT_TRANSMITTANCE_LUT_SIZE: UVec2 = UVec2::new(256, 64);
const DEFAULT_SKY_VIEW_LUT_SIZE: UVec2 = UVec2::new(256, 512);
pub(super) const HILLAIRE_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(super) struct RendererResources {
    pub params_buffer: wgpu::Buffer,
    pub sky_view_params_buffer: wgpu::Buffer,
    pub view_buffer: wgpu::Buffer,
    pub voxel_lighting_buffer: wgpu::Buffer,
    pub sun_buffer: wgpu::Buffer,
    pub sampler: wgpu::Sampler,
    pub transmittance_lut: Texture2d,
    pub sky_view_lut: Texture2d,
    pub ap_lut: AerialPerspectiveLut,
    pub aerosol_phase_lut: TextureArray,
}

impl RendererResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, HillaireRendererError> {
        let params_buffer = uniform_buffer(
            device,
            "hillaire.params.uniform",
            crate::params::HillaireParamsGpu::zeroed(),
        );
        let sky_view_params_buffer = uniform_buffer(
            device,
            "hillaire.sky_view_params.uniform",
            SkyViewParamsGpu::zeroed(),
        );
        let view_buffer = uniform_buffer(device, "hillaire.view.uniform", RuntimeViewGpu::zeroed());
        let voxel_lighting_buffer = uniform_buffer(
            device,
            "hillaire.voxel_lighting.uniform",
            VoxelAtmosphereLightingGpu::zeroed(),
        );
        let sun_buffer = uniform_buffer(
            device,
            "hillaire.sun.uniform",
            SunGpu::from_sun(Sun::default()),
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hillaire.lut.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let transmittance_lut = Texture2d::storage(
            device,
            DEFAULT_TRANSMITTANCE_LUT_SIZE,
            "hillaire.transmittance.lut",
        );
        let sky_view_lut =
            Texture2d::storage(device, DEFAULT_SKY_VIEW_LUT_SIZE, "hillaire.sky_view.lut");
        let ap_lut = AerialPerspectiveLut::new(device, queue, AP_LUT_DIM)?;
        let aerosol_phase_lut = TextureArray::aerosol_phase(device, queue);
        Ok(Self {
            params_buffer,
            sky_view_params_buffer,
            view_buffer,
            voxel_lighting_buffer,
            sun_buffer,
            sampler,
            transmittance_lut,
            sky_view_lut,
            ap_lut,
            aerosol_phase_lut,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RuntimeViewGpu {
    // sky/AP 只需要相机相对反投影；绝对位置只用于大气半径和高度参数。
    relative_world_from_clip: [[f32; 4]; 4],
    world_position: [f32; 4],
}

impl RuntimeViewGpu {
    #[must_use]
    pub const fn from_view(view: &ViewFrame) -> Self {
        Self {
            relative_world_from_clip: view.relative_world_from_clip,
            world_position: view.world_position,
        }
    }
}

pub(super) struct Texture2d {
    _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: UVec2,
}

impl Texture2d {
    fn storage(device: &wgpu::Device, size: UVec2, label: &'static str) -> Self {
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
            format: HILLAIRE_LUT_FORMAT,
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
            size,
        }
    }
}

pub(super) struct AerialPerspectiveLut {
    _inscatter_texture: wgpu::Texture,
    pub inscatter_view: wgpu::TextureView,
    _transmittance_texture: wgpu::Texture,
    pub transmittance_view: wgpu::TextureView,
    pub size: UVec3,
}

impl AerialPerspectiveLut {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: UVec3,
    ) -> Result<Self, HillaireRendererError> {
        let inscatter_texture =
            ap_texture(device, size, "hillaire.aerial_perspective.inscatter.lut");
        let inscatter_view = inscatter_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hillaire.aerial_perspective.inscatter.view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        initialize_ap_texture(queue, &inscatter_texture, size, 0x0000)?;

        let transmittance_texture = ap_texture(
            device,
            size,
            "hillaire.aerial_perspective.transmittance.lut",
        );
        let transmittance_view = transmittance_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hillaire.aerial_perspective.transmittance.view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        initialize_ap_texture(queue, &transmittance_texture, size, 0x3c00)?;

        Ok(Self {
            _inscatter_texture: inscatter_texture,
            inscatter_view,
            _transmittance_texture: transmittance_texture,
            transmittance_view,
            size,
        })
    }
}

pub(super) struct TextureArray {
    _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl TextureArray {
    fn aerosol_phase(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let width = crate::aerosol::PHASE_LUT_COS_BINS_U32;
        let layers = crate::aerosol::PHASE_LUT_SPECIES_U32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hillaire.aerosol_phase.lut"),
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

        for (z, species_lut) in
            (0..crate::aerosol::PHASE_LUT_SPECIES_U32).zip(crate::aerosol::PHASE_LUTS.iter())
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
            label: Some("hillaire.aerosol_phase.lut.view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            _texture: texture,
            view,
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

fn ap_texture(device: &wgpu::Device, size: UVec3, label: &'static str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: size.z.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: HILLAIRE_LUT_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn initialize_ap_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: UVec3,
    fill_half: u16,
) -> Result<(), HillaireRendererError> {
    let width = size.x.max(1);
    let height = size.y.max(1);
    let depth = size.z.max(1);
    let pixel_count = texel_lanes_4(width, height, depth)?;
    let data = vec![fill_half; pixel_count];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4 * 2),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
    Ok(())
}

fn texel_lanes_4(width: u32, height: u32, depth: u32) -> Result<usize, HillaireRendererError> {
    let width = usize::try_from(width).map_err(|_| HillaireRendererError::ResourceSizeOverflow)?;
    let height =
        usize::try_from(height).map_err(|_| HillaireRendererError::ResourceSizeOverflow)?;
    let depth = usize::try_from(depth).map_err(|_| HillaireRendererError::ResourceSizeOverflow)?;
    width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(depth))
        .and_then(|count| count.checked_mul(4))
        .ok_or(HillaireRendererError::ResourceSizeOverflow)
}

const _: () = assert!(core::mem::size_of::<RuntimeViewGpu>() == 80);
