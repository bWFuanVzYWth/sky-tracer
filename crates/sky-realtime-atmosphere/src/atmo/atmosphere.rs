//! 大气与体素渲染后端之间的共享 GPU 契约。
//!
//! raster 与 PT 后端只消费这里定义的体素大气契约；具体大气实现负责生产
//! 该 uniform、transmittance LUT、aerial perspective LUT 与 bind group。
//! layout 在这里定义为单一来源，避免后端通过约定复制。

use bytemuck::{Pod, Zeroable};

/// 体素后端使用的大气 bind group 索引。
pub const VOXEL_ATMOSPHERE_LIGHTING_BIND_GROUP_INDEX: u32 = 2;

/// 体素大气光照 WGSL 契约。
pub const VOXEL_ATMOSPHERE_LIGHTING_WGSL: &str = include_str!("../wgsl/atmosphere_lighting.wgsl");

/// 体素 aerial perspective WGSL helper。
pub const VOXEL_AERIAL_PERSPECTIVE_WGSL: &str = include_str!("../wgsl/aerial_perspective.wgsl");

/// 体素直射光所需的大气参数。
///
/// 该类型不表达具体大气算法，只要求生产者提供与绑定的 transmittance LUT
/// 一致的几何和太阳光谱。LUT 坐标参数化由 WGSL helper 定义。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VoxelAtmosphereLightingGpu {
    pub planet_radius_km: f32,
    pub atmosphere_thickness_km: f32,
    pub eye_distance_to_planet_center_km: f32,
    pub pad0: f32,
    pub sun_dir: [f32; 3],
    pub pad1: f32,
    pub sun_spectral_irradiance: [f32; 4],
}

impl VoxelAtmosphereLightingGpu {
    #[must_use]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }
}

/// 创建体素大气光照 bind group layout。
#[must_use]
pub fn voxel_atmosphere_lighting_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ca_render.voxel_atmosphere_lighting.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

const _: () = assert!(core::mem::size_of::<VoxelAtmosphereLightingGpu>() == 48);
