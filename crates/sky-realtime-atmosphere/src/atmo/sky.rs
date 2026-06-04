//! runtime `SkyView` 采样契约。
//!
//! `SkyView` 是单主摄像机性能近似，不是无偏物理天空采样。provider 必须发布
//! 与 LUT 匹配的参数；消费者只按这里的 source 契约读取。

use bytemuck::{Pod, Zeroable};
use glam::UVec2;

use crate::atmo::Sun;

/// PT 采样 `SkyView` 所需的 WGSL helper。
pub const SKY_VIEW_WGSL: &str = include_str!("../wgsl/sky_view.wgsl");

/// `SkyView` 采样参数 GPU 布局。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkyViewParamsGpu {
    pub earth_radius_km: f32,
    pub atmosphere_thickness_km: f32,
    pub eye_distance_to_earth_center_km: f32,
    pub eye_altitude_km: f32,
    pub sun_dir: [f32; 3],
    pub sky_view_height_km: f32,
}

impl SkyViewParamsGpu {
    #[must_use]
    pub const fn new(
        earth_radius_km: f32,
        atmosphere_thickness_km: f32,
        eye_distance_to_earth_center_km: f32,
        eye_altitude_km: f32,
        sun_dir: [f32; 3],
        sky_view_height_km: f32,
    ) -> Self {
        Self {
            earth_radius_km,
            atmosphere_thickness_km,
            eye_distance_to_earth_center_km,
            eye_altitude_km,
            sun_dir,
            sky_view_height_km,
        }
    }

    #[must_use]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }
}

/// 本帧可采样的 `SkyView` source。
#[derive(Clone, Copy, Debug)]
pub struct SkyViewSource<'a> {
    pub params_buffer: &'a wgpu::Buffer,
    pub sky_view: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub revision: u64,
    pub size: UVec2,
    pub sun: Sun,
}

impl<'a> SkyViewSource<'a> {
    #[must_use]
    pub const fn new(
        params_buffer: &'a wgpu::Buffer,
        sky_view: &'a wgpu::TextureView,
        sampler: &'a wgpu::Sampler,
        revision: u64,
        size: UVec2,
        sun: Sun,
    ) -> Self {
        Self {
            params_buffer,
            sky_view,
            sampler,
            revision,
            size,
            sun,
        }
    }
}

const _: () = assert!(core::mem::size_of::<SkyViewParamsGpu>() == 32);
