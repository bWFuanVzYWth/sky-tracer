//! 渲染系统共享太阳光源。
//!
//! 移植自旧 `ca_core::atmo` 与 `ca_render::atmo::sun`，用于保持 Hillaire
//! renderer 的 GPU 布局和 WGSL 契约不变。

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// 大气顶外太阳 irradiance，scene-linear Rec.2020，W/m^2。
pub const SUN_IRRADIANCE_REC2020_W_PER_M2: [f32; 3] = [205.0, 205.0, 205.0];

/// 太阳圆盘 WGSL helper。
pub const SUN_WGSL: &str = include_str!("../wgsl/sun.wgsl");

/// 共享太阳光源。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// normalized；太阳到场景的入射方向。
    pub sun_to_scene: Vec3,
    /// 大气顶外太阳 irradiance，scene-linear Rec.2020，W/m^2。
    pub irradiance_rec2020_w_m2: Vec3,
    /// 太阳视半径，单位 rad。
    pub angular_radius_rad: f32,
}

impl Default for Sun {
    fn default() -> Self {
        Self::earth_noon()
    }
}

impl Sun {
    #[must_use]
    pub const fn earth_noon() -> Self {
        Self {
            sun_to_scene: Vec3::new(-0.431_934, -0.863_868, -0.259_161),
            irradiance_rec2020_w_m2: Vec3::new(205.0, 205.0, 205.0),
            angular_radius_rad: 0.004_71,
        }
    }

    #[must_use]
    pub fn to_sun(self) -> Vec3 {
        -self.sun_to_scene
    }

    #[must_use]
    pub fn cos_angular_radius(self) -> f32 {
        self.angular_radius_rad.cos()
    }
}

/// GPU 太阳光源布局。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SunGpu {
    pub sun_to_scene: [f32; 3],
    pub angular_radius_rad: f32,
    pub irradiance_rec2020_w_m2: [f32; 3],
    pub cos_angular_radius: f32,
}

impl SunGpu {
    #[must_use]
    pub fn from_sun(sun: Sun) -> Self {
        Self {
            sun_to_scene: sun.sun_to_scene.to_array(),
            angular_radius_rad: sun.angular_radius_rad,
            irradiance_rec2020_w_m2: sun.irradiance_rec2020_w_m2.to_array(),
            cos_angular_radius: sun.cos_angular_radius(),
        }
    }
}

const _: () = assert!(core::mem::size_of::<SunGpu>() == 32);
