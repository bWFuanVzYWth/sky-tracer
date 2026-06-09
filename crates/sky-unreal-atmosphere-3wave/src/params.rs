//! Hillaire 大气 GPU 参数与气溶胶 preset。
//!
//! 气溶胶只保留当前项目已有的三种主导粒子：`waso`、`inso`、`soot`。缺失的
//! 海盐、矿尘与硫酸液滴不做占位，避免假精度。

use bytemuck::{Pod, Zeroable};
use glam::UVec3;

pub const AEROSOL_SPECIES: usize = 3;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AerosolPreset {
    RemoteContinental,
    #[default]
    Rural,
    ContinentalPolluted,
    Urban,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HillairePhaseMode {
    #[default]
    Lut,
    CornetteShanks,
}

impl HillairePhaseMode {
    #[must_use]
    pub const fn as_gpu_u32(self) -> u32 {
        match self {
            Self::Lut => 0,
            Self::CornetteShanks => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HillaireSettings {
    pub month: u32,
    pub aerosol_turbidity: f32,
    pub ground_albedo_spectral: [f32; 4],
}

impl Default for HillaireSettings {
    fn default() -> Self {
        Self {
            month: 2,
            aerosol_turbidity: 1.0,
            ground_albedo_spectral: [0.18, 0.18, 0.18, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HillaireSpeciesGpu {
    pub sigma_sca: [f32; 4],
    pub sigma_abs: [f32; 4],
    pub base_density: f32,
    pub bg_density: f32,
    pub height_scale: f32,
    pub pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HillaireParamsGpu {
    pub earth_radius_km: f32,
    pub atmosphere_thickness_km: f32,
    pub eye_distance_to_earth_center_km: f32,
    pub eye_altitude_km: f32,
    pub sun_dir: [f32; 3],
    pub sky_view_height_km: f32,
    pub sun_spectral_irradiance: [f32; 4],
    pub molecular_scattering_base: [f32; 4],
    pub ozone_absorption_cross_section: [f32; 4],
    pub species: [HillaireSpeciesGpu; AEROSOL_SPECIES],
    pub turbidity: f32,
    pub ozone_mean_dobson: f32,
    pub mie_phase_mode: u32,
    pub pad_misc1: u32,
    pub ground_albedo_spectral: [f32; 4],
}

impl HillaireParamsGpu {
    #[must_use]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }
}

const _: () = assert!(core::mem::size_of::<HillaireSpeciesGpu>() == 48);
const _: () = assert!(core::mem::size_of::<HillaireParamsGpu>() == 256);
pub type HillaireSpeciesDefaults = (f32, f32, f32);

const FT_INSO: f32 = 2.291e-6;
const FT_WASO: f32 = 4.571e-7;
const FT_SOOT: f32 = 1.362e-8;

#[must_use]
pub const fn aerosol_preset_defaults(preset: AerosolPreset) -> [HillaireSpeciesDefaults; 3] {
    match preset {
        AerosolPreset::RemoteContinental => [
            (5.53e-6, FT_WASO, 8.0),
            (3.83e-6, FT_INSO, 8.0),
            (0.0, FT_SOOT, 8.0),
        ],
        AerosolPreset::Rural => [
            (1.49e-5, FT_WASO, 8.0),
            (1.01e-5, FT_INSO, 8.0),
            (5.31e-7, FT_SOOT, 8.0),
        ],
        AerosolPreset::ContinentalPolluted => [
            (3.34e-5, FT_WASO, 8.0),
            (1.51e-5, FT_INSO, 8.0),
            (2.23e-6, FT_SOOT, 8.0),
        ],
        AerosolPreset::Urban => [
            (5.95e-5, FT_WASO, 8.0),
            (3.78e-5, FT_INSO, 8.0),
            (8.29e-6, FT_SOOT, 8.0),
        ],
    }
}

#[must_use]
pub const fn ozone_monthly_dobson(month: u32) -> f32 {
    match month % 12 {
        0 => 347.0,
        1 => 370.0,
        2 => 381.0,
        3 => 384.0,
        4 => 372.0,
        5 => 352.0,
        6 => 333.0,
        7 => 317.0,
        8 => 298.0,
        9 => 285.0,
        10 => 290.0,
        _ => 315.0,
    }
}

/// Wavelength order used by every 4-channel spectral GPU constant.
///
/// RGB store the optimized 480/570/660 nm samples; A is intentionally unused
/// and kept at zero so the copied renderer can preserve the existing RGBA ABI.
pub const SPECTRAL_SAMPLE_WAVELENGTHS_NM: [f32; 4] = [480.0, 570.0, 660.0, 0.0];
pub const SUN_SPECTRAL_IRRADIANCE: [f32; 4] = [2.056_643_2, 1.859_502_1, 1.542_136_2, 0.0];
pub const MOLECULAR_SCATTERING_BASE: [f32; 4] = [2.545_312_8e-2, 1.279_990_3e-2, 7.120_826e-3, 0.0];
pub const OZONE_ABSORPTION_CROSS_SECTION: [f32; 4] = [7.11e-26, 4.67e-25, 2.02e-25, 0.0];

const AP_SLICE_COUNT: u32 = 32;
pub const AP_LUT_DIM: UVec3 = UVec3::new(AP_SLICE_COUNT, AP_SLICE_COUNT, AP_SLICE_COUNT);
