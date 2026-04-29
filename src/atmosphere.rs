use crate::geometry::Planet;
use crate::math::{TAU, Vec3};
use crate::phase::MiePhaseTable;
use crate::spectrum::SpectralBand;

pub const SPECIES_COUNT: usize = 4;
pub const PHASE_BINS: usize = 1024;
pub const SPECIES_NAMES: [&str; SPECIES_COUNT] = ["inso", "waso", "soot", "suso"];
pub const GROUND_ALBEDO: f32 = 0.3;
pub const SOLAR_ANGULAR_RADIUS_RAD: f32 = 0.004_650_47;

#[derive(Clone, Copy, Debug)]
pub struct Sun {
    pub direction: Vec3,
    pub angular_radius_rad: f32,
    pub solid_angle_sr: f32,
}

impl Sun {
    pub fn from_degrees(elevation_deg: f32, azimuth_deg: f32) -> Self {
        let elevation = elevation_deg.to_radians();
        let azimuth = azimuth_deg.to_radians();
        let cos_e = elevation.cos();
        let direction = Vec3::new(
            cos_e * azimuth.sin(),
            elevation.sin(),
            cos_e * azimuth.cos(),
        )
        .normalized();
        let solid_angle_sr = TAU * (1.0 - SOLAR_ANGULAR_RADIUS_RAD.cos());
        Self {
            direction,
            angular_radius_rad: SOLAR_ANGULAR_RADIUS_RAD,
            solid_angle_sr,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AtmosphericProfilePoint {
    pub altitude_km: f32,
    pub temperature_k: f32,
    pub air_cm3: f32,
    pub ozone_cm3: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct AerosolProfilePoint {
    pub altitude_km: f32,
    pub mass_g_m3: [f32; SPECIES_COUNT],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AerosolOptics {
    pub scattering_km_inv_per_g_m3: f32,
    pub absorption_km_inv_per_g_m3: f32,
}

#[derive(Clone, Debug)]
pub struct SceneData {
    pub planet: Planet,
    pub sun: Sun,
    pub bands: Vec<SpectralBand>,
    pub atmospheric_profile: Vec<AtmosphericProfilePoint>,
    pub aerosol_profile: Vec<AerosolProfilePoint>,
    pub aerosol_optics: Vec<[AerosolOptics; SPECIES_COUNT]>,
    pub phase_table: MiePhaseTable,
    pub majorants_km_inv: Vec<f32>,
}

impl SceneData {
    pub fn solar_radiance_w_m2_sr(&self, band_index: usize) -> f32 {
        self.bands[band_index].solar_irradiance_w_m2 / self.sun.solid_angle_sr
    }
}
