use crate::geometry::Planet;
use crate::math::{TAU, Vec3};
use crate::phase::MiePhaseTable;
use crate::spectrum::SpectralBand;

pub const SPECIES_COUNT: usize = 4;
pub const PHASE_BINS: usize = 1024;
pub const SPECIES_NAMES: [&str; SPECIES_COUNT] = ["inso", "waso", "soot", "suso"];
pub const GROUND_ALBEDO: f32 = 0.18;
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
pub struct MajorantGrid {
    pub top_altitude_km: f32,
    pub layer_count: usize,
    pub band_count: usize,
    pub values_km_inv: Vec<f32>,
    pub minorants_km_inv: Vec<f32>,
}

impl MajorantGrid {
    pub fn new(top_altitude_km: f32, layer_count: usize, band_count: usize) -> Self {
        Self {
            top_altitude_km,
            layer_count,
            band_count,
            values_km_inv: vec![0.0; layer_count * band_count],
            minorants_km_inv: vec![0.0; layer_count * band_count],
        }
    }

    pub fn layer_thickness_km(&self) -> f32 {
        self.top_altitude_km / self.layer_count as f32
    }

    pub fn layer_for_altitude(&self, altitude_km: f32) -> usize {
        ((altitude_km.clamp(0.0, self.top_altitude_km - f32::EPSILON) / self.top_altitude_km)
            * self.layer_count as f32)
            .floor()
            .clamp(0.0, (self.layer_count - 1) as f32) as usize
    }

    pub fn layer_bounds_km(&self, layer: usize) -> (f32, f32) {
        let dz = self.layer_thickness_km();
        (layer as f32 * dz, (layer + 1) as f32 * dz)
    }

    pub fn get(&self, band_index: usize, layer: usize) -> f32 {
        self.values_km_inv[band_index * self.layer_count + layer]
    }

    pub fn minorant(&self, band_index: usize, layer: usize) -> f32 {
        self.minorants_km_inv[band_index * self.layer_count + layer]
    }

    pub fn set(&mut self, band_index: usize, layer: usize, value: f32) {
        self.values_km_inv[band_index * self.layer_count + layer] = value;
    }

    pub fn set_bounds(&mut self, band_index: usize, layer: usize, minorant: f32, majorant: f32) {
        let idx = band_index * self.layer_count + layer;
        self.minorants_km_inv[idx] = minorant.clamp(0.0, majorant.max(0.0));
        self.values_km_inv[idx] = majorant.max(self.minorants_km_inv[idx]);
    }

    pub fn global_for_band(&self, band_index: usize) -> f32 {
        (0..self.layer_count)
            .map(|layer| self.get(band_index, layer))
            .fold(0.0, f32::max)
    }
}

#[derive(Clone, Debug)]
pub struct SceneData {
    pub planet: Planet,
    pub sun: Sun,
    pub bands: Vec<SpectralBand>,
    pub rayleigh_cross_sections_m2: Vec<f32>,
    pub solar_radiance_w_m2_sr: Vec<f32>,
    pub atmospheric_profile: Vec<AtmosphericProfilePoint>,
    pub aerosol_profile: Vec<AerosolProfilePoint>,
    pub aerosol_optics: Vec<[AerosolOptics; SPECIES_COUNT]>,
    pub phase_table: MiePhaseTable,
    pub majorants_km_inv: Vec<f32>,
    pub majorant_grid: MajorantGrid,
}

impl SceneData {
    pub fn solar_radiance_w_m2_sr(&self, band_index: usize) -> f32 {
        self.solar_radiance_w_m2_sr[band_index]
    }
}
