use serde::{Deserialize, Serialize};

pub const SPECTRAL_PANORAMA_KIND: &str = "spectral_panorama_v0";
pub const SPECTRAL_SKY_VIEW_LUT_KIND: &str = "spectral_sky_view_lut_v0";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpectralAssetManifest {
    pub version: u32,
    pub kind: String,
    pub dimensions: [usize; 2],
    pub spp: usize,
    pub seed: u64,
    pub sun_elevation_deg: f32,
    pub sun_azimuth_deg: f32,
    pub observer_altitude_km: f32,
    pub band_centers_nm: Vec<f32>,
    pub data_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colorimetry: Option<SpectralAssetColorimetry>,
    pub files: SpectralAssetFiles,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpectralAssetColorimetry {
    pub cmf: String,
    pub cmf_source: String,
    pub rgb_color_space: String,
    pub white_balance: SpectralAssetWhiteBalance,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpectralAssetWhiteBalance {
    pub method: String,
    pub source_white_xyz_y1: [f32; 3],
    pub target_white_xyz_y1: [f32; 3],
    pub xyz_from_xyz: [[f32; 3]; 3],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpectralAssetFiles {
    pub rgb_exr: String,
    pub rgb_png: String,
    pub band_exrs: Vec<String>,
}

impl SpectralAssetManifest {
    pub fn spectral_panorama(
        dimensions: [usize; 2],
        spp: usize,
        seed: u64,
        sun_elevation_deg: f32,
        sun_azimuth_deg: f32,
        observer_altitude_km: f32,
        band_centers_nm: Vec<f32>,
        files: SpectralAssetFiles,
    ) -> Self {
        Self {
            version: 0,
            kind: SPECTRAL_PANORAMA_KIND.to_owned(),
            dimensions,
            spp,
            seed,
            sun_elevation_deg,
            sun_azimuth_deg,
            observer_altitude_km,
            band_centers_nm,
            data_hash: None,
            colorimetry: None,
            files,
        }
    }

    pub fn spectral_sky_view_lut(
        dimensions: [usize; 2],
        spp: usize,
        seed: u64,
        sun_elevation_deg: f32,
        sun_azimuth_deg: f32,
        observer_altitude_km: f32,
        band_centers_nm: Vec<f32>,
        files: SpectralAssetFiles,
    ) -> Self {
        Self {
            version: 0,
            kind: SPECTRAL_SKY_VIEW_LUT_KIND.to_owned(),
            dimensions,
            spp,
            seed,
            sun_elevation_deg,
            sun_azimuth_deg,
            observer_altitude_km,
            band_centers_nm,
            data_hash: None,
            colorimetry: None,
            files,
        }
    }
}
