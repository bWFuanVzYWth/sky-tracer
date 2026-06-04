use serde::{Deserialize, Serialize};

pub const SPECTRAL_PANORAMA_KIND: &str = "spectral_panorama_v0";

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
    pub files: SpectralAssetFiles,
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
            files,
        }
    }
}
