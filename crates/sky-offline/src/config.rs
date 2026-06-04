use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub spp: usize,
    pub seed: u64,
    pub out_dir: PathBuf,
    pub data_dir: PathBuf,
    pub sun_elevation_deg: f32,
    pub sun_azimuth_deg: f32,
    pub observer_altitude_km: f32,
    pub direct_light_samples: usize,
    pub png_exposure: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 512,
            spp: 256,
            seed: 0x5EC7_2026_0430,
            out_dir: PathBuf::from("out"),
            data_dir: PathBuf::from("data"),
            sun_elevation_deg: 0.0,
            sun_azimuth_deg: 0.0,
            observer_altitude_km: 0.2,
            direct_light_samples: 1,
            png_exposure: 0.01,
        }
    }
}
