use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    film::Film,
    transport::TransportSettings,
    volume::VolumeStats,
};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize)]
pub struct Record {
    pub source: PathBuf,
    pub source_bytes: u64,
    pub grid: String,
    pub volume_stats: VolumeStats,
    pub camera: Camera,
    pub transport: TransportSettings,
    pub render: RenderConfig,
    pub backend: String,
    pub adapter: Option<String>,
    pub asset_attribution: Option<String>,
}
pub fn save(directory: &Path, film: &Film, record: &Record, exposure: f32) -> Result<()> {
    film.validate()?;
    if !exposure.is_finite() || !(-30.0..=30.0).contains(&exposure) || directory.exists() {
        return Err("save requires exposure in [-30, 30] EV and a new directory".into());
    }
    fs::create_dir_all(directory)?;
    let width = film.width as usize;
    let height = film.height as usize;
    exr::prelude::write_rgb_file(directory.join("radiance.exr"), width, height, |x, y| {
        let v = film.mean[y * width + x];
        (v[0], v[1], v[2])
    })?;
    if film.samples_per_pixel > 1 {
        exr::prelude::write_rgb_file(
            directory.join("sample_variance.exr"),
            width,
            height,
            |x, y| {
                let v = film.sample_variance[y * width + x];
                (v[0], v[1], v[2])
            },
        )?;
        exr::prelude::write_rgb_file(
            directory.join("standard_error.exr"),
            width,
            height,
            |x, y| {
                let v = film.sample_variance[y * width + x]
                    .map(|s| (s / film.samples_per_pixel as f32).sqrt());
                (v[0], v[1], v[2])
            },
        )?;
    }
    let gain = 2.0f64.powf(f64::from(exposure));
    let mut preview = image::RgbImage::new(film.width, film.height);
    for (pixel, value) in preview.pixels_mut().zip(&film.mean) {
        *pixel = image::Rgb(value.map(|linear| {
            let v = f64::from(linear) * gain;
            let tone = v / (1.0 + v);
            let srgb = if tone <= 0.0031308 {
                12.92 * tone
            } else {
                1.055 * tone.powf(1.0 / 2.4) - 0.055
            };
            (srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        }));
    }
    preview.save(directory.join("preview.png"))?;
    let metadata = serde_json::json!({
        "kind":"cloud_path_trace_reference_v1","scene":record,
        "dimensions":[film.width,film.height],"samples_per_pixel":film.samples_per_pixel,
        "sample_variance_valid":film.samples_per_pixel>1,
        "complete_requested_samples":film.samples_per_pixel==record.render.spp,
        "color_space":"linear RGB (Disney example values); independent cloud model",
        "density_reconstruction":"selected original VDB lattice, trilinear; inactive values and uniform tiles retained",
        "estimator":"delta tracking, ratio-tracked delta sun NEE, ambient escape, HG, Lambert ground, unbiased roulette",
        "hard_scattering_depth_limit":null,"transmittance_threshold":null,
        "failed_paths_discarded":false,"diagnostics_passed":true,
        "shadow_roulette":{"enabled":record.transport.shadow_roulette,"candidate_interval":16,"weight_trigger":1e-4,"survival_probability":0.5,"survivor_weight_multiplier":2.0},
        "precision":if record.backend=="cpu-f64" {"f64 transport and Welford accumulation; EXR f32"}else{"f32 transport and Welford accumulation; compensated world slabs, entries and distances"},
        "preview":{"file":"preview.png","exposure_ev":exposure,"tone_map":"Reinhard","encoding":"sRGB"},
        "reference_file":"radiance.exr","variance_file":if film.samples_per_pixel>1 {Some("sample_variance.exr")}else{None},
        "standard_error_file":if film.samples_per_pixel>1 {Some("standard_error.exr")}else{None},
        "limits":"Mathematically unbiased for the declared density/material/light model; finite precision and pseudorandom sampling remain. Appearance agreement with Hyperion is a separate validation."
    });
    fs::write(
        directory.join("asset.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_rgb(path: &Path) -> Vec<[f32; 3]> {
        exr::prelude::read_first_rgba_layer_from_file(
            path,
            |size, _| (size.width(), vec![[0.0; 3]; size.width() * size.height()]),
            |pixels, position, (r, g, b, _): (f32, f32, f32, f32)| {
                pixels.1[position.y() * pixels.0 + position.x()] = [r, g, b];
            },
        )
        .unwrap()
        .layer_data
        .channel_data
        .pixels
        .1
    }

    #[test]
    fn exported_reference_and_uncertainty_remain_linear() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/cloud-output-tests");
        fs::create_dir_all(&root).unwrap();
        let directory = root.join(format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let film = Film {
            width: 2,
            height: 1,
            samples_per_pixel: 4,
            mean: vec![[0.25, 1.0, 4.0], [0.0, 0.5, 2.0]],
            sample_variance: vec![[0.04, 0.16, 0.64], [0.0, 0.25, 1.0]],
        };
        let record = Record {
            source: "example.vdb".into(),
            source_bytes: 1234,
            grid: "density".into(),
            volume_stats: VolumeStats::default(),
            camera: Camera::default(),
            transport: TransportSettings::default(),
            render: RenderConfig {
                width: 2,
                height: 1,
                spp: 8,
                seed: 1,
                ..RenderConfig::default()
            },
            backend: "cpu-f64".into(),
            adapter: None,
            asset_attribution: None,
        };
        save(&directory, &film, &record, 10.0).unwrap();
        assert_eq!(read_rgb(&directory.join("radiance.exr")), film.mean);
        assert_eq!(
            read_rgb(&directory.join("sample_variance.exr")),
            film.sample_variance
        );
        let stderr = read_rgb(&directory.join("standard_error.exr"));
        for (actual, variance) in stderr.iter().zip(&film.sample_variance) {
            for c in 0..3 {
                assert_eq!(actual[c], (variance[c] / 4.0).sqrt());
            }
        }
        let metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("asset.json")).unwrap()).unwrap();
        assert_eq!(metadata["samples_per_pixel"], 4);
        assert_eq!(metadata["complete_requested_samples"], false);
        assert_eq!(metadata["sample_variance_valid"], true);
        assert!(save(&directory, &film, &record, 0.0).is_err());
        let resolved = directory.canonicalize().unwrap();
        assert!(resolved.starts_with(root.canonicalize().unwrap()));
        fs::remove_dir_all(resolved).unwrap();
    }
}
