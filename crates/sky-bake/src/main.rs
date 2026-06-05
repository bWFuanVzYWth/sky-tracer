use std::error::Error;
use std::fs::File;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use sky_core::asset::{
    SpectralAssetColorimetry, SpectralAssetFiles, SpectralAssetManifest, SpectralAssetWhiteBalance,
};
use sky_core::atmosphere::SceneData;
use sky_core::data::load_scene_data;
use sky_core::spectrum::{
    CIE_1931_2DEG_CMF_NAME, CIE_1931_2DEG_CMF_SOURCE, LINEAR_SRGB_D65_NAME, SpectralBand,
    SpectralRgbConverter,
};
use sky_offline::config::{OutputProjection, RenderConfig};
use sky_offline::film::Film;
use sky_offline::integrator::render;

#[derive(Parser, Debug)]
#[command(version, about = "Offline spectral OPAC atmosphere path tracer")]
struct Cli {
    #[arg(long, default_value_t = 2048)]
    width: usize,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    #[arg(long, default_value_t = 1024)]
    spp: usize,
    #[arg(long, default_value_t = 0x5EC7_2026_0430_u64)]
    seed: u64,
    #[arg(long, default_value = "out")]
    out: PathBuf,
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long, default_value_t = 0.0)]
    sun_elevation_deg: f32,
    #[arg(long, default_value_t = 0.0)]
    sun_azimuth_deg: f32,
    #[arg(long, default_value_t = 0.2)]
    observer_altitude_km: f32,
    #[arg(long, default_value_t = 1)]
    direct_light_samples: usize,
    #[arg(long, default_value_t = 0.1)]
    png_exposure: f32,
    #[arg(long)]
    sky_view_lut: bool,
    #[arg(long)]
    rebuild_rgb_only: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    let cli = Cli::parse();
    let config = RenderConfig {
        width: cli.width,
        height: cli.height,
        spp: cli.spp,
        seed: cli.seed,
        out_dir: cli.out,
        data_dir: cli.data_dir,
        sun_elevation_deg: cli.sun_elevation_deg,
        sun_azimuth_deg: cli.sun_azimuth_deg,
        observer_altitude_km: cli.observer_altitude_km,
        direct_light_samples: cli.direct_light_samples,
        png_exposure: cli.png_exposure,
        output_projection: if cli.sky_view_lut {
            OutputProjection::SkyViewLut
        } else {
            OutputProjection::Panorama
        },
    };

    let load_start = Instant::now();
    let scene = load_scene_data(
        &config.data_dir,
        config.sun_elevation_deg,
        config.sun_azimuth_deg,
    )?;
    let load_elapsed = load_start.elapsed();

    if cli.rebuild_rgb_only {
        let output_start = Instant::now();
        let film = read_film_from_band_exrs(&config.out_dir, &scene.bands)?;
        film.write_rgb_outputs(&config.out_dir, &scene.bands, config.png_exposure)?;
        update_asset_colorimetry(&config.out_dir, &scene)?;
        let output_elapsed = output_start.elapsed();
        let total_elapsed = total_start.elapsed();
        println!(
            "rebuilt rgb outputs from {} band EXRs in {}",
            scene.bands.len(),
            config.out_dir.display()
        );
        println!(
            "timing: load={} output={} total={}",
            format_duration(load_elapsed),
            format_duration(output_elapsed),
            format_duration(total_elapsed)
        );
        return Ok(());
    }

    println!(
        "config: gpu-integrator output={} direct-light-samples={}",
        config.output_projection.label(),
        config.direct_light_samples
    );

    let render_start = Instant::now();
    let film = render(&scene, &config)?;
    let render_elapsed = render_start.elapsed();

    let output_start = Instant::now();
    film.write_outputs(&config.out_dir, &scene.bands, config.png_exposure)?;
    write_asset_manifest(&config, &scene)?;
    let output_elapsed = output_start.elapsed();
    let total_elapsed = total_start.elapsed();

    println!(
        "wrote {}x{} {} with {} spp to {}",
        film.width(),
        film.height(),
        config.output_projection.label(),
        config.spp,
        config.out_dir.display()
    );
    println!(
        "timing: load={} render={} output={} total={}",
        format_duration(load_elapsed),
        format_duration(render_elapsed),
        format_duration(output_elapsed),
        format_duration(total_elapsed)
    );
    Ok(())
}

fn read_film_from_band_exrs(
    out_dir: &Path,
    bands: &[SpectralBand],
) -> Result<Film, Box<dyn Error>> {
    if bands.is_empty() {
        return Err(IoError::new(ErrorKind::InvalidInput, "no spectral bands to rebuild").into());
    }

    let mut width = 0;
    let mut height = 0;
    let mut values = Vec::new();

    for (band_index, band) in bands.iter().enumerate() {
        let path = out_dir
            .join("bands")
            .join(format!("sky_{:03.0}nm.exr", band.center_nm));
        let image = read_band_exr(&path)?;
        if band_index == 0 {
            width = image.width;
            height = image.height;
            values.resize(width * height * bands.len(), 0.0);
        } else if image.width != width || image.height != height {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!(
                    "band EXR {} has dimensions {}x{}, expected {}x{}",
                    path.display(),
                    image.width,
                    image.height,
                    width,
                    height
                ),
            )
            .into());
        }

        for (pixel, value) in image.pixels.iter().enumerate() {
            values[pixel * bands.len() + band_index] = *value;
        }
    }

    Ok(Film::from_values(width, height, bands.len(), values))
}

#[derive(Debug)]
struct BandExrImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

fn read_band_exr(path: &Path) -> Result<BandExrImage, Box<dyn Error>> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            let width = resolution.width();
            let height = resolution.height();
            BandExrImage {
                width,
                height,
                pixels: vec![0.0; width * height],
            }
        },
        |image, position, (r, _g, _b, _a): (f32, f32, f32, f32)| {
            image.pixels[position.y() * image.width + position.x()] = r;
        },
    )?;
    Ok(image.layer_data.channel_data.pixels)
}

fn write_asset_manifest(config: &RenderConfig, scene: &SceneData) -> Result<(), Box<dyn Error>> {
    let band_exrs = scene
        .bands
        .iter()
        .map(|band| format!("bands/sky_{:03.0}nm.exr", band.center_nm))
        .collect();
    let files = SpectralAssetFiles {
        rgb_exr: "sky_rgb.exr".to_owned(),
        rgb_png: "sky_rgb.png".to_owned(),
        band_exrs,
    };
    let dimensions = [config.width, config.height];
    let band_centers_nm = scene.bands.iter().map(|band| band.center_nm).collect();
    let mut manifest = match config.output_projection {
        OutputProjection::Panorama => SpectralAssetManifest::spectral_panorama(
            dimensions,
            config.spp,
            config.seed,
            config.sun_elevation_deg,
            config.sun_azimuth_deg,
            config.observer_altitude_km,
            band_centers_nm,
            files,
        ),
        OutputProjection::SkyViewLut => SpectralAssetManifest::spectral_sky_view_lut(
            dimensions,
            config.spp,
            config.seed,
            config.sun_elevation_deg,
            config.sun_azimuth_deg,
            config.observer_altitude_km,
            band_centers_nm,
            files,
        ),
    };
    manifest.colorimetry = Some(colorimetry_from_scene(scene));
    let path = config.out_dir.join("asset.json");
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &manifest)?;
    Ok(())
}

fn update_asset_colorimetry(out_dir: &Path, scene: &SceneData) -> Result<(), Box<dyn Error>> {
    let path = out_dir.join("asset.json");
    if !path.exists() {
        return Ok(());
    }
    let file = File::open(&path)?;
    let mut manifest: SpectralAssetManifest = serde_json::from_reader(file)?;
    manifest.colorimetry = Some(colorimetry_from_scene(scene));
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &manifest)?;
    Ok(())
}

fn colorimetry_from_scene(scene: &SceneData) -> SpectralAssetColorimetry {
    let white_balance = SpectralRgbConverter::new_solar_d65(&scene.bands).white_balance();
    SpectralAssetColorimetry {
        cmf: CIE_1931_2DEG_CMF_NAME.to_owned(),
        cmf_source: CIE_1931_2DEG_CMF_SOURCE.to_owned(),
        rgb_color_space: LINEAR_SRGB_D65_NAME.to_owned(),
        white_balance: SpectralAssetWhiteBalance {
            method: white_balance.method.to_owned(),
            source_white_xyz_y1: white_balance.source_white_xyz_y1,
            target_white_xyz_y1: white_balance.target_white_xyz_y1,
            xyz_from_xyz: white_balance.xyz_from_xyz,
        },
    }
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1.0 {
        format!("{:.1}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.2}s")
    } else {
        let minutes = (secs / 60.0).floor();
        let seconds = secs - minutes * 60.0;
        format!("{minutes:.0}m{seconds:.1}s")
    }
}
