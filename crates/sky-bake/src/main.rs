use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use sky_core::asset::{SpectralAssetFiles, SpectralAssetManifest};
use sky_core::atmosphere::SceneData;
use sky_core::data::load_scene_data;
use sky_offline::config::RenderConfig;
use sky_offline::integrator::render;

#[derive(Parser, Debug)]
#[command(version, about = "Offline spectral OPAC atmosphere path tracer")]
struct Cli {
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 512)]
    height: usize,
    #[arg(long, default_value_t = 256)]
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
    #[arg(long, default_value_t = 0.01)]
    png_exposure: f32,
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
    };

    let load_start = Instant::now();
    let scene = load_scene_data(
        &config.data_dir,
        config.sun_elevation_deg,
        config.sun_azimuth_deg,
    )?;
    let load_elapsed = load_start.elapsed();

    println!(
        "config: gpu-integrator direct-light-samples={}",
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
        "wrote {}x{} panorama with {} spp to {}",
        film.width(),
        film.height(),
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
    let manifest = SpectralAssetManifest::spectral_panorama(
        [config.width, config.height],
        config.spp,
        config.seed,
        config.sun_elevation_deg,
        config.sun_azimuth_deg,
        config.observer_altitude_km,
        scene.bands.iter().map(|band| band.center_nm).collect(),
        files,
    );
    let path = config.out_dir.join("asset.json");
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &manifest)?;
    Ok(())
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
