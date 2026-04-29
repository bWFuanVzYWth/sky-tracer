use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use sky_tracer::config::RenderConfig;
use sky_tracer::data::load_scene_data;
use sky_tracer::integrator::render;

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
    #[arg(long, default_value_t = 16)]
    max_depth: usize,
    #[arg(long, default_value_t = 0.01)]
    png_exposure: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
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
        max_depth: cli.max_depth,
        png_exposure: cli.png_exposure,
    };

    let scene = load_scene_data(
        &config.data_dir,
        config.sun_elevation_deg,
        config.sun_azimuth_deg,
    )?;
    let film = render(&scene, &config);
    film.write_outputs(&config.out_dir, &scene.bands, config.png_exposure)?;

    println!(
        "wrote {}x{} panorama with {} spp to {}",
        film.width(),
        film.height(),
        config.spp,
        config.out_dir.display()
    );
    Ok(())
}
