//! Independent hemisphere quadrature over the saved first-order radiance.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{asset::Manifest, quadrature};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    first: PathBuf,
    #[arg(long)]
    second: PathBuf,
    #[arg(long, default_value_t = 17)]
    band: usize,
}
fn main() -> sky_atmosphere_lut::Result<()> {
    let args = Args::parse();
    let first = Manifest::open(&args.first)?.read_band(&args.first, args.band)?;
    let second = Manifest::open(&args.second)?.read_band(&args.second, args.band)?;
    for solar in [-6.0_f32, -4.0, 0.0, 20.0] {
        let e = solar.to_radians();
        let sun = Vec3::new(e.cos(), 0.0, e.sin());
        let baked = second.ground_irradiance_at(sun.z) - first.ground_irradiance_at(sun.z);
        for n in [16, 32, 64, 128] {
            let mut irradiance = 0.0;
            for (v, w) in quadrature::hemisphere(n, 4 * n) {
                irradiance += first.sample(0.0, v, sun, false)? * v.z * w;
            }
            println!(
                "sun {solar}, {n}x{}: integrated {irradiance:e}, baked {baked:e}, ratio {}",
                4 * n,
                baked / irradiance
            );
        }
    }
    Ok(())
}
