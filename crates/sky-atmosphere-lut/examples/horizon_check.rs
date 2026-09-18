//! CPU-only noon/horizon regression against independent first scattering.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    asset::{BandLut, Manifest},
    mapping::State,
    model::{Model, phase_weight},
    quadrature,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/lut_corner_first_probe")]
    first: PathBuf,
    #[arg(long, default_value = "out/lut_reference_v3")]
    full: PathBuf,
    #[arg(long, default_value = "out/noon_horizon_before.csv")]
    out: PathBuf,
}

fn direct(lut: &BandLut, model: &Model, s: State, ray: Vec3, sun: Vec3) -> f32 {
    let g = lut.geometry;
    let disk: Vec<_> = quadrature::sun_disk(4, 16, model.sun_radius)
        .into_iter()
        .map(|(v, w)| {
            let light = quadrature::rotate(v, sun);
            (
                light.z,
                ray.dot(light),
                model.bands[17].phases(ray.dot(light)),
                w,
            )
        })
        .collect();
    let dx = g.distance(s.altitude_km, s.mu, false) / 4096.0;
    let mut result = 0.0;
    let mut trans = 1.0;
    for j in 0..4096 {
        let d = (j as f32 + 0.5) * dx;
        let point = g.advanced(s, d);
        let c = model.bands[17].coefficients(point.altitude_km);
        let mut source = 0.0;
        for &(sun_z, nu, phases, w) in &disk {
            let mu = ((g.bottom + s.altitude_km) * sun_z + d * nu) / (g.bottom + point.altitude_km);
            if !g.hits_ground(point.altitude_km, mu) {
                source +=
                    w * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..point
                    }) * phase_weight(c, phases);
            }
        }
        let tau = c.extinction * dx;
        let weight = if tau < 0.001 {
            dx * (1.0 - tau * 0.5 + tau * tau / 6.0)
        } else {
            (1.0 - (-tau).exp()) / c.extinction
        };
        result += trans * source * weight * lut.info.solar_irradiance_w_m2;
        trans *= (-tau).exp();
    }
    result
}

fn main() -> sky_atmosphere_lut::Result<()> {
    let a = Args::parse();
    let first = Manifest::open(&a.first)?.read_band(&a.first, 17)?;
    let full = Manifest::open(&a.full)?.read_band(&a.full, 17)?;
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let h = 0.2;
    let horizon = first.geometry.horizon(h).asin().to_degrees();
    let mut out = BufWriter::new(File::create(a.out)?);
    writeln!(
        out,
        "sun_deg,azimuth_deg,above_horizon_deg,elevation_deg,first,direct,full"
    )?;
    for solar in [60.0_f32, 70.0, 80.0, 85.0, 88.0, 89.0, 89.5, 90.0] {
        let sun = Vec3::new(solar.to_radians().cos(), 0.0, solar.to_radians().sin());
        for azimuth in [0.0_f32, 30.0, 90.0, 180.0] {
            for i in 0..=80 {
                let above = 0.002 + 10.0 * (i as f32 / 80.0).powi(2);
                let elevation = (horizon + above).to_radians();
                let ray = Vec3::new(
                    elevation.cos() * azimuth.to_radians().cos(),
                    elevation.cos() * azimuth.to_radians().sin(),
                    elevation.sin(),
                );
                let s = State {
                    altitude_km: h,
                    mu: ray.z,
                    mu_s: sun.z,
                    nu: ray.dot(sun),
                    ground: false,
                };
                writeln!(
                    out,
                    "{solar},{azimuth},{above},{},{},{},{}",
                    elevation.to_degrees(),
                    first.sample(h, ray, sun, false)?,
                    direct(&first, &model, s, ray, sun),
                    full.sample(h, ray, sun, false)?
                )?;
            }
        }
        eprintln!("sampled {solar}");
    }
    Ok(())
}
