//! CPU-only limb probe against an independent first-order line integral.
use glam::Vec3;
use sky_atmosphere_lut::{
    asset::Manifest,
    mapping::State,
    model::{Model, phase_weight},
    quadrature,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

fn main() -> sky_atmosphere_lut::Result<()> {
    let first_path = Path::new("out/lut_corner_first_probe");
    let first = Manifest::open(first_path)?.read_band(first_path, 17)?;
    let full_path = Path::new("out/lut_reference_v3");
    let full = Manifest::open(full_path)?.read_band(full_path, 17)?;
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let g = first.geometry;
    let altitude = 400.0;
    let horizon = g.horizon(altitude).asin();
    let mut out = BufWriter::new(File::create("out/lut_v3_orbit_first_order.csv")?);
    writeln!(
        out,
        "sun_deg,azimuth_deg,elevation_deg,tangent_height_km,first,direct,full"
    )?;
    for solar in [20.0_f32, -6.0] {
        let sun = Vec3::new(solar.to_radians().cos(), 0.0, solar.to_radians().sin());
        let disk: Vec<_> = quadrature::sun_disk(4, 16, model.sun_radius)
            .into_iter()
            .map(|(v, w)| (quadrature::rotate(v, sun), w))
            .collect();
        for azimuth in [0.0_f32, 90.0, 180.0] {
            for i in 1..=80 {
                let elevation = horizon + (i as f32 * 0.05).to_radians();
                let ray = Vec3::new(
                    elevation.cos() * azimuth.to_radians().cos(),
                    elevation.cos() * azimuth.to_radians().sin(),
                    elevation.sin(),
                );
                let original = State {
                    altitude_km: altitude,
                    mu: ray.z,
                    mu_s: sun.z,
                    nu: ray.dot(sun),
                    ground: false,
                };
                let Some((s, entry)) = g.atmosphere_entry(original) else {
                    continue;
                };
                let dx = g.distance(s.altitude_km, s.mu, false) / 4096.0;
                let mut direct = 0.0;
                let mut trans = 1.0;
                for j in 0..4096 {
                    let d = (j as f32 + 0.5) * dx;
                    let point = g.advanced(s, d);
                    let c = model.bands[17].coefficients(point.altitude_km);
                    let mut source = 0.0;
                    for &(light, w) in &disk {
                        let nu = ray.dot(light);
                        let mu = ((g.bottom + altitude) * light.z + (entry + d) * nu)
                            / (g.bottom + point.altitude_km);
                        if !g.hits_ground(point.altitude_km, mu) {
                            source +=
                                w * first.transmittance(State {
                                    mu,
                                    ground: false,
                                    ..point
                                }) * phase_weight(c, model.bands[17].phases(nu));
                        }
                    }
                    let tau = c.extinction * dx;
                    let weight = if tau < 0.001 {
                        dx * (1.0 - tau * 0.5 + tau * tau / 6.0)
                    } else {
                        (1.0 - (-tau).exp()) / c.extinction
                    };
                    direct += trans * source * weight * first.info.solar_irradiance_w_m2;
                    trans *= (-tau).exp();
                }
                writeln!(
                    out,
                    "{solar},{azimuth},{},{},{},{direct},{}",
                    elevation.to_degrees(),
                    (g.bottom + altitude) * elevation.cos() - g.bottom,
                    first.sample(altitude, ray, sun, false)?,
                    full.sample(altitude, ray, sun, false)?
                )?;
            }
        }
    }
    Ok(())
}
