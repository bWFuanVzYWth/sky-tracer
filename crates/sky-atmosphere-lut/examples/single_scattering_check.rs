//! Independent f32 single-scattering ray integral, for diagnosing source-grid bias.
//! This is a small validation probe, never the asset baking implementation.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    asset::BandLut,
    config::BakeConfig,
    mapping::State,
    model::{Model, phase_weight},
    quadrature,
    solver::GpuBaker,
};
fn direction(e: f32, a: f32) -> Vec3 {
    let e = e.to_radians();
    let a = a.to_radians();
    Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
}
fn main() -> sky_atmosphere_lut::Result<()> {
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        config: Option<std::path::PathBuf>,
    }
    let args = Args::parse();
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let mut config = if let Some(path) = args.config {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        BakeConfig::default()
    };
    config.max_orders = 1;
    config.min_orders = 1;
    let gpu = GpuBaker::new()?;
    let b = gpu.bake_band(&model, &config, 17, |s| eprintln!("{s}"))?;
    let lut = BandLut {
        geometry: model.geometry,
        config: config.clone(),
        info: model.bands[17].info.clone(),
        sun_radius: model.sun_radius,
        optical_depth: b.optical_depth,
        radiance: b.radiance,
        ground_irradiance: b.ground_irradiance,
        scattering_cosines: (0..config.scattering[3])
            .map(|i| {
                sky_atmosphere_lut::mapping::scattering_cosine(
                    i as f32 / (config.scattering[3] - 1) as f32,
                )
            })
            .collect(),
    };
    for h in [40.0, 50.0, 60.0, 80.0] {
        let s = State {
            altitude_km: h,
            mu: (-6.0_f32).to_radians().sin(),
            mu_s: 0.0,
            nu: 0.0,
            ground: false,
        };
        let ds = model.geometry.distance(h, s.mu, false) / 16384.0;
        let mut tau = 0.0;
        for j in 0..16384 {
            tau += ds
                * model.bands[17]
                    .coefficients(
                        model
                            .geometry
                            .advanced(s, (j as f32 + 0.5) * ds)
                            .altitude_km,
                    )
                    .extinction;
        }
        println!(
            "h={h} solar T: LUT {}, integrated {} (tau {tau})",
            lut.transmittance(s),
            (-tau).exp()
        );
    }
    let mut cases = Vec::new();
    for solar in [-6.0, 0.0, 20.0] {
        for (e, a) in [
            (90.0, 0.0),
            (10.0, 180.0),
            (2.0, 0.0),
            (2.0, 90.0),
            (10.0, 45.0),
        ] {
            cases.push((solar, e, a));
        }
    }
    for delta in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        cases.push((20.0, 20.0 + delta, 0.0));
    }
    for a in [0.5, 1.0, 2.0, 5.0] {
        cases.push((20.0, 20.0, a));
    }
    for (solar, e, a) in cases {
        let ray = direction(e, a);
        let sun = direction(solar, 0.0);
        let g = model.geometry;
        let s = State {
            altitude_km: 0.2,
            mu: ray.z,
            mu_s: sun.z,
            nu: ray.dot(sun),
            ground: false,
        };
        let ds = g.distance(0.2, ray.z, false) / 4096.0;
        let mut trans = 1.0;
        let mut exact = 0.0;
        let sun_rule = quadrature::sun_disk(4, 16, model.sun_radius);
        for j in 0..4096 {
            let point = g.advanced(s, (j as f32 + 0.5) * ds);
            let c = model.bands[17].coefficients(point.altitude_km);
            let (view, sun) = point.directions();
            let mut source = 0.0;
            for &(v, w) in &sun_rule {
                let v = quadrature::rotate(v, sun);
                if g.hits_ground(point.altitude_km, v.z) {
                    continue;
                }
                let light = State {
                    mu: v.z,
                    ground: false,
                    ..point
                };
                source += w
                    * lut.transmittance(light)
                    * phase_weight(c, model.bands[17].phases(view.dot(v)));
            }
            let optical = c.extinction * ds;
            let weight = if optical < 0.001 {
                ds * (1.0 - optical * 0.5 + optical * optical / 6.0)
            } else {
                (1.0 - (-optical).exp()) / c.extinction
            };
            exact += trans * source * weight * lut.info.solar_irradiance_w_m2;
            trans *= (-optical).exp();
        }
        let value = lut.sample(0.2, ray, sun, false)?;
        println!(
            "sun={solar} view={e}/{a}: LUT {value:e}, direct integral {exact:e}, ratio {}",
            value / exact.max(1e-30)
        );
    }
    Ok(())
}
