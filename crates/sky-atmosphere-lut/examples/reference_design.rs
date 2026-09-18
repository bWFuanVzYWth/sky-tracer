//! Deterministic first-order acceptance probes for the reference chart.
use glam::Vec3;
use sky_atmosphere_lut::{
    Result,
    asset::{BandLut, Manifest},
    config::BakeConfig,
    mapping::{State, sample_radiance_state},
    model::{Model, phase_weight},
    quadrature, reference_mapping,
};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
fn direct(lut: &BandLut, model: &Model, s: State, steps: usize) -> f32 {
    let (ray, sun) = s.directions();
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
    let dx = g.distance(s.altitude_km, s.mu, false) / steps as f32;
    let mut result = 0.0;
    let mut trans = 1.0;
    for j in 0..steps {
        let d = (j as f32 + 0.5) * dx;
        let point = g.advanced(s, d);
        let c = model.bands[17].coefficients(point.altitude_km);
        let mut source = 0.0;
        for &(sun_z, nu, phase, w) in &disk {
            let mu = ((g.bottom + s.altitude_km) * sun_z + d * nu) / (g.bottom + point.altitude_km);
            if !g.hits_ground(point.altitude_km, mu) {
                source +=
                    w * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..point
                    }) * phase_weight(c, phase);
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

fn profiles(first: &BandLut, full: &BandLut, nodes: &[f32], root: &Path) -> Result<()> {
    let g = full.geometry;
    let mut csv = BufWriter::new(File::create(root.join("dense_profiles.csv"))?);
    writeln!(
        csv,
        "sun_deg,azimuth_deg,above_horizon_deg,first,full,q_phase,q_view"
    )?;
    for solar in [60.0_f32, 80.0, 85.0, 87.0, 89.0] {
        for az in [0.0_f32, 30.0, 90.0, 180.0] {
            for i in 0..1501 {
                let above = 0.002 + i as f32 * 0.01;
                let e = g.horizon(0.2).asin() + above.to_radians();
                let se = solar.to_radians();
                let a = az.to_radians();
                let s = State {
                    altitude_km: 0.2,
                    mu: e.sin(),
                    mu_s: se.sin(),
                    nu: e.sin() * se.sin() + e.cos() * se.cos() * a.cos(),
                    ground: false,
                };
                let q = g.coords_config(s, &full.config);
                let f = sample_radiance_state(&first.radiance, g, &first.config, s, nodes);
                let l = sample_radiance_state(&full.radiance, g, &full.config, s, nodes);
                writeln!(csv, "{solar},{az},{above},{f},{l},{},{}", q[3], q[1])?;
            }
        }
    }
    Ok(())
}

fn angular_probe(full: &BandLut, model: &Model, nodes: &[f32], root: &Path) -> Result<()> {
    let g = full.geometry;
    let mut csv = BufWriter::new(File::create(root.join("angular_source.csv"))?);
    writeln!(
        csv,
        "height_km,sun_deg,azimuth_deg,above_horizon_deg,n_mu,n_phi,source"
    )?;
    for (h, sun, az, above) in [
        (0.2_f32, -6.0_f32, 0.0_f32, 0.2_f32),
        (0.2, -6.0, 180.0, 0.2),
        (2.0, -6.0, 180.0, 1.0),
        (10.0, -6.0, 180.0, 1.0),
        (0.2, 85.0, 0.0, 0.2),
        (0.2, 20.0, 0.0, 20.95438),
        (30.0, 20.0, 0.0, 0.2),
        (120.0, 20.0, 0.0, 0.2),
    ] {
        let e = g.horizon(h).asin() + above.to_radians();
        let a = az.to_radians();
        let se = sun.to_radians();
        let view = Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin());
        let light = Vec3::new(se.cos(), 0.0, se.sin());
        let coeff = model.bands[17].coefficients(h);
        for (nm, np) in [
            (16, 32),
            (32, 32),
            (16, 64),
            (32, 64),
            (64, 128),
            (128, 256),
        ] {
            let mut sum = 0.0;
            let mut correction = 0.0;
            for (local, w) in quadrature::sphere(nm, np) {
                for (axis, other) in [(view, light), (light, view)] {
                    let v = quadrature::rotate(local, axis);
                    let pa = 0.0004 + (1.0 - v.dot(axis)).max(0.0);
                    let pb = 0.0004 + (1.0 - v.dot(other)).max(0.0);
                    let weight = pb * pb / (pa * pa + pb * pb);
                    let s = State {
                        altitude_km: h,
                        mu: v.z,
                        mu_s: light.z,
                        nu: v.dot(light),
                        ground: g.hits_ground(h, v.z),
                    };
                    let value = sample_radiance_state(&full.radiance, g, &full.config, s, nodes);
                    let term = value
                        * phase_weight(coeff, model.bands[17].phases(v.dot(view)))
                        * w
                        * weight
                        - correction;
                    let next = sum + term;
                    correction = (next - sum) - term;
                    sum = next;
                }
            }
            writeln!(csv, "{h},{sun},{az},{above},{nm},{np},{sum}")?;
        }
        csv.flush()?;
        eprintln!("Angular source h={h} sun={sun} az={az}");
    }
    Ok(())
}

fn phase_field(model: &Model) -> Result<()> {
    let c = BakeConfig::reference();
    let g = model.geometry;
    let disk = quadrature::sun_disk(4, 16, model.sun_radius);
    let mut out = BufWriter::new(File::create("out/lut_reference_design/phase_field.csv")?);
    writeln!(out, "nm,sun_deg,offset_deg,exact,interpolated")?;
    for band in [6, 17, 30] {
        let coeff = model.bands[band].coefficients(0.2);
        let phase = |nu: f32| {
            let sun = Vec3::new((1.0 - nu * nu).max(0.0).sqrt(), 0.0, nu);
            disk.iter()
                .map(|&(v, w)| {
                    w * phase_weight(
                        coeff,
                        model.bands[band].phases(quadrature::rotate(v, sun).z),
                    )
                })
                .sum::<f32>()
        };
        let mut cache = HashMap::new();
        for solar in [0.0_f32, 20.0, 45.0, 60.0, 80.0, 85.0, 89.0, 90.0] {
            for offset in [
                -20.0_f32, -10.0, -5.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0,
                10.0, 20.0,
            ] {
                let elevation = solar + offset;
                if elevation > 90.0 || elevation < g.horizon(0.2).asin().to_degrees() {
                    continue;
                }
                let s = State {
                    altitude_km: 0.2,
                    mu: elevation.to_radians().sin(),
                    mu_s: solar.to_radians().sin(),
                    nu: offset.to_radians().cos(),
                    ground: false,
                };
                let actual = reference_mapping::sample_with(g, &c, s, |index| {
                    *cache
                        .entry(index)
                        .or_insert_with(|| phase(g.state_config(index, &c).nu))
                });
                writeln!(
                    out,
                    "{},{solar},{offset},{},{actual}",
                    model.bands[band].info.center_nm,
                    phase(s.nu)
                )?;
            }
        }
    }
    Ok(())
}

fn audit_tau(path: &Path) -> Result<()> {
    let lut = Manifest::open(path)?.read_band(path, 17)?;
    let g = lut.geometry;
    let csv = fs::read_to_string("out/lut_sampling_audit/optical_depth.csv")?;
    let mut out = BufWriter::new(File::create("out/lut_reference_design/optical_depth.csv")?);
    writeln!(
        out,
        "height_km,above_horizon_deg,ground,tau_old,tau_direct,tau_new"
    )?;
    for line in csv.lines().skip(1) {
        let f: Vec<_> = line.split(',').collect();
        let h: f32 = f[0].parse()?;
        let above: f32 = f[1].parse()?;
        let mu = (g.horizon(h).asin() + above.to_radians()).sin();
        let state = State {
            altitude_km: h,
            mu,
            mu_s: 0.0,
            nu: 0.0,
            ground: f[2] == "1",
        };
        let coords = [
            g.height_coord_mapped(h, lut.config.mapping) * (lut.config.optical_depth[0] - 1) as f32,
            g.view_coord(h, mu, state.ground, lut.config.optical_depth[1]),
        ];
        let tau = sky_atmosphere_lut::mapping::sample(
            &lut.optical_depth,
            lut.config.optical_depth,
            coords,
        );
        writeln!(out, "{h},{above},{},{},{},{tau}", f[2], f[3], f[5])?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--profiles") {
        let full = Manifest::open(Path::new(&args[2]))?.read_band(Path::new(&args[2]), 17)?;
        let first = Manifest::open(Path::new(&args[3]))?.read_band(Path::new(&args[3]), 17)?;
        return profiles(&first, &full, &[], Path::new("out/lut_reference_design"));
    }
    if args.get(1).is_some_and(|s| s == "--tau") {
        return audit_tau(Path::new(&args[2]));
    }

    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let m = Manifest::open(Path::new("out/lut_noon_cap97_first"))?;
    let old = m.read_band(Path::new("out/lut_noon_cap97_first"), 17)?;
    if args.get(1).is_some_and(|s| s == "--angular") {
        let path = Path::new(&args[2]);
        let lut = Manifest::open(path)?.read_band(path, 17)?;
        return angular_probe(&lut, &model, &[], Path::new("out/lut_reference_design"));
    }
    if args.get(1).is_some_and(|s| s == "--phase-field") {
        return phase_field(&model);
    }
    let c = BakeConfig::reference();
    let g = old.geometry;
    fs::create_dir_all("out/lut_reference_design")?;
    fs::write(
        "out/lut_reference_design/config.json",
        serde_json::to_vec_pretty(&c)?,
    )?;
    let args: Vec<_> = std::env::args().collect();
    let baked = if let Some(path) = args.get(1) {
        let path = Path::new(path);
        Some(Manifest::open(path)?.read_band(path, 17)?)
    } else {
        None
    };
    let mut out = BufWriter::new(File::create(if baked.is_some() {
        "out/lut_reference_design/baked_probes.csv"
    } else {
        "out/lut_reference_design/corner_probes.csv"
    })?);
    writeln!(
        out,
        "case,observer_km,sun_deg,azimuth_deg,above_horizon_deg,old,direct,new"
    )?;
    let mut cache = HashMap::new();
    let csv = fs::read_to_string("out/lut_sampling_audit/axis_probes.csv")?;
    for (j, line) in csv.lines().skip(1).enumerate() {
        let f: Vec<_> = line.split(',').collect();
        let n = |i: usize| f[i].parse::<f32>().unwrap();
        let h = n(1);
        let sun = n(2).to_radians();
        let az = n(3).to_radians();
        let e = g.horizon(h.min(g.top_height())).asin() + n(4).to_radians();
        // The CSV already stores the clipped entry state; reconstruct from it for orbital probes.
        let eh = n(5);
        let es = n(6).to_radians();
        let ev = n(7).to_radians();
        let theta = n(8).to_radians();
        let s = if h > g.top_height() {
            State {
                altitude_km: eh,
                mu: ev.sin(),
                mu_s: es.sin(),
                nu: theta.cos(),
                ground: false,
            }
        } else {
            State {
                altitude_km: h,
                mu: e.sin(),
                mu_s: sun.sin(),
                nu: e.sin() * sun.sin() + e.cos() * sun.cos() * az.cos(),
                ground: false,
            }
        };
        let value = if let Some(lut) = &baked {
            sample_radiance_state(&lut.radiance, g, &lut.config, s, &lut.scattering_cosines)
        } else {
            reference_mapping::sample_with(g, &c, s, |idx| {
                *cache
                    .entry(idx)
                    .or_insert_with(|| direct(&old, &model, g.state_config(idx, &c), 4096))
            })
        };
        writeln!(
            out,
            "{},{h},{},{},{},{},{},{value}",
            f[0], f[2], f[3], f[4], f[13], f[15]
        )?;
        if j % 100 == 0 {
            eprintln!("{j} probes, {} direct corners", cache.len());
        }
    }
    Ok(())
}
