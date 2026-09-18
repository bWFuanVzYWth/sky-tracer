//! CPU-only analysis. Does not change the baker, renderer, or existing assets.
use glam::Vec3;
use sky_atmosphere_lut::{
    Result,
    asset::{BandLut, Manifest},
    config::BakeConfig,
    mapping::{Geometry, State, log_mix, sample_radiance_state, scattering_cosine, unit},
    model::{Model, phase_weight},
    quadrature,
};
use std::{
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

fn project(g: Geometry, s: State) -> State {
    let nu = g.cone_cosine(s.altitude_km, s.mu_s, s.nu, false);
    let center = s.mu_s * nu;
    let extent = ((1.0 - s.mu_s * s.mu_s) * (1.0 - nu * nu)).max(0.0).sqrt();
    let mu = s.mu.clamp(
        (center - extent).max(g.horizon(s.altitude_km)),
        (center + extent).max(g.horizon(s.altitude_km)),
    );
    State { mu, nu, ..s }
}

// Reproduce the production lookup's projection with one outer coordinate
// moved to a grid endpoint and the other coordinates kept continuous.
fn axis_end(g: Geometry, c: &BakeConfig, s: State, axis: usize, index: usize) -> State {
    let q = g.coords_config(s, c);
    if axis == 1 {
        return State {
            mu: g.optical_cone_view(s, index, c.scattering[1]),
            ..s
        };
    }
    let mut end = s;
    match axis {
        0 => {
            end.altitude_km = g.height(unit(index, c.scattering[0]));
            end.mu_s = g.solar_cosine_mapped(
                end.altitude_km,
                q[2] / (c.scattering[2] - 1) as f32,
                c.mapping,
            );
        }
        2 => {
            end.mu_s = g.solar_cosine_mapped(s.altitude_km, unit(index, c.scattering[2]), c.mapping)
        }
        3 => end.nu = scattering_cosine(unit(index, c.scattering[3])),
        _ => unreachable!(),
    }
    project(g, end)
}

fn axis_values(first: &BandLut, model: &Model, s: State) -> [f32; 4] {
    let c = &first.config;
    let q = first.geometry.coords_config(s, c);
    std::array::from_fn(|axis| {
        let lo = q[axis].floor().max(0.0) as usize;
        let hi = (lo + 1).min(c.scattering[axis] - 1);
        let t = (q[axis] - lo as f32).clamp(0.0, 1.0);
        let a = direct(first, model, axis_end(first.geometry, c, s, axis, lo), 4096);
        if lo == hi || t < 1e-6 {
            return a;
        }
        let b = direct(first, model, axis_end(first.geometry, c, s, axis, hi), 4096);
        if axis == 2 {
            log_mix(a, b, t)
        } else {
            a * (1.0 - t) + b * t
        }
    })
}

fn corner_metrics(g: Geometry, c: &BakeConfig, s: State) -> [f32; 4] {
    let q = g.coords_config(s, c);
    let mut result = [0.0_f32; 4]; // weighted view shift, max view shift, weighted phase shift, clamped weight
    for mask in 0..8 {
        let mut weight = 1.0;
        let mut ix = [0; 4];
        for (bit, axis) in [0, 2, 3].into_iter().enumerate() {
            let lo = q[axis].floor().max(0.0) as usize;
            let t = q[axis] - lo as f32;
            let up = (mask >> bit) & 1;
            ix[axis] = (lo + up).min(c.scattering[axis] - 1);
            weight *= if up == 1 { t } else { 1.0 - t };
        }
        if weight < 1e-7 {
            continue;
        }
        let h = g.height(unit(ix[0], c.scattering[0]));
        let mu_s = g.solar_cosine_mapped(h, unit(ix[2], c.scattering[2]), c.mapping);
        let raw_nu = scattering_cosine(unit(ix[3], c.scattering[3]));
        let end = project(
            g,
            State {
                altitude_km: h,
                mu_s,
                nu: raw_nu,
                ..s
            },
        );
        let shift = (end.mu.asin() - s.mu.asin()).abs().to_degrees();
        result[0] += shift * weight;
        result[1] = result[1].max(shift);
        result[2] += (end.nu.acos() - raw_nu.acos()).abs().to_degrees() * weight;
        if shift > 0.0001 {
            result[3] += weight;
        }
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

// Frozen-source convergence: integrate the already baked incident radiance.
// This isolates angular quadrature sensitivity, not converged full transport.
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

fn optical_depth_probe(full: &BandLut, model: &Model, root: &Path) -> Result<()> {
    let g = full.geometry;
    let mut csv = BufWriter::new(File::create(root.join("optical_depth.csv"))?);
    writeln!(
        csv,
        "height_km,above_horizon_deg,ground,tau_lut,tau2048,tau8192,tau_height_only,tau_view_only"
    )?;
    for h in [
        0.0_f32, 0.0001, 0.001, 0.01, 0.1, 0.2, 1.0, 2.0, 10.0, 30.0, 60.0, 100.0, 120.0,
    ] {
        for above in [
            -30.0_f32, -5.0, -1.0, -0.1, -0.001, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 30.0, 60.0,
        ] {
            let e = g.horizon(h).asin() + above.to_radians();
            if e <= -std::f32::consts::FRAC_PI_2 || e >= std::f32::consts::FRAC_PI_2 {
                continue;
            }
            let s = State {
                altitude_km: h,
                mu: e.sin(),
                mu_s: 0.0,
                nu: 0.0,
                ground: above < 0.0,
            };
            let q = [
                g.height_coord(h) * (full.config.optical_depth[0] - 1) as f32,
                g.view_coord(h, s.mu, s.ground, full.config.optical_depth[1]),
            ];
            let tau = sky_atmosphere_lut::mapping::sample(
                &full.optical_depth,
                full.config.optical_depth,
                q,
            );
            let direct_tau = |state: State, steps| {
                let dx = g.distance(state.altitude_km, state.mu, state.ground) / steps as f32;
                let mut sum = 0.0;
                let mut correction = 0.0;
                for j in 0..steps {
                    let p = g.advanced(state, (j as f32 + 0.5) * dx);
                    let term =
                        model.bands[17].coefficients(p.altitude_km).extinction * dx - correction;
                    let next = sum + term;
                    correction = (next - sum) - term;
                    sum = next;
                }
                sum
            };
            let decode = |height: f32, view: f32| {
                let half = full.config.optical_depth[1] / 2;
                let x = (view - if s.ground { 0.0 } else { half as f32 }) / (half - 1) as f32;
                let warp = x.powi(3) / (x.powi(3) + (1.0 - x).powi(3));
                let horizon = g.horizon(height);
                State {
                    altitude_km: height,
                    mu: if s.ground {
                        horizon - (1.0 + horizon) * warp
                    } else {
                        horizon + (1.0 - horizon) * warp
                    },
                    ..s
                }
            };
            let axis = |which: usize| {
                let lo = q[which].floor();
                let hi = (lo + 1.0).min((full.config.optical_depth[which] - 1) as f32);
                let t = q[which] - lo;
                let endpoint = |v: f32| {
                    if which == 0 {
                        decode(
                            g.height(v / (full.config.optical_depth[0] - 1) as f32),
                            q[1],
                        )
                    } else {
                        decode(h, v)
                    }
                };
                direct_tau(endpoint(lo), 8192) * (1.0 - t) + direct_tau(endpoint(hi), 8192) * t
            };
            writeln!(
                csv,
                "{h},{above},{},{tau},{},{},{},{}",
                u32::from(s.ground),
                direct_tau(s, 2048),
                direct_tau(s, 8192),
                axis(0),
                axis(1)
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let root = Path::new("out/lut_sampling_audit");
    fs::create_dir_all(root)?;
    let source = Path::new("out/lut_noon_cap97_first");
    let first = Manifest::open(source)?.read_band(source, 17)?;
    let full_dir = Path::new("out/lut_reference_v4");
    let full = Manifest::open(full_dir)?.read_band(full_dir, 17)?;
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let g = first.geometry;
    let c = &first.config;
    let nodes: Vec<_> = (0..c.scattering[3])
        .map(|i| scattering_cosine(unit(i, c.scattering[3])))
        .collect();
    if std::env::args().any(|a| a == "--tau") {
        optical_depth_probe(&full, &model, root)?;
        return Ok(());
    }
    if std::env::args().any(|a| a == "--source-and-profiles") {
        profiles(&first, &full, &nodes, root)?;
        angular_probe(&full, &model, &nodes, root)?;
        return Ok(());
    }
    let heights: Vec<_> = (0..c.scattering[0])
        .map(|i| g.height(unit(i, c.scattering[0])))
        .collect();
    let solar:Vec<_>=[0.0,0.2,2.0,10.0,30.0,60.0,120.0].into_iter().map(|h|serde_json::json!({
        "altitude_km":h,"elevation_deg":(0..c.scattering[2]).map(|i|g.solar_cosine_mapped(h,unit(i,c.scattering[2]),c.mapping).asin().to_degrees()).collect::<Vec<_>>()
    })).collect();
    let phase:Vec<_>=[6,17,30].into_iter().map(|i|{
        let b=&model.bands[i];let coeff=b.coefficients(0.2);
        serde_json::json!({"nm":b.info.center_nm,"node_phase":nodes.iter().map(|&nu|phase_weight(coeff,b.phases(nu))).collect::<Vec<_>>(),
            "mid_phase":(0..255).map(|j|phase_weight(coeff,b.phases(scattering_cosine((j as f32+0.5)/255.0)))).collect::<Vec<_>>()})
    }).collect();
    let angular: Vec<_> = quadrature::gauss_legendre(16)
        .into_iter()
        .map(|(u, _)| (1.0 - 2.0 * u.powi(3)).acos().to_degrees())
        .collect();
    fs::write(
        root.join("nodes.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"config":full.config,"geometry":g,"height_km":heights,"solar":solar,
        "phase_angle_deg":nodes.iter().map(|n|n.acos().to_degrees()).collect::<Vec<_>>(),"phase_mid_angle_deg":(0..255).map(|j|scattering_cosine((j as f32+0.5)/255.0).acos().to_degrees()).collect::<Vec<_>>(),"phase":phase,"integration_ring_angle_deg":angular}),
        )?,
    )?;
    let mut queries: Vec<(&str, f32, f32, f32, f32)> = Vec::new();
    for sun in [
        -12.0, -8.0, -6.0, -4.0, -2.0, 0.0, 20.0, 45.0, 60.0, 70.0, 75.0, 80.0, 82.5, 85.0, 87.5,
        89.0, 90.0,
    ] {
        for az in [0.0, 90.0, 180.0] {
            for above in [0.002, 0.05, 0.2, 1.0, 3.0, 10.0, 30.0] {
                queries.push(("ground_sweep", 0.2, sun, az, above));
            }
        }
    }
    for h in [
        0.0, heights[1], heights[2], heights[3], 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0,
    ] {
        for sun in [-6.0, 20.0, 85.0] {
            for az in [0.0, 90.0, 180.0] {
                for above in [0.02, 0.3, 3.0] {
                    queries.push(("height_sweep", h, sun, az, above));
                }
            }
        }
    }
    for sun in [0.0_f32, 20.0, 45.0, 80.0, 85.0, 89.0] {
        for offset in [-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let elevation = sun + offset;
            if elevation > g.horizon(0.2).asin().to_degrees() && elevation < 90.0 {
                queries.push((
                    "aureole",
                    0.2,
                    sun,
                    0.0,
                    elevation - g.horizon(0.2).asin().to_degrees(),
                ));
            }
        }
    }
    for sun in [-6.0, 20.0, 85.0] {
        for az in [0.0, 90.0, 180.0] {
            for above in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0] {
                queries.push(("orbit", 400.0, sun, az, above));
            }
        }
    }
    let interior = std::env::args().any(|a| a == "--interior-only");
    if interior {
        queries.clear();
        for h in [0.2, 10.0, 60.0] {
            for sun in [-6.0, 20.0, 60.0, 85.0] {
                for az in [15.0, 30.0, 45.0, 60.0, 120.0, 150.0] {
                    for above in [0.05, 1.0, 10.0, 30.0, 60.0] {
                        queries.push(("interior", h, sun, az, above));
                    }
                }
            }
        }
    }
    let mut csv = BufWriter::new(File::create(root.join(if interior {
        "interior_probes.csv"
    } else {
        "axis_probes.csv"
    }))?);
    writeln!(
        csv,
        "case,observer_km,sun_deg,azimuth_deg,above_horizon_deg,entry_km,entry_sun_deg,entry_view_deg,phase_deg,q_height,q_view,q_sun,q_phase,first,full,direct,direct256,height_only,view_only,solar_only,phase_only,corner_mean_view_shift_deg,corner_max_view_shift_deg,corner_mean_phase_clamp_deg,corner_clamped_weight,path_km"
    )?;
    let start = std::time::Instant::now();
    for (i, &(case, h, sun, az, above)) in queries.iter().enumerate() {
        let e = g.horizon(h).asin() + above.to_radians();
        let se = sun.to_radians();
        let az = az.to_radians();
        let ray = Vec3::new(e.cos() * az.cos(), e.cos() * az.sin(), e.sin());
        let light = Vec3::new(se.cos(), 0.0, se.sin());
        let s = State {
            altitude_km: h,
            mu: ray.z,
            mu_s: light.z,
            nu: ray.dot(light),
            ground: false,
        };
        let Some((s, _)) = g.atmosphere_entry(s) else {
            continue;
        };
        let exact = direct(&first, &model, s, 4096);
        let low = direct(&first, &model, s, 256);
        let axes = axis_values(&first, &model, s);
        let q = g.coords_config(s, c);
        let cm = corner_metrics(g, c, s);
        let a = sample_radiance_state(&first.radiance, g, c, s, &nodes);
        let b = sample_radiance_state(&full.radiance, g, &full.config, s, &nodes);
        writeln!(
            csv,
            "{case},{h},{sun},{},{above},{},{},{},{},{},{},{},{},{a},{b},{exact},{low},{},{},{},{},{},{},{},{},{}",
            az.to_degrees(),
            s.altitude_km,
            s.mu_s.asin().to_degrees(),
            s.mu.asin().to_degrees(),
            s.nu.acos().to_degrees(),
            q[0],
            q[1],
            q[2],
            q[3],
            axes[0],
            axes[1],
            axes[2],
            axes[3],
            cm[0],
            cm[1],
            cm[2],
            cm[3],
            g.distance(s.altitude_km, s.mu, false)
        )?;
        if i % 50 == 0 {
            csv.flush()?;
            eprintln!(
                "{i}/{} probes, {:.1}s",
                queries.len(),
                start.elapsed().as_secs_f32()
            );
        }
    }
    eprintln!(
        "Complete {} CPU probes, {:.1}s",
        queries.len(),
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
