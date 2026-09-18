//! CPU fixed-point residual: compare stored L with direct light + K[L].
//! Does not rebake or alter the reference; residual is not a global error bound.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    Result,
    asset::{BandLut, Manifest, fingerprint},
    mapping::{State, sample_radiance_state},
    model::{BandModel, Model, phase_weight},
    quadrature,
    reference_mapping::{ReferenceStencil, radius_nodes},
};
use std::{f32::consts::PI, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "17")]
    bands: Vec<usize>,
    #[arg(long, default_value_t = 256)]
    steps: usize,
    #[arg(long, default_value_t = 32)]
    angular_mu: usize,
    #[arg(long, default_value_t = 64)]
    angular_phi: usize,
    /// Optional JSON array of [altitude, Sun elevation, azimuth, above horizon].
    #[arg(long)]
    poses: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    sun_mu: usize,
    #[arg(long, default_value_t = 16)]
    sun_phi: usize,
    /// Change only the query interpolation for a frozen-field diagnostic.
    #[arg(long)]
    cubic_view: bool,
}

fn direction(e: f32, a: f32) -> Vec3 {
    let (e, a) = (e.to_radians(), a.to_radians());
    Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
}
fn add(sum: &mut f32, correction: &mut f32, value: f32) {
    let y = value - *correction;
    let next = *sum + y;
    *correction = (next - *sum) - y;
    *sum = next;
}

#[allow(clippy::too_many_arguments)]
fn probe(
    lut: &BandLut,
    model: &BandModel,
    heights: &[f32],
    pose: [f32; 4],
    steps: usize,
    angular: &[(Vec3, f32)],
    disk: &[(Vec3, f32)],
) -> serde_json::Value {
    let [h, solar, az, above] = pose;
    let g = lut.geometry;
    let view = direction(g.horizon(h).asin().to_degrees() + above, az);
    let sun = direction(solar, 0.0);
    let original = State {
        altitude_km: h,
        mu: view.z,
        mu_s: sun.z,
        nu: view.dot(sun),
        ground: g.hits_ground(h, view.z),
    };
    let (s, _) = g.atmosphere_entry(original).unwrap();
    // Work in the local frame at atmospheric entry, then move its radial up
    // vector along the ray. View/light and scattering angle stay world fixed.
    let (ray, light) = s.directions();
    let sphere: Vec<_> = angular
        .iter()
        .flat_map(|&(local, weight)| {
            [(ray, light), (light, ray)].map(|(axis, other)| {
                let v = quadrature::rotate(local, axis);
                let a = 0.0004 + (1.0 - v.dot(axis)).max(0.0);
                let b = 0.0004 + (1.0 - v.dot(other)).max(0.0);
                (
                    v,
                    v.dot(light),
                    model.phases(v.dot(ray)),
                    weight * b * b / (a * a + b * b),
                )
            })
        })
        .collect();
    let solar_disk: Vec<_> = disk
        .iter()
        .map(|&(v, w)| {
            let v = quadrature::rotate(v, light);
            (v, model.phases(v.dot(ray)), w)
        })
        .collect();
    let length = g.distance(s.altitude_km, s.mu, s.ground);
    let dx = length / steps as f32;
    let mut direct = 0.0;
    let mut indirect = 0.0;
    let mut transmittance = 1.0;
    for j in 0..steps {
        let d = (j as f32 + 0.5) * dx;
        let point = g.advanced(s, d);
        let up = (Vec3::Z * (g.bottom + s.altitude_km) + ray * d) / (g.bottom + point.altitude_km);
        let c = model.coefficients(point.altitude_km);
        let mut source_direct = 0.0;
        for &(v, phase, w) in &solar_disk {
            let mu = v.dot(up).clamp(-1.0, 1.0);
            if !g.hits_ground(point.altitude_km, mu) {
                source_direct += phase_weight(c, phase)
                    * w
                    * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..point
                    })
                    * lut.info.solar_irradiance_w_m2;
            }
        }
        let mut source_indirect = 0.0;
        let mut correction = 0.0;
        for &(v, nu, phase, w) in &sphere {
            let mu = v.dot(up).clamp(-1.0, 1.0);
            let incoming = State {
                mu,
                nu,
                ground: g.hits_ground(point.altitude_km, mu),
                ..point
            };
            let radiance = if lut.config.mapping.is_reference() {
                ReferenceStencil::new(g, &lut.config, incoming, Some(heights))
                    .sample_with(|i| lut.radiance[i])
            } else {
                sample_radiance_state(
                    &lut.radiance,
                    g,
                    &lut.config,
                    incoming,
                    &lut.scattering_cosines,
                )
            };
            add(
                &mut source_indirect,
                &mut correction,
                radiance * phase_weight(c, phase) * w,
            );
        }
        let tau = c.extinction * dx;
        let weight = if tau < 0.001 {
            dx * (1.0 - tau * 0.5 + tau * tau / 6.0)
        } else {
            (1.0 - (-tau).exp()) / c.extinction
        };
        direct += transmittance * source_direct * weight;
        indirect += transmittance * source_indirect * weight;
        transmittance *= (-tau).exp();
    }
    let boundary = if s.ground {
        transmittance * lut.config.ground_albedo / PI
            * lut.ground_irradiance_at(g.advanced(s, length).mu_s)
    } else {
        0.0
    };
    let stored = sample_radiance_state(&lut.radiance, g, &lut.config, s, &lut.scattering_cosines);
    let reconstructed = direct + indirect + boundary;
    serde_json::json!({"altitude_km":h,"sun_deg":solar,"azimuth_deg":az,"above_horizon_deg":above,
        "nm":lut.info.center_nm,"stored":stored,"direct":direct,"indirect":indirect,"boundary":boundary,
        "reconstructed":reconstructed,"relative_residual_percent":100.0*(stored/reconstructed-1.0)})
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.steps == 0 || args.angular_mu == 0 || args.angular_phi == 0 {
        return Err("integration dimensions must be nonzero".into());
    }
    let manifest = Manifest::open(&args.source)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != manifest.model_fingerprint_fnv1a64 {
        return Err("reference and input model fingerprints differ".into());
    }
    let default_poses = [
        [0.2, 85.0, 0.0, 0.05],
        [0.2, 85.0, 0.0, 3.0],
        [0.2, 60.0, 0.0, 3.0],
        [0.2, 20.0, 0.0, 20.9544],
        [0.2, 45.0, 0.0, 20.0],
        [0.2, 0.0, 0.0, 0.2],
        [0.2, -6.0, 0.0, 0.2],
        [0.2, -6.0, 180.0, 5.0],
        [0.2, -12.0, 180.0, 5.0],
        [2.0, -6.0, 180.0, 1.0],
        [10.0, -6.0, 180.0, 1.0],
        [30.0, 20.0, 0.0, 0.2],
        [60.0, 20.0, 0.0, 0.2],
        [120.0, 20.0, 0.0, 0.2],
        [400.0, 20.0, 0.0, 0.2],
        [400.0, 20.0, 180.0, 0.5],
        [400.0, -6.0, 0.0, 0.5],
        [400.0, -6.0, 180.0, 0.5],
        [0.001, 85.0, 0.0, -0.1],
        [0.2, -6.0, 180.0, -5.0],
    ];
    let poses: Vec<[f32; 4]> = if let Some(path) = &args.poses {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        default_poses.to_vec()
    };
    let heights = radius_nodes(manifest.geometry, &manifest.config);
    let angular = quadrature::sphere(args.angular_mu, args.angular_phi);
    let disk = quadrature::sun_disk(args.sun_mu, args.sun_phi, model.sun_radius);
    let started = Instant::now();
    let mut output = Vec::new();
    for &band_index in &args.bands {
        let mut lut = manifest.read_band(&args.source, band_index)?;
        if args.cubic_view {
            lut.config.view_interpolation =
                sky_atmosphere_lut::config::ViewInterpolation::MonotoneCubic;
            lut.config.validate(manifest.bands.len())?;
        }
        for chunk in poses.chunks(8) {
            let rows = std::thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|&pose| {
                        let (lut, model, heights, angular, disk) =
                            (&lut, &model.bands[band_index], &heights, &angular, &disk);
                        scope.spawn(move || {
                            probe(lut, model, heights, pose, args.steps, angular, disk)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for row in rows {
                eprintln!("{row}");
                output.push(row);
            }
        }
        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &args.out,
            serde_json::to_vec_pretty(&serde_json::json!({
                "source":args.source,"lookup_override":args.cubic_view,"lookup_config":lut.config,"mapping":manifest.coordinate_mapping,"model":manifest.model_fingerprint_fnv1a64,
                "steps":args.steps,"angular_mu":args.angular_mu,"angular_phi":args.angular_phi,
                "sun_mu":args.sun_mu,"sun_phi":args.sun_phi,"elapsed_seconds":started.elapsed().as_secs_f32(),"probes":output,
                "note":"CPU frozen-field fixed-point residual. Ground uses stored irradiance. Direct light uses stored spectral optical depth. No global error bound, independent quadrature refinement required."
            }))?,
        )?;
    }
    Ok(())
}
