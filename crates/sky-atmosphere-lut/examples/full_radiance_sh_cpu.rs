//! Optimistic angular-only audit of full sky/ground radiance SH compression.
//! No single-scattering subtraction, phase factoring, or source convolution.
use clap::Parser;
use glam::Vec3;
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result, anisotropic as sh,
    asset::Manifest,
    mapping::State,
    quadrature,
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb,
};
use std::{f32::consts::PI, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 96)]
    angular_mu: usize,
    #[arg(long, default_value_t = 192)]
    angular_phi: usize,
    #[arg(long, default_value_t = 64)]
    degree: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.degree > 96 || a.angular_mu < a.degree + 1 || a.angular_phi < 2 * a.degree + 1 {
        return Err("projection quadrature must resolve the requested SH degree".into());
    }
    fs::create_dir_all(&a.out)?;
    let start = Instant::now();
    let m = Manifest::open(&a.source)?;
    let channels: Vec<_> = (0..3)
        .map(|k| rgb::read_channel(&m, &a.source, k).map(|x| x.1))
        .collect::<Result<_>>()?;
    let heights = radius_nodes(m.geometry, &m.config);
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let scenes = plan["images"].as_array().ok_or("missing images")?;
    let degrees: Vec<_> = [4, 8, 12, 16, 17, 24, 32, 48, 64, 96]
        .into_iter()
        .filter(|&x| x <= a.degree)
        .collect();
    let sphere = quadrature::sphere(a.angular_mu, a.angular_phi);
    let radial = quadrature::gauss_legendre(a.angular_mu);
    let solar = Vec3::from_array(m.rgb.as_ref().ok_or("need RGB reference")?.solar_irradiance);
    let floor = solar.max(Vec3::splat(1e-8)) * 1e-10;
    let rows: Vec<_> = scenes
        .par_iter()
        .map(|scene| -> Result<serde_json::Value> {
            let number = |k: &str| scene[k].as_f64().unwrap() as f32;
            let h = number("altitude_km");
            let e = number("sun_elevation_deg").to_radians();
            let sun = Vec3::new(e.cos(), 0.0, e.sin());
            let g = m.geometry;
            let sample = |v: Vec3| {
                let state = State {
                    altitude_km: h,
                    mu: v.z,
                    mu_s: sun.z,
                    nu: v.dot(sun),
                    ground: g.hits_ground(h, v.z),
                };
                let Some((s, _)) = g.atmosphere_entry(state) else {
                    return Vec3::ZERO;
                };
                let stencil = ReferenceStencil::new(g, &m.config, s, Some(&heights));
                Vec3::from_array(std::array::from_fn(|k| {
                    stencil.sample_with(|i| channels[k][i])
                }))
            };
            let sun_weight = |v: Vec3| {
                let d = 0.0004 + (1.0 - v.dot(sun)).max(0.0);
                0.01 / (0.01 + d * d)
            };
            let count = sh::count(a.degree);
            let mut coefficients = [vec![Vec3::ZERO; count], vec![Vec3::ZERO; count]];
            let mut correction = coefficients.clone();
            let mut basis = vec![0.0; count];
            let mut project = |v: Vec3, w: f32| {
                let l = sample(v);
                let log = Vec3::from_array((l.max(floor) / solar).to_array().map(f32::ln));
                sh::cosine_sh(v, a.degree, &mut basis);
                for (mode, light) in [l, log].into_iter().enumerate() {
                    for j in 0..count {
                        let term = light * (basis[j] * w) - correction[mode][j];
                        let next = coefficients[mode][j] + term;
                        correction[mode][j] = (next - coefficients[mode][j]) - term;
                        coefficients[mode][j] = next;
                    }
                }
            };
            for &(v, w) in &sphere {
                let v = quadrature::rotate(v, sun);
                project(v, w * sun_weight(v));
            }
            let hor = g.horizon(h);
            let mut boundaries = vec![-1.0, hor, 1.0];
            if h > g.top_height() {
                boundaries.insert(2, -(1.0 - (g.top / (g.bottom + h)).powi(2)).max(0.0).sqrt());
            }
            for pair in boundaries.windows(2) {
                for &(u, w) in &radial {
                    let z = pair[0] + (pair[1] - pair[0]) * u;
                    let r = (1.0 - z * z).max(0.0).sqrt();
                    for j in 0..a.angular_phi {
                        let phi = 2.0 * PI * (j as f32 + 0.5) / a.angular_phi as f32;
                        let v = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                        project(
                            v,
                            w * (pair[1] - pair[0]) * 2.0 * PI / a.angular_phi as f32
                                * (1.0 - sun_weight(v)),
                        );
                    }
                }
            }
            let name = scene["name"].as_str().unwrap();
            let width = number("width") as usize;
            let height = number("height") as usize;
            let yaw = number("yaw").to_radians();
            let pitch = number("pitch").to_radians();
            let forward = Vec3::new(
                pitch.cos() * yaw.cos(),
                pitch.cos() * yaw.sin(),
                pitch.sin(),
            );
            let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
            let up = right.cross(forward);
            // The camera's screen y grows downward; up above is the down vector.
            let tangent = (number("horizontal_fov").to_radians() * 0.5).tan();
            let mut truth = Vec::new();
            let mut outputs = vec![Vec::<[f32; 4]>::new(); 2 * degrees.len()];
            for y in 0..height {
                for x in 0..width {
                    let dx = (2.0 * (x as f32 + 0.5) / width as f32 - 1.0) * tangent;
                    let dy =
                        (2.0 * (y as f32 + 0.5) / height as f32 - 1.0) * tangent * height as f32
                            / width as f32;
                    let v = (forward + right * dx + up * dy).normalize();
                    truth.push(sample(v).extend(1.0).to_array());
                    sh::cosine_sh(v, a.degree, &mut basis);
                    for mode in 0..2 {
                        let mut value = Vec3::ZERO;
                        let mut begin = 0;
                        for (di, &degree) in degrees.iter().enumerate() {
                            let end = sh::count(degree);
                            for j in begin..end {
                                value += coefficients[mode][j] * basis[j];
                            }
                            begin = end;
                            let decoded = if mode == 0 {
                                value
                            } else {
                                Vec3::from_array(
                                    value.to_array().map(|x| x.clamp(-80.0, 40.0).exp()),
                                ) * solar
                            };
                            // Retain negative linear SH values so the audit can count ringing.
                            outputs[mode * degrees.len() + di].push(decoded.extend(1.0).to_array());
                        }
                    }
                }
            }
            fs::write(
                a.out.join(format!("{name}_rgb_teacher.f32")),
                bytemuck::cast_slice(&truth),
            )?;
            for (mode, label) in ["linear", "log"].iter().enumerate() {
                for (di, degree) in degrees.iter().enumerate() {
                    fs::write(
                        a.out.join(format!("{name}_{label}_{degree}.f32")),
                        bytemuck::cast_slice(&outputs[mode * degrees.len() + di]),
                    )?;
                }
            }
            eprintln!(
                "{name}: local SH <= {}, {:.1} s",
                a.degree,
                start.elapsed().as_secs_f32()
            );
            Ok(serde_json::json!({"scene":name,"altitude_km":h,"sun_elevation_deg":e.to_degrees()}))
        })
        .collect::<Result<_>>()?;
    let budgets:Vec<_>=degrees.iter().map(|&d|serde_json::json!({"degree":d,"coefficients":sh::count(d),
        "rgb_f16_bytes_80x193":80*193*(sh::count(d)*3*2+3*4),"includes_angular_quantization_error":false})).collect();
    fs::write(
        a.out.join("audit.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"full_radiance_local_sh_audit_v1","source":a.source,"model":m.model_fingerprint_fnv1a64,
            "degree":a.degree,"degrees":degrees,"angular_quadrature":[a.angular_mu,a.angular_phi],"seconds":start.elapsed().as_secs_f32(),
            "scenes":rows,"budgets":budgets,"includes_single_scattering":true,"includes_visible_sun_disk":false,
            "limitations":["Exact observer height/Sun projection: no inter-state interpolation or coefficient quantization; optimistic feasibility experiment, not a packed LUT", "RGB-grid interpolation is the local teacher; also compare its difference from bandwise spectral interpolation", "Space cases project the observer's whole sphere directly; a production top-boundary chart would require a separate validation"]
        }))?,
    )?;
    Ok(())
}
