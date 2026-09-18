//! CPU-only mapping pilot from the frozen-reference-derived directional source.
//! This isolates interpolation; it does not re-solve transport or use a GPU.
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use sky_atmosphere_lut::{
    Result, asset::fingerprint, four_wave::CpuSource, mapping::State, model::Model,
    reference_mapping,
};
use std::{f32::consts::PI, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/four_wave_source_v1")]
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// New physical contexts for a check after fitting, never used as training.
    #[arg(long)]
    shifted_validation: bool,
    #[arg(long, default_value_t = 1)]
    sample_multiplier: usize,
}

#[derive(Serialize)]
struct Curve {
    // Fixed coordinates; the named axis is replaced by each sample.
    pose: [f32; 4], // height km, solar elevation deg, theta deg, cone deg
    split: &'static str,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() || !(1..=4).contains(&a.sample_multiplier) {
        return Err("choose a new output directory and sample multiplier 1..=4".into());
    }
    let source = CpuSource::open(&a.source)?;
    let r = &source.resource;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != r.model {
        return Err("model mismatch".into());
    }
    let indices = r.wavelengths_nm.map(|nm| {
        model
            .bands
            .iter()
            .position(|b| b.info.center_nm == nm)
            .unwrap()
    });
    let models = indices.map(|i| model.bands[i].clone());
    let aux = r.read(&a.source, "aux.f32")?;
    let scales: Vec<[f32; 4]> = aux
        .chunks_exact(16)
        .take(r.heights.len() * r.sun_count)
        .map(|b| {
            std::array::from_fn(|k| {
                // SH coefficient zero is sqrt(4*pi) times angular mean radiance.
                (f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap()) / (4.0 * PI).sqrt())
                    .max(1e-30)
                    .ln()
            })
        })
        .collect();
    fs::create_dir_all(&a.out)?;
    let start = Instant::now();
    let mut datasets = Vec::new();
    for axis in ["height", "solar", "phase", "cone"] {
        let mut poses = Vec::new();
        let heights: &[f32] = match axis {
            "height" => &[0.0],
            "solar" => &[
                0.002, 0.05, 0.2, 0.7, 2.0, 5.0, 12.0, 24.0, 34.0, 45.0, 80.0, 108.0,
            ],
            _ => &[0.002, 0.2, 1.0, 2.0, 5.0, 12.0, 24.0, 34.0],
        };
        let suns: &[f32] = if axis == "solar" {
            &[0.0]
        } else {
            &[
                -24.0, -15.0, -10.0, -7.0, -6.0, -3.0, 0.0, 5.0, 20.0, 47.0, 70.0, 85.0, 89.5,
            ]
        };
        let phases: &[f32] = match axis {
            "phase" => &[0.0],
            "cone" => &[3.0, 12.0, 30.0, 60.0, 90.0, 120.0, 160.0],
            _ => &[3.0, 30.0, 90.0, 150.0],
        };
        let cones: &[f32] = if axis == "cone" {
            &[0.0]
        } else {
            &[0.0, 60.0, 120.0, 180.0]
        };
        for &h in heights {
            for &e in suns {
                for &t in phases {
                    for &c in cones {
                        poses.push(if a.shifted_validation {
                            [
                                if h < 0.01 {
                                    h * 0.73
                                } else {
                                    h * 1.013 + 0.007
                                },
                                e * 0.971 + 0.27,
                                (t * 0.963 + 1.7).min(179.9),
                                (c * 0.973 + 1.1).min(179.9),
                            ]
                        } else {
                            [h, e, t, c]
                        });
                    }
                }
            }
        }
        let n = a.sample_multiplier
            * match axis {
                "solar" => 1024,
                "height" => 768,
                _ => 512,
            }
            + 1;
        let rows: Vec<_> = poses
            .par_iter()
            .enumerate()
            .map(|(ci, &pose)| {
                let mut x = Vec::with_capacity(n);
                let mut values = Vec::with_capacity(n * 8);
                for i in 0..n {
                    let u = i as f32 / (n - 1) as f32;
                    let [mut h, e, t, cone] = pose;
                    let mut mu_s = e.to_radians().sin();
                    let mut nu = t.to_radians().cos();
                    let mut cp = cone.to_radians().cos();
                    let physical = match axis {
                        "height" => {
                            h = reference_mapping::height(r.geometry, u);
                            h
                        }
                        "solar" => {
                            mu_s = reference_mapping::solar_cosine(r.geometry, h, u);
                            mu_s
                        }
                        "phase" => {
                            nu = (PI * u).cos();
                            (1.0 - nu) * 0.5
                        }
                        _ => {
                            cp = (PI * u).cos();
                            (1.0 - cp) * 0.5
                        }
                    };
                    let mu = (mu_s * nu
                        + ((1.0 - mu_s * mu_s) * (1.0 - nu * nu)).max(0.0).sqrt() * cp)
                        .clamp(-1.0, 1.0);
                    let point = State {
                        altitude_km: h,
                        mu,
                        mu_s,
                        nu,
                        ground: r.geometry.hits_ground(h, mu),
                    };
                    let hi = r
                        .heights
                        .partition_point(|&v| v <= h)
                        .clamp(1, r.heights.len() - 1);
                    let lo = hi - 1;
                    let th =
                        ((h - r.heights[lo]) / (r.heights[hi] - r.heights[lo])).clamp(0.0, 1.0);
                    let means = [lo, hi].map(|layer| {
                        let z = reference_mapping::solar_coord(r.geometry, r.heights[layer], mu_s)
                            * (r.sun_count - 1) as f32;
                        let si = (z as usize).min(r.sun_count - 2);
                        let ts = z - si as f32;
                        std::array::from_fn::<_, 4, _>(|k| {
                            scales[layer * r.sun_count + si][k] * (1.0 - ts)
                                + scales[layer * r.sun_count + si + 1][k] * ts
                        })
                    });
                    let log_mean: [f32; 4] =
                        std::array::from_fn(|k| means[0][k] * (1.0 - th) + means[1][k] * th);
                    let light = source.source(point, &models).to_array();
                    let shape: [f32; 4] = std::array::from_fn(|k| {
                        let sigma: f32 = models[k].coefficients(h).scattering.iter().sum();
                        light[k] / (sigma * log_mean[k].exp()).max(1e-30)
                    });
                    x.push(physical);
                    values.extend(shape);
                    values.extend(log_mean);
                }
                // Split entire contexts, never alternating samples on a fitted curve.
                let split = if a.shifted_validation
                    || ((ci as u32).wrapping_mul(2654435761) >> 16) % 5 == 0
                {
                    "validation"
                } else {
                    "train"
                };
                (Curve { pose, split }, x, values)
            })
            .collect();
        let x: Vec<f32> = rows.iter().flat_map(|r| r.1.iter().copied()).collect();
        let values: Vec<f32> = rows.iter().flat_map(|r| r.2.iter().copied()).collect();
        if x.iter().chain(&values).any(|v| !v.is_finite()) {
            return Err("nonfinite source curve".into());
        }
        fs::write(
            a.out.join(format!("{axis}.x.f32")),
            bytemuck::cast_slice(&x),
        )?;
        fs::write(
            a.out.join(format!("{axis}.values.f32")),
            bytemuck::cast_slice(&values),
        )?;
        datasets.push(serde_json::json!({"axis":axis,"shape":[poses.len(),n,8],
            "curves":rows.into_iter().map(|r| r.0).collect::<Vec<_>>()}));
        eprintln!(
            "{axis}: {} curves, {} samples, {:.1}s",
            poses.len(),
            poses.len() * n,
            start.elapsed().as_secs_f32()
        );
    }
    fs::write(
        a.out.join("curves.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"hybrid_mapping_source_curves_cpu_v1", "source":a.source,
            "model":r.model, "source_files":r.files, "wavelengths_nm":r.wavelengths_nm,
            "geometry":r.geometry, "datasets":datasets,
            "shifted_validation":a.shifted_validation, "sample_multiplier":a.sample_multiplier,
            "value_layout":["normalized_source_4wave","log_incident_mean_4wave"],
            "limitations":["teacher is reference-derived SH16/f16, not raw reference convolution",
                "height/Sun interpolation and source truncation of the teacher are inherited",
                "isolated-axis source interpolation, not end-to-end sky/transport error"],
            "seconds":start.elapsed().as_secs_f32()
        }))?,
    )?;
    Ok(())
}
