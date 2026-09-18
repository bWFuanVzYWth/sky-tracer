//! Audit hybrid Sun-transmittance and phase-table allocation without a GPU.
use clap::Parser;
use glam::Vec4;
use half::f16;
use rayon::prelude::*;
use sky_atmosphere_lut::{Result, mapping::Geometry, model::Model, reference_mapping};
use std::{fs, path::PathBuf, time::Instant};
static ANCHOR_INDICES: std::sync::OnceLock<[usize; 7]> = std::sync::OnceLock::new();

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    export_curves_only: bool,
    #[arg(long, num_args = 7)]
    anchor_indices: Vec<usize>,
}
fn altitude(h: f32, mu: f32, d: f32, radius: f32) -> f32 {
    let r = radius + h;
    let dh = d * (d + 2.0 * r * mu);
    (h + dh / ((r * r + dh).max(0.0).sqrt() + r)).max(0.0)
}
fn tau(g: Geometry, profile: &[(f32, Vec4)], h: f32, mu: f32, steps: usize) -> Vec4 {
    let dt = g.distance(h, mu, false) / steps as f32;
    let mut sum = Vec4::ZERO;
    for i in 0..steps {
        let hp = altitude(h, mu, (i as f32 + 0.5) * dt, g.bottom);
        let hi = profile
            .partition_point(|p| p.0 <= hp)
            .clamp(1, profile.len() - 1);
        let (a, b) = (profile[hi - 1], profile[hi]);
        let t = ((hp - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
        sum += a.1.lerp(b.1, t) * dt;
    }
    sum
}
fn angle(u: f32, two_sided: bool) -> f32 {
    let a = u * u * u;
    if two_sided {
        a / (a + (1.0 - u).powi(3))
    } else {
        a
    }
}
fn coord(x: f32, two_sided: bool) -> f32 {
    let a = x.cbrt();
    if two_sided {
        a / (a + (1.0 - x).cbrt())
    } else {
        a
    }
}
fn height_nodes(g: Geometry, n: usize, kind: u32) -> Vec<f32> {
    let h = [0.0_f32, 1.0, 2.0, 11.0, 12.0, 35.0, g.top_height()];
    let bins = *ANCHOR_INDICES
        .get()
        .unwrap_or(&[0, 24, 56, 112, 128, 176, 255]);
    (0..n)
        .map(|i| {
            let u = i as f32 / (n - 1) as f32;
            if kind == 0 {
                reference_mapping::height(g, u)
            } else if kind == 1 {
                g.top_height() * u * u
            } else {
                let x = u * 255.0;
                let j = bins
                    .partition_point(|&b| b as f32 <= x)
                    .clamp(1, bins.len() - 1)
                    - 1;
                let t = (x - bins[j] as f32) / (bins[j + 1] - bins[j]) as f32;
                (h[j].sqrt() * (1.0 - t) + h[j + 1].sqrt() * t).powi(2)
            }
        })
        .collect()
}
fn height_coord(g: Geometry, h: f32, kind: u32) -> f32 {
    if kind == 0 {
        reference_mapping::height_coord(g, h)
    } else if kind == 1 {
        (h / g.top_height()).sqrt()
    } else {
        let anchors = [0.0_f32, 1.0, 2.0, 11.0, 12.0, 35.0, g.top_height()];
        let bins = *ANCHOR_INDICES
            .get()
            .unwrap_or(&[0, 24, 56, 112, 128, 176, 255]);
        let j = anchors
            .partition_point(|&v| v <= h)
            .clamp(1, anchors.len() - 1)
            - 1;
        (bins[j] as f32
            + (bins[j + 1] - bins[j]) as f32 * (h.sqrt() - anchors[j].sqrt())
                / (anchors[j + 1].sqrt() - anchors[j].sqrt()))
            / 255.0
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    if !a.anchor_indices.is_empty() {
        let bins: [usize; 7] = a
            .anchor_indices
            .clone()
            .try_into()
            .map_err(|_| "need seven anchor indices")?;
        if bins[0] != 0 || bins[6] != 255 || bins.windows(2).any(|w| w[0] >= w[1]) {
            return Err("anchor indices must increase from 0 to 255".into());
        }
        ANCHOR_INDICES.set(bins).unwrap();
    }
    if a.out.exists() {
        return Err("choose a new output directory".into());
    }
    fs::create_dir_all(&a.out)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let m = Model::from_scene(&scene)?;
    let g = m.geometry;
    let indices = [7, 13, 20, 27];
    let profile: Vec<_> = m.bands[7]
        .profile
        .iter()
        .map(|p| {
            (
                p.0,
                Vec4::from_array(indices.map(|k| m.bands[k].coefficients(p.0).extinction)),
            )
        })
        .collect();
    if a.export_curves_only {
        let anchors = [0.0_f32, 1.0, 2.0, 11.0, 12.0, 35.0, g.top_height()];
        let curves: Vec<[f32; 6]> = (0..6 * 129 * 129)
            .into_par_iter()
            .map(|i| {
                let segment = i / (129 * 129);
                let iy = (i / 129) % 129;
                let ix = i % 129;
                let u = iy as f32 / 128.0;
                let v = ix as f32 / 128.0;
                let h =
                    (anchors[segment].sqrt() * (1.0 - u) + anchors[segment + 1].sqrt() * u).powi(2);
                let mu = g.horizon(h) + (1.0 - g.horizon(h)) * v * v * v;
                let t = tau(g, &profile, h, mu, 2048).to_array();
                [h, v, t[0], t[1], t[2], t[3]]
            })
            .collect();
        fs::write(
            a.out.join("optical_curves.f32"),
            bytemuck::cast_slice(&curves),
        )?;
        return Ok(());
    }
    let mut queries = Vec::new();
    let mut probe_heights = vec![
        0.0, 0.002, 0.02, 0.2, 1.0, 2.0, 11.0, 12.0, 24.0, 35.0, 60.0, 80.0, 108.0, 119.0,
    ];
    for i in 0..96 {
        probe_heights.push(reference_mapping::height(g, (i as f32 + 0.419) / 96.0));
    }
    for h in [1.0, 2.0, 11.0, 12.0, 35.0] {
        for offset in [-0.03, -0.001, 0.001, 0.03] {
            probe_heights.push(h + offset);
        }
    }
    for h in probe_heights {
        let hor = g.horizon(h);
        for i in 0..257 {
            let u = (i as f32 + 0.371) / 257.0;
            queries.push([h, hor + (1.0 - hor) * angle(u, true)]);
        }
    }
    let oracle: Vec<[f32; 10]> = queries
        .par_iter()
        .map(|&[h, mu]| {
            let low = tau(g, &profile, h, mu, 4096).to_array();
            let high = tau(g, &profile, h, mu, 8192).to_array();
            [
                h, mu, low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
            ]
        })
        .collect();
    fs::write(
        a.out.join("optical_queries.f32"),
        bytemuck::cast_slice(&oracle),
    )?;
    let mut variants = Vec::new();
    for (nh, nm, two_sided, height_kind) in [
        (256, 1024, true, 0),
        (256, 512, true, 0),
        (128, 1024, true, 0),
        (128, 512, true, 0),
        (256, 512, false, 0),
        (128, 512, false, 0),
        (256, 512, true, 1),
        (256, 512, true, 2),
        (256, 512, false, 2),
        (256, 1024, false, 2),
    ] {
        let now = Instant::now();
        let name = format!(
            "h{nh}_m{nm}_{}_{}",
            if two_sided { "two" } else { "one" },
            ["current", "sqrt", "anchors"][height_kind as usize]
        );
        let heights = height_nodes(g, nh, height_kind);
        let table: Vec<_> = (0..nh * nm)
            .into_par_iter()
            .map(|i| {
                let h = heights[i / nm];
                let mu = g.horizon(h)
                    + (1.0 - g.horizon(h)) * angle((i % nm) as f32 / (nm - 1) as f32, two_sided);
                tau(g, &profile, h, mu, 1024).min(Vec4::splat(80.0))
            })
            .collect();
        for quantized in [false, true] {
            let read = |i: usize| {
                if quantized {
                    Vec4::from_array(table[i].to_array().map(|v| f16::from_f32(v).to_f32()))
                } else {
                    table[i]
                }
            };
            let output: Vec<[f32; 4]> = queries
                .iter()
                .map(|&[h, mu]| {
                    let yh = height_coord(g, h, height_kind);
                    let y = yh * (nh - 1) as f32;
                    let x = coord(
                        ((mu - g.horizon(h)) / (1.0 - g.horizon(h))).clamp(0.0, 1.0),
                        two_sided,
                    ) * (nm - 1) as f32;
                    let ix = (x as usize).min(nm - 2);
                    let iy = (y as usize).min(nh - 2);
                    let row0 = read(iy * nm + ix).lerp(read(iy * nm + ix + 1), x - ix as f32);
                    let row1 =
                        read((iy + 1) * nm + ix).lerp(read((iy + 1) * nm + ix + 1), x - ix as f32);
                    row0.lerp(row1, y - iy as f32).to_array()
                })
                .collect();
            let id = format!("{name}_{}", if quantized { "f16" } else { "f32" });
            fs::write(
                a.out.join(format!("{id}.f32")),
                bytemuck::cast_slice(&output),
            )?;
            variants.push(serde_json::json!({"id":id,"dims":[nh,nm],"bytes":nh*nm*4*if quantized {2} else {4},"quantized":quantized}));
        }
        eprintln!("{name}: {:.2}s", now.elapsed().as_secs_f32());
    }
    let mut phase_rows = Vec::new();
    for n in [256, 512, 1024] {
        for band in indices {
            for species in 1..5 {
                let b = &m.bands[band];
                let nodes: Vec<_> = (0..n)
                    .map(|i| {
                        let u = (i as f32 + 0.5) / n as f32;
                        if n == 1024 {
                            b.phase[(species - 1) * 1024 + i]
                        } else {
                            b.phases(1.0 - 2.0 * u * u * u)[species]
                        }
                    })
                    .collect();
                let mut errors = Vec::new();
                let mut truth_mass = 0.0;
                let mut candidate_mass = 0.0;
                for i in 0..8192 {
                    let u = (i as f32 + 0.371) / 8192.0;
                    let mu = 1.0 - 2.0 * u * u * u;
                    let truth = b.phases(mu)[species];
                    let x = (((1.0 - mu) * 0.5).max(0.0).cbrt() * n as f32 - 0.5)
                        .clamp(0.0, (n - 1) as f32);
                    let j = (x as usize).min(n - 2);
                    let t = x - j as f32;
                    let value = nodes[j] * (1.0 - t) + nodes[j + 1] * t;
                    errors.push((value / truth.max(1e-30) - 1.0).abs());
                    truth_mass += truth * u * u;
                    candidate_mass += value * u * u;
                }
                errors.sort_by(f32::total_cmp);
                phase_rows.push(serde_json::json!({"bins":n,"nm":b.info.center_nm,"species":species,"p95_percent":100.0*errors[errors.len()*95/100],"max_percent":100.0*errors[errors.len()-1],"mass_error_percent":100.0*(candidate_mass/truth_mass-1.0)}));
            }
        }
    }
    fs::write(
        a.out.join("audit.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"hybrid_aux_allocation_cpu_v1","queries":queries.len(),"variants":variants,"phase":phase_rows,
            "oracle_steps":[4096,8192],"table_steps":1024,"wavelengths_nm":indices.map(|i|m.bands[i].info.center_nm),
            "limitations":["CPU f32 physical optical integration; not bit-identical to GPU FMA/filtering", "phase resampling compares to the existing input table, not the original measured continuous phase", "solar transmission probes are stratified, not a global error bound"]
        }))?,
    )?;
    Ok(())
}
