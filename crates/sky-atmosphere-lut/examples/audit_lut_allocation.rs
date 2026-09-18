//! CPU-only one-axis restorations; these are diagnostics, not 16 MB assets.
use rayon::prelude::*;
use serde::Deserialize;
use sky_atmosphere_lut::{
    Result,
    asset::Manifest,
    config::BakeConfig,
    mapping::State,
    reference_mapping::{self as mapping, ReferenceStencil},
    synthesis::SamplePoint,
};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

#[derive(Deserialize)]
struct Color {
    rgb: [f32; 3],
    spectral: Option<[f32; 3]>,
}
fn norm(v: [f32; 3]) -> f32 {
    v.into_iter().map(|x| x * x).sum::<f32>().sqrt()
}
fn stats(reference: &[[f32; 3]], values: &[[f32; 3]], indices: &[usize]) -> serde_json::Value {
    let mut errors: Vec<_> = indices
        .iter()
        .filter_map(|&i| {
            let n = norm(reference[i]);
            (n > 1e-8).then(|| norm(std::array::from_fn(|c| values[i][c] - reference[i][c])) / n)
        })
        .collect();
    errors.sort_unstable_by(f32::total_cmp);
    let n = errors.len();
    serde_json::json!({"count":n,"p50":errors.get(n/2),"p95":errors.get(n*95/100),
        "p99":errors.get(n*99/100),"max":errors.last()})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let source = Path::new(args.get(1).ok_or("RGB source")?);
    let candidate = Path::new(args.get(2).ok_or("candidate directory")?);
    let output = Path::new(args.get(3).ok_or("output directory")?);
    if output.exists() {
        return Err("output already exists".into());
    }
    let m = Manifest::open(source)?;
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(candidate.join("candidate.json"))?)?;
    let original: BakeConfig = serde_json::from_value(meta["source_config"].clone())?;
    if original != m.config || meta["source_model"] != m.model_fingerprint_fnv1a64 {
        return Err("source mismatch".into());
    }
    let checksums: [String; 3] = serde_json::from_value(meta["source_channel_checksums"].clone())?;
    if checksums
        != m.rgb
            .as_ref()
            .ok_or("RGB source required")?
            .channel_checksums
    {
        return Err("source checksums mismatch".into());
    }
    let base: BakeConfig = serde_json::from_value(meta["config"].clone())?;
    let sparse_heights: Vec<usize> = serde_json::from_value(meta["height_indices"].clone())?;
    let [rh, rv, rs, rp] = m.config.scattering;
    let offset = 8 + (m.config.optical_depth.iter().product::<usize>() * 4) as u64;
    let mut raw = Vec::new();
    for c in 0..3 {
        let mut file = fs::File::open(source.join(format!("channel_{c}.bin")))?;
        if file.metadata()?.len() != offset + (m.config.scattering_len() * 4) as u64 {
            return Err("wrong source length".into());
        }
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"SKYRGB01" {
            return Err("wrong RGB magic".into());
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; m.config.scattering_len() * 4];
        file.read_exact(&mut bytes)?;
        raw.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect::<Vec<_>>(),
        );
    }
    let queries: Vec<SamplePoint> =
        serde_json::from_slice(&fs::read("out/lut_v6_packed_cpu.queries.json")?)?;
    let colors: Vec<Color> =
        serde_json::from_slice(&fs::read("out/lut_v6_packed_cpu.colors.json")?)?;
    if queries.len() != colors.len() || queries.len() < 66472 {
        return Err("cached queries mismatch".into());
    }
    let rgb: Vec<_> = colors.iter().map(|c| c.rgb).collect();
    let spectral: Vec<_> = colors
        .iter()
        .map(|c| c.spectral.expect("spectral teacher required"))
        .collect();
    let states: Vec<_> = queries
        .iter()
        .map(|p| {
            let [h, mu, mu_s, nu] = p.packed().unwrap();
            m.geometry
                .atmosphere_entry(State {
                    altitude_km: h,
                    mu,
                    mu_s,
                    nu,
                    ground: m.geometry.hits_ground(h, mu),
                })
                .map(|x| x.0)
        })
        .collect();
    let names = [
        "noon",
        "afternoon",
        "sunset",
        "blue_hour",
        "aircraft_twilight",
        "stratosphere_shadow",
        "atmosphere_edge",
        "orbit",
    ];
    let mut regions: Vec<(String, Vec<usize>)> = names
        .iter()
        .enumerate()
        .map(|(j, n)| (n.to_string(), (j * 8309..(j + 1) * 8309).collect()))
        .collect();
    regions.push(("random".into(), (66472..queries.len()).collect()));
    regions.push(("all".into(), (0..queries.len()).collect()));
    regions.push((
        "noon_sun_0.27_to_2_deg".into(),
        (0..8309)
            .filter(|&i| {
                let a = queries[i].packed().unwrap()[3]
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                (0.27..2.0).contains(&a)
            })
            .collect(),
    ));
    fs::create_dir_all(output)?;
    let mut reports = Vec::new();
    for mask in 0..9 {
        let mut config = base.clone();
        let heights = if mask < 8 && mask & 1 != 0 {
            (0..rh).collect::<Vec<_>>()
        } else {
            sparse_heights.clone()
        };
        config.scattering[0] = heights.len();
        config.scattering_altitudes_km = heights
            .iter()
            .map(|&i| m.config.scattering_altitudes_km[i])
            .collect();
        if mask < 8 && mask & 2 != 0 {
            config.scattering[2] = rs;
        }
        if mask < 8 && mask & 4 != 0 {
            config.scattering[3] = rp;
        }
        if mask == 8 {
            config.scattering[1] = 16;
        }
        let [_, v, s, p] = config.scattering;
        let source_index = |i: usize| {
            let hi = heights[i / (v * s * p)];
            let vi = i / (s * p) % v;
            let si = i / p % s;
            let pi = i % p;
            let ng = (v / 4).max(2);
            let old_ng = (rv / 4).max(2);
            let vc = if vi < ng {
                vi as f32 * (old_ng - 1) as f32 / (ng - 1) as f32
            } else {
                old_ng as f32 + (vi - ng) as f32 * (rv - old_ng - 1) as f32 / (v - ng - 1) as f32
            };
            let vl = vc.floor() as usize;
            let vu = (vl + 1).min(rv - 1);
            let t = vc - vl as f32;
            let idx = |view| {
                ((hi * rv + view) * rs + si * (rs - 1) / (s - 1)) * rp + pi * (rp - 1) / (p - 1)
            };
            (idx(vl), idx(vu), t)
        };
        let start = Instant::now();
        let values: Vec<[f32; 3]> = states
            .par_iter()
            .map(|state| {
                state.map_or([0.0; 3], |state| {
                    let stencil = ReferenceStencil::new(
                        m.geometry,
                        &config,
                        state,
                        Some(&config.scattering_altitudes_km),
                    );
                    std::array::from_fn(|c| {
                        stencil.sample_with(|i| {
                            let (lo, hi, t) = source_index(i);
                            let a = raw[c][lo];
                            a + (raw[c][hi] - a) * t
                        })
                    })
                })
            })
            .collect();
        if values.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite interpolation".into());
        }
        let mut cases = serde_json::Map::new();
        for (name, indices) in &regions {
            cases.insert(name.clone(),serde_json::json!({"vs_rgb":stats(&rgb,&values,indices),"vs_spectral":stats(&spectral,&values,indices)}));
        }
        let report = serde_json::json!({"mask":mask,"shape":config.scattering,"f32_nodes":config.scattering_len(),
            "seconds":start.elapsed().as_secs_f32(),"cases":cases});
        println!(
            "mask {mask}, {:?}: {}",
            config.scattering, report["cases"]["all"]
        );
        fs::write(
            output.join(format!("queries_{mask}.f32")),
            values
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        reports.push(report);
    }
    let mut charts = Vec::new();
    for h in [0.0_f32, 0.2, 12.0, 30.0, 120.0] {
        let solar: Vec<_> = (0..base.scattering[2])
            .map(|i| {
                mapping::solar_cosine(m.geometry, h, i as f32 / (base.scattering[2] - 1) as f32)
                    .asin()
                    .to_degrees()
            })
            .collect();
        let mut phase = Vec::new();
        for e in [-6.0_f32, 0.0, 47.0, 85.0, 90.0] {
            for ground in [false, true] {
                let state = State {
                    altitude_km: h,
                    mu: 0.0,
                    mu_s: e.to_radians().sin(),
                    nu: 0.0,
                    ground,
                };
                let angles: Vec<_> = (0..base.scattering[3])
                    .map(|i| {
                        mapping::phase_cosine(
                            m.geometry,
                            state,
                            i as f32 / (base.scattering[3] - 1) as f32,
                        )
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees()
                    })
                    .collect();
                phase.push(serde_json::json!({"sun_deg":e,"ground":ground,"theta_deg":angles}));
            }
        }
        charts.push(serde_json::json!({"altitude_km":h,"horizon_deg":m.geometry.horizon(h).asin().to_degrees(),"solar_deg":solar,"phase":phase}));
    }
    fs::write(
        output.join("allocation.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
        "cpu_only":true,"diagnostic_only_not_budgeted_assets":true,"reference_source":source,
        "candidate":candidate,"height_nodes_km":base.scattering_altitudes_km,
        "mask_note":"bits 0/1/2 restore height/Sun/phase to reference; 8 reduces view to 16",
        "reports":reports,"charts":charts}))?,
    )?;
    Ok(())
}
