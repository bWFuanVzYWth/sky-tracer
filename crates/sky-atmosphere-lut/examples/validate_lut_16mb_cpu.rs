//! CPU-only audit of compression prototypes using frozen non-grid queries.
use half::f16;
use rayon::prelude::*;
use serde::Deserialize;
use sky_atmosphere_lut::{
    Result,
    asset::Manifest,
    config::BakeConfig,
    mapping::State,
    reference_mapping::{ReferenceStencil, radius_nodes},
    synthesis::SamplePoint,
};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Candidate {
    source_config: BakeConfig,
    source_channel_checksums: [String; 3],
    source_model: String,
    shape: [usize; 4],
    height_groups: usize,
    sun_groups: usize,
    groups: usize,
    rank: usize,
    coefficient: String,
    transform: String,
    transform_floor: f32,
    gpu_payload_budget_bytes: usize,
}
struct Pca {
    m: Candidate,
    coefficient: Vec<f32>,
    basis: Vec<f32>,
    mean: Vec<f32>,
    scale: Vec<f32>,
    cs: Vec<f32>,
}
#[derive(Deserialize)]
struct JointMetadata {
    shape: [usize; 4],
    rank: usize,
    height_groups: usize,
    sun_groups: usize,
    groups: usize,
    transform: String,
    transform_floor: f32,
    mean_storage: String,
    gpu_payload_budget_bytes: usize,
}
struct Joint {
    m: JointMetadata,
    coefficient: [Vec<f32>; 2],
    basis: [Vec<f32>; 2],
    mean: [Vec<f32>; 2],
    scale: [Vec<f32>; 2],
}
impl Joint {
    fn read(dir: &Path) -> Result<Self> {
        let m: JointMetadata = serde_json::from_slice(&fs::read(dir.join("candidate.json"))?)?;
        let mut result = Self {
            m,
            coefficient: std::array::from_fn(|_| Vec::new()),
            basis: std::array::from_fn(|_| Vec::new()),
            mean: std::array::from_fn(|_| Vec::new()),
            scale: std::array::from_fn(|_| Vec::new()),
        };
        let [h, v, s, n] = result.m.shape;
        let ng = (v / 4).max(2);
        if result.m.groups != result.m.height_groups * result.m.sun_groups
            || result.m.gpu_payload_budget_bytes > 16_000_000
        {
            return Err("invalid joint groups or budget".into());
        }
        for (b, views) in [ng, v - ng].into_iter().enumerate() {
            result.coefficient[b] = f16s(&dir.join(format!("coefficients_{b}.f16")))?;
            result.basis[b] = f16s(&dir.join(format!("basis_{b}.f16")))?;
            result.mean[b] = match result.m.mean_storage.as_str() {
                "f16" => f16s(&dir.join(format!("mean_{b}.f16")))?,
                "f32" => f32s(&dir.join(format!("mean_{b}.f32")))?,
                _ => return Err("unknown mean storage".into()),
            };
            result.scale[b] = f32s(&dir.join(format!("scale_{b}.f32")))?;
            let f = views * n * 3;
            if result.coefficient[b].len() != h * s * result.m.rank
                || result.basis[b].len() != result.m.groups * f * result.m.rank
                || result.mean[b].len() != result.m.groups * f
                || result.scale[b].len() != h * s
            {
                return Err("wrong joint factor dimensions".into());
            }
        }
        Ok(result)
    }
    fn fetch(&self, index: usize, channel: usize) -> f32 {
        let [nh, nv, ns, nn] = self.m.shape;
        let h = index / (nv * ns * nn);
        let v = index / (ns * nn) % nv;
        let s = index / nn % ns;
        let p = index % nn;
        let ng = (nv / 4).max(2);
        let branch = usize::from(v >= ng);
        let row = h * ns + s;
        let scale = self.scale[branch][row];
        if scale == 0.0 {
            return 0.0;
        }
        let g = (h * self.m.height_groups / nh) * self.m.sun_groups + s * self.m.sun_groups / ns;
        let views = if branch == 0 { ng } else { nv - ng };
        let feature = (v - branch * ng) * nn * 3 + p * 3 + channel;
        let f = views * nn * 3;
        let rank = self.m.rank;
        let mut z = 0.0;
        for k in 0..rank {
            z += self.coefficient[branch][row * rank + k]
                * self.basis[branch][(g * f + feature) * rank + k];
        }
        z += self.mean[branch][g * f + feature];
        let value = match self.m.transform.as_str() {
            "sqrt" => z.max(0.0).powi(2),
            "log1p" => z.clamp(0.0, 20.0).exp_m1() * self.m.transform_floor,
            "log" => z.min(0.0).exp(),
            _ => panic!("unknown joint transform"),
        };
        value * scale
    }
}
fn f32s(path: &Path) -> Result<Vec<f32>> {
    Ok(fs::read(path)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}
fn f16s(path: &Path) -> Result<Vec<f32>> {
    Ok(fs::read(path)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f16::from_le_bytes(b).to_f32())
        .collect())
}
impl Pca {
    fn read(dir: &Path) -> Result<Self> {
        let m: Candidate = serde_json::from_slice(&fs::read(dir.join("candidate.json"))?)?;
        let coefficient = if m.coefficient == "f16" {
            f16s(&dir.join("coefficients.bin"))?
        } else {
            fs::read(dir.join("coefficients.bin"))?
                .iter()
                .map(|&b| (b as i8) as f32)
                .collect()
        };
        let p = Self {
            m,
            coefficient,
            basis: f16s(&dir.join("basis.f16"))?,
            mean: f32s(&dir.join("mean.f32"))?,
            scale: f32s(&dir.join("row_scale.f32"))?,
            cs: f32s(&dir.join("coefficient_scale.f32"))?,
        };
        let [h, v, s, n] = p.m.shape;
        let rows = h * v * s;
        let f = n * 3;
        let k = p.m.rank;
        let g = p.m.groups;
        if p.coefficient.len() != rows * k
            || p.basis.len() != g * k * f
            || p.mean.len() != g * f
            || p.scale.len() != rows
            || p.cs.len() != g * k
            || p.m.gpu_payload_budget_bytes > 16_000_000
        {
            return Err("invalid PCA dimensions or budget".into());
        }
        Ok(p)
    }
    fn fetch(&self, index: usize, channel: usize) -> f32 {
        let [nh, nv, ns, nn] = self.m.shape;
        let row = index / nn;
        let phase = index % nn;
        let scale = self.scale[row];
        if scale == 0.0 {
            return 0.0;
        }
        let h = row / (nv * ns);
        let v = row / ns % nv;
        let s = row % ns;
        let g = ((h * self.m.height_groups / nh) * self.m.sun_groups + s * self.m.sun_groups / ns)
            * 2
            + usize::from(v >= (nv / 4).max(2));
        let feature = phase * 3 + channel;
        let features = nn * 3;
        let k = self.m.rank;
        let mut z = self.mean[g * features + feature];
        for j in 0..k {
            z += self.coefficient[row * k + j]
                * self.cs[g * k + j]
                * self.basis[(g * k + j) * features + feature];
        }
        let value = match self.m.transform.as_str() {
            "sqrt" => z.max(0.0).powi(2),
            "log1p" => z.min(20.0).exp_m1().max(0.0) * self.m.transform_floor,
            "log" => z.min(0.0).exp(),
            _ => panic!("unknown transform"),
        };
        value * scale
    }
}
#[derive(Deserialize)]
struct ReferenceColor {
    rgb: [f32; 3],
    spectral: Option<[f32; 3]>,
}
fn norm(v: [f32; 3]) -> f32 {
    v.into_iter().map(|v| v * v).sum::<f32>().sqrt()
}
fn statistics(
    reference: &[[f32; 3]],
    candidate: &[[f32; 3]],
    indices: &[usize],
) -> serde_json::Value {
    let mut errors = Vec::new();
    let mut worst = 0;
    let mut max = 0.0;
    let mut sum_e = 0.0;
    let mut sum_r = 0.0;
    let mut absolute = 0.0_f32;
    let mut more_1 = 0;
    let mut more_5 = 0;
    let mut more_10 = 0;
    for &i in indices {
        let n = norm(reference[i]);
        let e = norm(std::array::from_fn(|c| candidate[i][c] - reference[i][c]));
        absolute = absolute.max(e);
        sum_e += e * e;
        sum_r += n * n;
        if n > 1e-8 {
            let r = e / n;
            errors.push(r);
            if r > max {
                max = r;
                worst = i;
            }
            more_1 += usize::from(r > 0.01);
            more_5 += usize::from(r > 0.05);
            more_10 += usize::from(r > 0.10);
        }
    }
    errors.sort_unstable_by(f32::total_cmp);
    let n = errors.len();
    serde_json::json!({"queries":indices.len(),"lit_queries":n,"relative_p50":errors.get(n/2),"relative_p95":errors.get(n*95/100),
        "relative_p99":errors.get(n*99/100),"relative_max":max,"worst_query":worst,"over_1_percent":more_1 as f32/n.max(1) as f32,
        "over_5_percent":more_5 as f32/n.max(1) as f32,"over_10_percent":more_10 as f32/n.max(1) as f32,
        "relative_rmse":(sum_e/sum_r.max(1e-30)).sqrt(),"absolute_max":absolute})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let source = Path::new(args.get(1).ok_or("RGB source")?);
    let dir = Path::new(args.get(2).ok_or("candidate")?);
    let m = Manifest::open(source)?;
    let raw_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("candidate.json"))?)?;
    let pca = if raw_metadata["kind"] == "cpu_local_pca_16mb_probe_v1" {
        Some(Pca::read(dir)?)
    } else {
        None
    };
    let joint = if raw_metadata["kind"] == "cpu_joint_pair_svd_16mb_v1" {
        Some(Joint::read(dir)?)
    } else {
        None
    };
    let mut decoded = Vec::new();
    let config: BakeConfig = if let Some(p) = &pca {
        if p.m.source_config != m.config
            || p.m.source_channel_checksums != m.rgb.as_ref().unwrap().channel_checksums
            || p.m.source_model != m.model_fingerprint_fnv1a64
        {
            return Err("candidate provenance mismatch".into());
        }
        p.m.source_config.clone()
    } else {
        let source_config: BakeConfig =
            serde_json::from_value(raw_metadata["source_config"].clone())?;
        let checksums: [String; 3] =
            serde_json::from_value(raw_metadata["source_channel_checksums"].clone())?;
        if source_config != m.config
            || checksums != m.rgb.as_ref().unwrap().channel_checksums
            || raw_metadata["source_model"] != m.model_fingerprint_fnv1a64
        {
            return Err("candidate provenance mismatch".into());
        }
        let c: BakeConfig = serde_json::from_value(raw_metadata["config"].clone())?;
        c.validate(m.bands.len())?;
        c.validate_top_height(m.geometry.top_height())?;
        if let Some(j) = &joint {
            if j.m.shape != c.scattering {
                return Err("joint shape/config mismatch".into());
            }
        } else {
            let decoded_dir = raw_metadata["cpu_base_dir"].as_str().map_or(dir, Path::new);
            for channel in 0..3 {
                decoded.push(f32s(&decoded_dir.join(format!("decoded_{channel}.f32")))?);
            }
            if decoded.iter().any(|v| v.len() != c.scattering_len()) {
                return Err("wrong decoded table size".into());
            }
            if raw_metadata["kind"] == "cpu_joint_tt_top_16mb_v1" {
                let words = |name: &str| -> Result<Vec<u32>> {
                    Ok(fs::read(dir.join(name))?
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| u32::from_le_bytes(b))
                        .collect())
                };
                let map = words("top_blocks.u32")?;
                let data = words("top_radiance.u32")?;
                let mantissa = raw_metadata["top_mantissa_bits"]
                    .as_u64()
                    .ok_or("missing top precision")? as u32;
                if !(4..=11).contains(&mantissa) {
                    return Err("invalid top precision".into());
                }
                let top_count = c.scattering[1..].iter().product::<usize>();
                if map.len() != top_count.div_ceil(16) {
                    return Err("invalid top block map".into());
                }
                for (channel, values) in decoded.iter_mut().enumerate() {
                    let offset = values.len() - top_count;
                    for i in 0..top_count {
                        let q = sky_atmosphere_lut::packed::decode_block(
                            &data[map[i / 16] as usize..],
                            16,
                            i % 16,
                            channel,
                        )
                        .to_bits()
                            >> 12;
                        values[offset + i] = f32::from_bits(q << (23 - mantissa));
                    }
                }
            }
        }
        c
    };
    let query_path = args
        .get(3)
        .map_or("out/lut_v6_packed_cpu.queries.json", String::as_str);
    let color_path = args
        .get(4)
        .map_or("out/lut_v6_packed_cpu.colors.json", String::as_str);
    let prefix = args.get(5).map_or("", String::as_str);
    let custom_queries = args.get(3).is_some();
    let spectral_only = Path::new(color_path)
        .extension()
        .is_some_and(|x| x == "f32");
    let queries: Vec<SamplePoint> = serde_json::from_slice(&fs::read(query_path)?)?;
    let colors: Vec<ReferenceColor> = if spectral_only {
        let values = f32s(Path::new(color_path))?;
        if values.len() != queries.len() * 3 {
            return Err("wrong reference RGB sample count".into());
        }
        values
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&v| ReferenceColor {
                rgb: v,
                spectral: Some(v),
            })
            .collect()
    } else {
        serde_json::from_slice(&fs::read(color_path)?)?
    };
    if colors.len() != queries.len() {
        return Err("reference query count mismatch".into());
    }
    let heights = radius_nodes(m.geometry, &config);
    let candidate: Vec<[f32; 3]> = queries
        .par_iter()
        .map(|p| {
            let [h, mu, mu_s, nu] = p.packed().unwrap();
            let s = State {
                altitude_km: h,
                mu,
                mu_s,
                nu,
                ground: m.geometry.hits_ground(h, mu),
            };
            if let Some((s, _)) = m.geometry.atmosphere_entry(s) {
                let stencil = ReferenceStencil::new(m.geometry, &config, s, Some(&heights));
                std::array::from_fn(|c| {
                    stencil.sample_with(|i| {
                        if let Some(p) = &pca {
                            p.fetch(i, c)
                        } else if let Some(j) = &joint {
                            j.fetch(i, c)
                        } else {
                            decoded[c][i]
                        }
                    })
                })
            } else {
                [0.0; 3]
            }
        })
        .collect();
    if candidate.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite candidate query".into());
    }
    let spectral: Vec<_> = colors
        .iter()
        .map(|c| c.spectral.expect("cached spectral teacher is required"))
        .collect();
    let rgb: Vec<_> = colors.iter().map(|c| c.rgb).collect();
    let mut cases = serde_json::Map::new();
    for (j, name) in [
        "noon",
        "afternoon",
        "sunset",
        "blue_hour",
        "aircraft_twilight",
        "stratosphere_shadow",
        "atmosphere_edge",
        "orbit",
    ]
    .iter()
    .enumerate()
    {
        if custom_queries {
            break;
        }
        let indices: Vec<_> = (j * 8309..(j + 1) * 8309).collect();
        cases.insert(name.to_string(),serde_json::json!({"vs_spectral":statistics(&spectral,&candidate,&indices),"vs_rgb":statistics(&rgb,&candidate,&indices)}));
    }
    for (name, range) in [
        ("off_grid_random", 66472..queries.len()),
        ("all", 0..queries.len()),
    ] {
        if custom_queries && name != "all" {
            continue;
        }
        let indices: Vec<_> = range.collect();
        cases.insert(name.into(),serde_json::json!({"vs_spectral":statistics(&spectral,&candidate,&indices),"vs_rgb":statistics(&rgb,&candidate,&indices)}));
    }
    if spectral_only {
        for value in cases.values_mut() {
            value.as_object_mut().unwrap().remove("vs_rgb");
        }
    }
    let report = serde_json::json!({"candidate":dir,"gpu_payload_budget_bytes":raw_metadata["gpu_payload_budget_bytes"],"cpu_only":true,
        "direct_sun_disk":false,"relative_metrics_min_reference_rgb_norm":1e-8,"cases":cases,
        "reference_queries":query_path,"reference_colors":color_path});
    fs::write(
        dir.join(format!("{prefix}validation.json")),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let bytes: Vec<u8> = candidate
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    fs::write(dir.join(format!("{prefix}query_rgb.f32")), bytes)?;
    println!("{} {}", dir.display(), report["cases"]["all"]);
    Ok(())
}
