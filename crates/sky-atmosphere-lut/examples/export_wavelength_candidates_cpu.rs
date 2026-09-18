//! Materialize searched candidates as fixed-medium f32 tables, without a GPU.
use clap::Parser;
use serde::Deserialize;
use sky_atmosphere_lut::{
    Result,
    asset::{Manifest, fingerprint},
    model::Model,
    rgb::rec2020_weights,
};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    search: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Deserialize)]
struct Search {
    model: String,
    source_band_checksums: Vec<String>,
    selected: Vec<Candidate>,
}
#[derive(Deserialize)]
struct Candidate {
    id: String,
    indices: Vec<usize>,
    wavelengths_nm: Vec<f32>,
    quadrature_weights_nm: Vec<f32>,
    rgb_from_integrated: Vec<[f32; 3]>,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() {
        return Err("use a new output directory".into());
    }
    let m = Manifest::open(&a.source)?;
    if !m.complete() || m.rgb.is_some() {
        return Err("complete spectral reference required".into());
    }
    let search: Search = serde_json::from_slice(&fs::read(&a.search)?)?;
    let checksums: Vec<_> = m
        .records
        .iter()
        .map(|r| r.as_ref().unwrap().checksum_fnv1a64.clone())
        .collect();
    if search.model != m.model_fingerprint_fnv1a64
        || search.source_band_checksums != checksums
        || search.selected.is_empty()
    {
        return Err("search/reference mismatch".into());
    }
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != search.model {
        return Err("model/reference mismatch".into());
    }
    let rgb = rec2020_weights(&m);
    let bins = sky_core::atmosphere::PHASE_BINS;
    for c in &search.selected {
        let count = c.indices.len();
        if c.id.is_empty()
            || !c.id.bytes().all(|x| x.is_ascii_alphanumeric() || x == b'_')
            || ![3, 4, 8].contains(&count)
            || c.wavelengths_nm.len() != count
            || c.quadrature_weights_nm.len() != count
            || c.rgb_from_integrated.len() != count
            || c.indices.windows(2).any(|w| w[0] >= w[1])
            || c.indices[count - 1] >= m.bands.len()
        {
            return Err("invalid candidate id or indices".into());
        }
        for (k, &i) in c.indices.iter().enumerate() {
            let b = &m.bands[i];
            let width = b.upper_nm - b.lower_nm;
            let weight = c.quadrature_weights_nm[k];
            if c.wavelengths_nm[k] != b.center_nm || !weight.is_finite() || weight <= 0.0 {
                return Err("invalid wavelength or quadrature weight".into());
            }
            for channel in 0..3 {
                let expected = rgb[i][channel] * (weight / width);
                let found = c.rgb_from_integrated[k][channel];
                if !found.is_finite() || (expected - found).abs() > 3e-6 * expected.abs().max(1.0) {
                    return Err("candidate matrix is not a scalar spectral quadrature".into());
                }
            }
        }
        let dir = a.out.join(&c.id);
        let heights: Vec<_> = model.bands[c.indices[0]]
            .profile
            .iter()
            .map(|p| p.0)
            .collect();
        let mut profile = Vec::new();
        let groups = count.div_ceil(4);
        for (h, &height) in heights.iter().enumerate() {
            for group in 0..groups {
                for coefficient in 0..6 {
                    for lane in 0..4 {
                        let Some(&index) = c.indices.get(group * 4 + lane) else {
                            profile.push(0.0);
                            continue;
                        };
                        let p = model.bands[index].profile[h];
                        if p.0 != height {
                            return Err("inconsistent profile height grids".into());
                        }
                        profile.push(if coefficient == 0 {
                            p.1.extinction
                        } else {
                            p.1.scattering[coefficient - 1]
                        });
                    }
                }
            }
        }
        let mut phase = Vec::new();
        for group in 0..groups {
            for species in 0..4 {
                for bin in 0..bins {
                    for lane in 0..4 {
                        phase.push(
                            c.indices.get(group * 4 + lane).map_or(0.0, |&index| {
                                model.bands[index].phase[species * bins + bin]
                            }),
                        );
                    }
                }
            }
        }
        let converter: Vec<[f32; 3]> = (0..count)
            .map(|k| {
                let b = &m.bands[c.indices[k]];
                c.rgb_from_integrated[k].map(|x| x * (b.upper_nm - b.lower_nm))
            })
            .collect();
        let sun: Vec<f32> = (0..count)
            .map(|k| {
                let b = &m.bands[c.indices[k]];
                b.solar_irradiance_w_m2 / (b.upper_nm - b.lower_nm)
            })
            .collect();
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("heights.f32"), bytemuck::cast_slice(&heights))?;
        fs::write(dir.join("profile.f32"), bytemuck::cast_slice(&profile))?;
        fs::write(dir.join("phase.f32"), bytemuck::cast_slice(&phase))?;
        fs::write(
            dir.join("parameters.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "kind":"precomputed_frozen_medium_spectral_f32_v2","candidate":c.id,"model":search.model,
                "active_wavelength_count":count,"vec4_groups":groups,"padding":"unused profile and phase lanes are zero; converter and sun contain only active lanes",
                "source_band_checksums":checksums,"wavelengths_nm":c.wavelengths_nm,"quadrature_weights_nm":c.quadrature_weights_nm,
                "sun_irradiance_per_nm":sun,"rec2020_from_per_nm":converter,
                "profile_shape":[heights.len(),groups,6,4],"profile_coefficients":["extinction","rayleigh_scattering","inso_scattering","waso_scattering","soot_scattering","suso_scattering"],
                "phase_shape":[groups,4,bins,4],"phase_coordinate":"cbrt((1-cos_theta)/2); texel-center linear interpolation",
                "color_space":"linear Rec.2020; unchanged full-reference solar-to-D65 transform",
                "payload_bytes":(heights.len()+profile.len()+phase.len())*4,
                "includes_optical_depth_or_multiple_lut":false,"gpu_integrated":false
            }))?,
        )?;
        eprintln!("exported {}: {:?}", c.id, c.wavelengths_nm);
    }
    Ok(())
}
