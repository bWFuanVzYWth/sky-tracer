//! Use the existing demo's optimized 8 wavelengths for the runtime part only.
//! Reuses a completed 41-band CPU probe as teacher and multiple-scattering LUT.
#[path = "support/eight_wave.rs"]
mod eight_wave;
#[path = "support/frozen_direct.rs"]
mod frozen_direct;

use clap::Parser;
use eight_wave::EightWave;
use frozen_direct::direct;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sky_atmosphere_lut::{
    Result,
    asset::{Manifest, fingerprint},
    mapping::State,
    model::Model,
    synthesis::SamplePoint,
};
use sky_unreal_atmosphere_8wave::params::{
    SPECTRAL_QUADRATURE_WEIGHTS_NM, SPECTRAL_SAMPLE_WAVELENGTHS_NM,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    /// Completed all-bands probe.json, with a spectral teacher and bf16 remainder.
    #[arg(long)]
    full_probe: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16)]
    threads: usize,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
struct Spec {
    steps: usize,
    point_sun: bool,
    log_height: bool,
}
#[derive(Deserialize)]
struct FullProbe {
    model: String,
    source_checksums: Vec<Option<String>>,
    full_spectral_rec2020: bool,
    bands: Vec<usize>,
    steps: usize,
    sun_samples: usize,
    runtime_specs: Vec<Spec>,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    region: String,
    point: SamplePoint,
    teacher: [f32; 3],
    single: [f32; 3],
    boundary: [f32; 3],
    runtime_direct: Vec<[f32; 3]>,
    candidates: std::collections::BTreeMap<String, Candidate>,
}
#[derive(Deserialize)]
struct Candidate {
    hybrid_bf16: [f32; 3],
}
fn add(target: &mut [f32; 3], v: f32, w: [f32; 3]) {
    for c in 0..3 {
        target[c] += v * w[c];
    }
}

/// Small CPU-precomputed tables already arranged in the two vec4 groups used
/// by the demo. This pack has no GPU adapter or dynamic atmosphere parameters.
fn export_parameters(dir: &Path, m: &Manifest, model: &Model, eight: &EightWave) -> Result<()> {
    let heights: Vec<_> = model.bands[eight.indices[0]]
        .profile
        .iter()
        .map(|p| p.0)
        .collect();
    let mut profile = Vec::<f32>::new();
    for (h, &height) in heights.iter().enumerate() {
        for group in 0..2 {
            for coefficient in 0..6 {
                for lane in 0..4 {
                    let band = &model.bands[eight.indices[group * 4 + lane]];
                    if band.profile[h].0 != height {
                        return Err("spectral profiles have different height grids".into());
                    }
                    let c = band.profile[h].1;
                    profile.push(if coefficient == 0 {
                        c.extinction
                    } else {
                        c.scattering[coefficient - 1]
                    });
                }
            }
        }
    }
    let bins = sky_core::atmosphere::PHASE_BINS;
    let mut phase = Vec::<f32>::new();
    for group in 0..2 {
        for species in 0..4 {
            for bin in 0..bins {
                for lane in 0..4 {
                    phase.push(
                        model.bands[eight.indices[group * 4 + lane]].phase[species * bins + bin],
                    );
                }
            }
        }
    }
    fs::create_dir_all(dir)?;
    fs::write(dir.join("heights.f32"), bytemuck::cast_slice(&heights))?;
    fs::write(dir.join("profile.f32"), bytemuck::cast_slice(&profile))?;
    fs::write(dir.join("phase.f32"), bytemuck::cast_slice(&phase))?;
    let converter: [[_; 3]; 8] = std::array::from_fn(|k| {
        let b = &m.bands[eight.indices[k]];
        eight.rgb_from_integrated[k].map(|x| x * (b.upper_nm - b.lower_nm))
    });
    let sun: [f32; 8] = std::array::from_fn(|k| {
        let b = &m.bands[eight.indices[k]];
        b.solar_irradiance_w_m2 / (b.upper_nm - b.lower_nm)
    });
    fs::write(
        dir.join("parameters.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"precomputed_frozen_medium_eight_wave_f32_v1","model":m.model_fingerprint_fnv1a64,
            "wavelengths_nm":SPECTRAL_SAMPLE_WAVELENGTHS_NM,"quadrature_weights_nm":SPECTRAL_QUADRATURE_WEIGHTS_NM,
            "sun_irradiance_per_nm":sun,"rec2020_from_per_nm":converter,
            "white_balance":"same full-reference solar-to-D65 as the demo, not recalibrated to eight samples",
            "profile_shape":[heights.len(),2,6,4],"profile_coefficients":["extinction","rayleigh_scattering","inso_scattering","waso_scattering","soot_scattering","suso_scattering"],
            "phase_shape":[2,4,bins,4],"phase_species":["inso","waso","soot","suso"],
            "phase_coordinate":"cbrt((1-cos_theta)/2); texel-center linear interpolation",
            "payload_bytes":(heights.len()+profile.len()+phase.len())*4,
            "includes_optical_depth_or_multiple_lut":false,"gpu_integrated":false
        }))?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.out.exists() || args.threads == 0 {
        return Err("use a new output directory and positive thread count".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let started = Instant::now();
    let m = Manifest::open(&args.source)?;
    let full: FullProbe = serde_json::from_slice(&fs::read(&args.full_probe)?)?;
    let checksums: Vec<_> = m
        .records
        .iter()
        .map(|r| r.as_ref().map(|r| r.checksum_fnv1a64.clone()))
        .collect();
    if !full.full_spectral_rec2020
        || full.bands != (0..m.bands.len()).collect::<Vec<_>>()
        || full.model != m.model_fingerprint_fnv1a64
        || full.source_checksums != checksums
        || full.sun_samples != m.config.sun_mu * m.config.sun_phi
        || full.steps == 0
        || full.rows.is_empty()
        || full.runtime_specs.is_empty()
        || full.runtime_specs.iter().any(|s| s.steps == 0)
        || full.rows.iter().any(|r| {
            r.runtime_direct.len() != full.runtime_specs.len()
                || !r.candidates.contains_key("budget_rgb16")
        })
    {
        return Err("the full-spectral probe does not match this frozen reference".into());
    }
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != m.model_fingerprint_fnv1a64 {
        return Err("model fingerprint mismatch".into());
    }
    let eight = EightWave::new(&m)?;
    let states: Vec<_> = full
        .rows
        .iter()
        .map(|r| {
            let [h, mu, mu_s, nu] = r.point.packed()?;
            Ok(m.geometry
                .atmosphere_entry(State {
                    altitude_km: h,
                    mu,
                    mu_s,
                    nu,
                    ground: m.geometry.hits_ground(h, mu),
                })
                .map(|(s, _)| s))
        })
        .collect::<Result<_>>()?;
    let mut singles = vec![[0.0; 3]; full.rows.len()];
    let mut boundaries = singles.clone();
    let mut runtime = vec![singles.clone(); full.runtime_specs.len()];
    let mut retained_spectrum = vec![[0.0_f32; 8]; full.rows.len()];
    let mut per_band_seconds = Vec::new();
    fs::create_dir_all(&args.out)?;
    export_parameters(&args.out.join("runtime_parameters"), &m, &model, &eight)?;
    for (k, &bi) in eight.indices.iter().enumerate() {
        let t = Instant::now();
        let lut = m.read_band(&args.source, bi)?;
        let w = eight.rgb_from_integrated[k];
        let bm = &model.bands[bi];
        let evaluate = |steps, point, height| -> Vec<[f32; 2]> {
            states
                .par_iter()
                .map(|&s| s.map_or([0.0; 2], |s| direct(&lut, bm, s, steps, point, height)))
                .collect()
        };
        let parts = evaluate(full.steps, false, false);
        for j in 0..full.rows.len() {
            add(&mut singles[j], parts[j][0], w);
            add(&mut boundaries[j], parts[j][1], w);
            let b = &m.bands[bi];
            retained_spectrum[j][k] = parts[j][0] / (b.upper_nm - b.lower_nm);
        }
        for (i, spec) in full.runtime_specs.iter().enumerate() {
            let p = if spec.steps == full.steps && !spec.point_sun && !spec.log_height {
                parts.clone()
            } else {
                evaluate(spec.steps, spec.point_sun, spec.log_height)
            };
            for j in 0..full.rows.len() {
                add(&mut runtime[i][j], p[j][0] + p[j][1], w);
            }
        }
        per_band_seconds.push(t.elapsed().as_secs_f32());
        eprintln!(
            "lane {k}, {:.0} nm: {:.2}s",
            m.bands[bi].center_nm,
            t.elapsed().as_secs_f32()
        );
    }
    let rows:Vec<_>=full.rows.iter().enumerate().map(|(j,r)|{
        let original=r.candidates["budget_rgb16"].hybrid_bf16;
        let multiple:[f32;3]=std::array::from_fn(|c|original[c]-r.single[c]-r.boundary[c]);
        let hybrid:[f32;3]=std::array::from_fn(|c|multiple[c]+singles[j][c]+boundaries[j][c]);
        let runtime_rows:Vec<_>=full.runtime_specs.iter().enumerate().map(|(i,spec)|{
            let hybrid8:[f32;3]=std::array::from_fn(|c|multiple[c]+runtime[i][j][c]);
            let hybrid41:[f32;3]=std::array::from_fn(|c|multiple[c]+r.runtime_direct[i][c]);
            serde_json::json!({"spec":spec,"direct8":runtime[i][j],"direct41":r.runtime_direct[i],"hybrid8":hybrid8,"hybrid41":hybrid41})
        }).collect();
        serde_json::json!({"region":r.region,"point":r.point,"teacher":r.teacher,"single41":r.single,"single8":singles[j],
            "boundary41":r.boundary,"boundary8":boundaries[j],"hybrid41":original,"hybrid8":hybrid,"runtime":runtime_rows})
    }).collect();
    fs::write(
        args.out.join("single_per_nm.f32"),
        bytemuck::cast_slice(&retained_spectrum),
    )?;
    fs::write(
        args.out.join("probe.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"cpu_optimized_eight_wave_single_plus_multi_v1","source":args.source,"full_probe":args.full_probe,
            "model":m.model_fingerprint_fnv1a64,"source_checksums":checksums,"elapsed_seconds":started.elapsed().as_secs_f32(),"per_band_seconds":per_band_seconds,
            "runtime_wavelengths_nm":SPECTRAL_SAMPLE_WAVELENGTHS_NM,"runtime_quadrature_weights_nm":SPECTRAL_QUADRATURE_WEIGHTS_NM,
            "rgb_from_integrated_band":eight.rgb_from_integrated,"sun_disk_samples":full.sun_samples,"rows":rows,
            "limitations":["CPU spectral-reduction audit only, no GPU timing or demo integration",
                "Optical depth and medium match the frozen teacher; the existing demo's analytic profiles, three-species simplification and isotropic multiple scattering are not copied",
                "Multiple-scattering RGB values still come from the full 41-band teacher",
                "Finite-distance fog, auxiliary-table compression and finite-Sun acceleration are not yet tested"]
        }))?,
    )?;
    Ok(())
}
