//! Stream all 41 bands of current LUT and matching direct transport into a CPU dataset.
#[path = "support/eight_wave.rs"]
mod eight_wave;
#[path = "support/frozen_direct.rs"]
mod frozen_direct;
use clap::Parser;
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result,
    asset::{Manifest, fingerprint},
    mapping::State,
    model::Model,
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb::rec2020_weights,
    synthesis::SamplePoint,
};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 256)]
    steps: usize,
    #[arg(long, default_value_t = 16)]
    threads: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() || a.steps == 0 || a.threads == 0 {
        return Err("use new output directory and positive counts".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()?;
    let start = Instant::now();
    let m = Manifest::open(&a.source)?;
    if !m.complete() || m.rgb.is_some() {
        return Err("complete spectral reference required".into());
    }
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let points: Vec<SamplePoint> = serde_json::from_value(plan["queries"].clone())?;
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != m.model_fingerprint_fnv1a64 {
        return Err("reference/model mismatch".into());
    }
    let g = m.geometry;
    let heights = radius_nodes(g, &m.config);
    let states: Vec<_> = points
        .iter()
        .map(|p| {
            let [h, mu, mu_s, nu] = p.packed()?;
            Ok(g.atmosphere_entry(State {
                altitude_km: h,
                mu,
                mu_s,
                nu,
                ground: g.hits_ground(h, mu),
            })
            .map(|(s, _)| s))
        })
        .collect::<Result<_>>()?;
    let stencils: Vec<_> = states
        .iter()
        .map(|s| s.map(|s| ReferenceStencil::new(g, &m.config, s, Some(&heights))))
        .collect();
    fs::create_dir_all(&a.out)?;
    fs::copy(&a.queries, a.out.join("queries.json"))?;
    let names = ["total.f32", "single.f32", "boundary.f32"];
    let mut writers = names
        .iter()
        .map(|n| Ok(BufWriter::new(fs::File::create(a.out.join(n))?)))
        .collect::<Result<Vec<_>>>()?;
    let mut times = Vec::new();
    for bi in 0..m.bands.len() {
        let t = Instant::now();
        let lut = m.read_band(&a.source, bi)?;
        let band = &model.bands[bi];
        let values: Vec<[f32; 3]> = states
            .par_iter()
            .zip(&stencils)
            .map(|(s, st)| {
                let total = st
                    .as_ref()
                    .map_or(0.0, |st| st.sample_with(|i| lut.radiance[i]));
                let direct = s.map_or([0.0; 2], |s| {
                    frozen_direct::direct(&lut, band, s, a.steps, false, false)
                });
                [total, direct[0], direct[1]]
            })
            .collect();
        for (c, writer) in writers.iter_mut().enumerate() {
            let values: Vec<_> = values.iter().map(|v| v[c]).collect();
            if values.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite generated spectrum".into());
            }
            writer.write_all(bytemuck::cast_slice(&values))?;
            writer.flush()?;
        }
        times.push(t.elapsed().as_secs_f32());
        eprintln!(
            "CPU spectral dataset {}/{}: {:.0} nm, {} samples, {:.1}s",
            bi + 1,
            m.bands.len(),
            band.info.center_nm,
            points.len(),
            times[bi]
        );
        fs::write(
            a.out.join("progress.json"),
            serde_json::to_vec(
                &serde_json::json!({"completed_bands":bi+1,"band_seconds":times,"elapsed_seconds":start.elapsed().as_secs_f32()}),
            )?,
        )?;
    }
    for w in &mut writers {
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    let old = eight_wave::EightWave::new(&m)?;
    let metadata = serde_json::json!({"kind":"current_lut_wavelength_search_dataset_v1","source":a.source,"model":m.model_fingerprint_fnv1a64,
        "source_band_checksums":m.records.iter().map(|r|&r.as_ref().unwrap().checksum_fnv1a64).collect::<Vec<_>>(),
        "bands":m.bands,"rgb_from_integrated":rec2020_weights(&m),"array_shape":[m.bands.len(),points.len()],
        "array_format":"little-endian-f32, wavelength-major, band-integrated radiance","files":names,
        "old_demo_indices":old.indices,"old_demo_rgb_from_integrated":old.rgb_from_integrated,
        "ray_steps":a.steps,"sun_disk_samples":m.config.sun_mu*m.config.sun_phi,"elapsed_seconds":start.elapsed().as_secs_f32(),
        "note":"Current frozen LUT total plus directly integrated single and ground boundary; no old PT training samples. Total excludes direct solar disk. Wavelengths are restricted to existing 10 nm reference bands; no sub-band accuracy claim."});
    fs::write(
        a.out.join("dataset.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}
