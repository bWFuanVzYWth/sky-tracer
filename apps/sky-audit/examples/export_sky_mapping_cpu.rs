//! Dense directional teacher samples. No GPU device or direct-light integration.
use clap::Parser;
use sky_realtime::Wavelengths;
use sky_reference::{
    Result,
    asset::Manifest,
    mapping::State,
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb::rec2020_weights,
    synthesis::SamplePoint,
};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    queries: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    wavelengths: Option<PathBuf>,
    /// Full 41-band RGB; otherwise use the current four-band surrogate.
    #[arg(long)]
    full_spectrum: bool,
    #[arg(long, default_value_t = 8)]
    threads: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() || !(1..=8).contains(&a.threads) {
        return Err("use a new output directory and 1..=8 CPU threads".into());
    }
    let start = Instant::now();
    let m = Manifest::open(&a.source)?;
    if !m.complete() || m.rgb.is_some() || !m.config.mapping.is_reference() {
        return Err("complete spectral reference mapping required".into());
    }
    let w: Wavelengths = if let Some(path) = a.wavelengths {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        Wavelengths::optimized_four()
    };
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let points: Vec<SamplePoint> = serde_json::from_value(plan["queries"].clone())?;
    if points.is_empty() {
        return Err("no teacher queries".into());
    }
    let g = m.geometry;
    if let Some(geometry) = plan.get("geometry") {
        let bottom = geometry["bottom_km"]
            .as_f64()
            .ok_or("missing curve geometry")? as f32;
        let height = geometry["top_height_km"]
            .as_f64()
            .ok_or("missing curve geometry")? as f32;
        if bottom != g.bottom || height != g.top_height() {
            return Err("dense curves and teacher geometry do not match".into());
        }
    }
    if let Some(geometry) = plan.get("geometry") {
        let bottom = geometry["bottom_km"]
            .as_f64()
            .ok_or("missing curve geometry")? as f32;
        let height = geometry["top_height_km"]
            .as_f64()
            .ok_or("missing curve geometry")? as f32;
        if bottom != g.bottom || height != g.top_height() {
            return Err("dense curves and teacher geometry do not match".into());
        }
    }
    let heights = radius_nodes(g, &m.config);
    let mut stencils: Vec<_> = points
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
            .map(|(s, _)| ReferenceStencil::new(g, &m.config, s, Some(&heights))))
        })
        .collect::<Result<_>>()?;
    if let Some(indices) = plan["vacuum_query_indices"].as_array() {
        for index in indices {
            let index = index.as_u64().ok_or("invalid vacuum endpoint index")? as usize;
            *stencils
                .get_mut(index)
                .ok_or("vacuum endpoint index outside query array")? = None;
        }
    }
    let prepare_seconds = start.elapsed().as_secs_f64();
    let selected = if a.full_spectrum {
        (0..m.bands.len()).collect::<Vec<_>>()
    } else {
        w.indices.to_vec()
    };
    if selected.iter().any(|&i| i >= m.bands.len())
        || w.rgb.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("invalid wavelength selection".into());
    }
    let weights = rec2020_weights(&m);
    let mut values = vec![[0.0f32; 3]; points.len()];
    let mut compensation = values.clone();
    let mut spectra = vec![[0.0f32; 4]; points.len()];
    let chunk = points.len().div_ceil(a.threads).max(1);
    let mut timings = Vec::new();
    for (selected_index, &band_index) in selected.iter().enumerate() {
        let band_start = Instant::now();
        let lut = m.read_band(&a.source, band_index)?;
        let read_seconds = band_start.elapsed().as_secs_f64();
        let band = m
            .bands
            .get(band_index)
            .ok_or("wavelength index outside teacher")?;
        let width = band.upper_nm - band.lower_nm;
        let rgb = if a.full_spectrum {
            weights[band_index]
        } else {
            if (band.center_nm - w.wavelengths_nm[selected_index]).abs() > 0.01 {
                return Err("four-wave wavelength/teacher band mismatch".into());
            }
            w.rgb[selected_index].map(|v| v / width)
        };
        std::thread::scope(|scope| {
            for (((s, v), c), sp) in stencils
                .chunks(chunk)
                .zip(values.chunks_mut(chunk))
                .zip(compensation.chunks_mut(chunk))
                .zip(spectra.chunks_mut(chunk))
            {
                let radiance = &lut.radiance;
                let full_spectrum = a.full_spectrum;
                scope.spawn(move || {
                    for (((s, v), c), sp) in s.iter().zip(v).zip(c).zip(sp) {
                        let l = s.as_ref().map_or(0.0, |s| s.sample_with(|i| radiance[i]));
                        for k in 0..3 {
                            let y = l * rgb[k] - c[k];
                            let t = v[k] + y;
                            c[k] = (t - v[k]) - y;
                            v[k] = t;
                        }
                        if !full_spectrum {
                            sp[selected_index] = l / width;
                        }
                    }
                });
            }
        });
        timings.push(
            serde_json::json!({"band_index":band_index,"nm":band.center_nm,
            "read_seconds":read_seconds,"total_seconds":band_start.elapsed().as_secs_f64()}),
        );
        eprintln!(
            "CPU teacher {}/{} {:.0} nm: {:.2}s",
            selected_index + 1,
            selected.len(),
            band.center_nm,
            band_start.elapsed().as_secs_f64()
        );
    }
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite teacher samples".into());
    }
    fs::create_dir_all(&a.out)?;
    fs::copy(&a.queries, a.out.join("queries.json"))?;
    fs::write(a.out.join("rgb.f32"), bytemuck::cast_slice(&values))?;
    if !a.full_spectrum {
        fs::write(
            a.out.join("four_per_nm.f32"),
            bytemuck::cast_slice(&spectra),
        )?;
    }
    let source = a.source.to_string_lossy().replace('\\', "/");
    fs::write(
        a.out.join("dataset.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"dense_teacher_sky_mapping_cpu_v1","source":source,"samples":points.len(),
            "format":"query-major little-endian-f32 RGB, linear Rec.2020","wavelengths":w,
            "full_spectrum":a.full_spectrum,"selected_band_indices":selected,"threads":a.threads,
            "prepare_seconds":prepare_seconds,"band_timings":timings,"elapsed_seconds":start.elapsed().as_secs_f64(),
            "source_model":m.model_fingerprint_fnv1a64,"source_mapping":m.coordinate_mapping,
            "source_checksums":m.records.iter().map(|r|r.as_ref().map(|r|&r.checksum_fnv1a64)).collect::<Vec<_>>(),
            "includes_direct_sun_disk":false,"includes_ground":true,"gpu_initialized":false,
            "single_light_integrations":0,
            "note":"Teacher total only. Four-wave surrogate has its existing heldout spectral-fit error; full-spectrum option avoids that approximation. This does not export or invert the local normalized scattering source."
        }))?,
    )?;
    eprintln!(
        "{} CPU samples, {:.2}s; no GPU",
        points.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
