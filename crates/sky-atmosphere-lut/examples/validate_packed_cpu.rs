//! Off-grid codec audit and separate spectral-to-RGB interpolation audit.
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result,
    asset::Manifest,
    mapping::State,
    packed::PackedLut,
    reference_mapping::{ReferenceStencil, radius_nodes},
    synthesis::{SamplePoint, sample_rec2020},
};
use std::{fs, path::PathBuf};

fn norm(v: [f32; 3]) -> f32 {
    v.into_iter().map(|v| v * v).sum::<f32>().sqrt()
}
fn metric(a: &[[f32; 3]], b: &[[f32; 3]], indices: &[usize]) -> serde_json::Value {
    let mut relative = Vec::new();
    let mut channel = Vec::new();
    let mut absolute = 0.0_f32;
    let mut sum_err = 0.0;
    let mut sum_ref = 0.0;
    for &i in indices {
        let n = norm(a[i]);
        let diff = std::array::from_fn(|c| b[i][c] - a[i][c]);
        let e = norm(diff);
        absolute = absolute.max(e);
        sum_err += e * e;
        sum_ref += n * n;
        // Separate rendered-light values from physically negligible tails.
        if n > 1e-8 {
            relative.push(e / n);
            channel.push(
                (0..3)
                    .map(|c| diff[c].abs() / a[i][c].abs().max(n * 0.01))
                    .fold(0.0, f32::max),
            );
        }
    }
    let percentile = |mut v: Vec<f32>| {
        v.sort_unstable_by(f32::total_cmp);
        if v.is_empty() {
            return serde_json::Value::Null;
        }
        serde_json::json!({"p50":v[v.len()/2],"p95":v[v.len()*95/100],"p99":v[v.len()*99/100],"max":v[v.len()-1]})
    };
    serde_json::json!({"queries":indices.len(),"queries_above_rgb_norm_1e_8":relative.len(),"relative_rgb":percentile(relative),
        "relative_channel_floor_1percent_norm":percentile(channel),"max_absolute_rgb_norm":absolute,"relative_rmse":(sum_err/sum_ref.max(1e-30)).sqrt()})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let source = PathBuf::from(args.get(1).ok_or("RGB source")?);
    let packed_dir = PathBuf::from(args.get(2).ok_or("packed source")?);
    let out = PathBuf::from(args.get(3).ok_or("output JSON")?);
    let spectral = args.get(4).map(PathBuf::from);
    let m = Manifest::open(&source)?;
    let pm = Manifest::open(&packed_dir)?;
    if m.config != pm.config
        || m.model_fingerprint_fnv1a64 != pm.model_fingerprint_fnv1a64
        || m.rgb.as_ref().unwrap().channel_checksums != pm.rgb.as_ref().unwrap().channel_checksums
    {
        return Err("codec/reference provenance mismatch".into());
    }
    let compact = PackedLut::read(&pm, &packed_dir)?;
    let mut channels = Vec::new();
    for c in 0..3 {
        channels.push(sky_atmosphere_lut::rgb::read_channel(&m, &source, c)?.1);
    }
    let mut points = Vec::new();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (name, h, s) in [
        ("noon", 0.2, 85.0),
        ("afternoon", 0.2, 47.0),
        ("sunset", 0.2, 0.0),
        ("blue_hour", 0.2, -6.0),
        ("aircraft_twilight", 12.0, -4.0),
        ("stratosphere_shadow", 30.0, -5.58),
        ("atmosphere_edge", 120.0, -11.0),
        ("orbit", 400.0, -6.0),
    ] {
        let horizon = m.geometry.horizon(h).asin().to_degrees();
        let mut indices = Vec::new();
        for row in 0..64 {
            for column in 0..128 {
                // Half the samples resolve a 6-degree horizon band; the rest cover
                // the sphere. Columns include forward, backward and both limbs.
                let elevation = if row < 32 {
                    horizon + ((row as f32 + 0.5) / 32.0 - 0.5) * 6.0
                } else {
                    ((row - 32) as f32 + 0.5) / 32.0 * 180.0 - 90.0
                };
                indices.push(points.len());
                points.push(SamplePoint {
                    altitude_km: h,
                    sun_elevation_deg: s,
                    view_elevation_deg: elevation.clamp(-90.0, 90.0),
                    relative_azimuth_deg: (column as f32 + 0.5) / 128.0 * 360.0,
                });
            }
        }
        // Dense scan around the solar disk and the earlier 20+ degree ring.
        for delta in [
            -30.0, -20.0, -10.0, -2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0, 10.0, 20.0, 30.0,
        ] {
            for az in [0.0, 0.1, 0.5, 1.0, 5.0, 20.0, 40.0, 90.0, 180.0] {
                indices.push(points.len());
                points.push(SamplePoint {
                    altitude_km: h,
                    sun_elevation_deg: s,
                    view_elevation_deg: (s + delta).clamp(-90.0, 90.0),
                    relative_azimuth_deg: az,
                });
            }
        }
        groups.push((name.into(), indices));
    }
    let mut seed = 91397u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / 16777216.0
    };
    let mut indices = Vec::new();
    for _ in 0..16384 {
        indices.push(points.len());
        points.push(SamplePoint {
            altitude_km: (random() * 10.0).exp() - 1.0,
            sun_elevation_deg: random() * 180.0 - 90.0,
            view_elevation_deg: (random() * 2.0 - 1.0).asin().to_degrees(),
            relative_azimuth_deg: random() * 360.0,
        });
    }
    groups.push(("off_grid_random".into(), indices));
    let heights = radius_nodes(m.geometry, &m.config);
    let results: Vec<_> = points
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
                let stencil = ReferenceStencil::new(m.geometry, &m.config, s, Some(&heights));
                let a = std::array::from_fn(|c| stencil.sample_with(|i| channels[c][i]));
                let b = std::array::from_fn(|c| stencil.sample_with(|i| compact.fetch(i, c)));
                (a, b)
            } else {
                ([0.0; 3], [0.0; 3])
            }
        })
        .collect();
    let (reference, decoded): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    let teacher = if let Some(path) = &spectral {
        Some(sample_rec2020(path, &points)?)
    } else {
        None
    };
    let mut cases = serde_json::Map::new();
    groups.push(("all".into(), (0..points.len()).collect()));
    for (name, indices) in &groups {
        cases.insert(
            name.clone(),
            serde_json::json!({"codec_vs_rgb":metric(&reference,&decoded,indices),
            "rgb_export_vs_spectral":teacher.as_ref().map(|t|metric(t,&reference,indices)),
            "packed_vs_spectral":teacher.as_ref().map(|t|metric(t,&decoded,indices))}),
        );
    }
    fs::write(
        &out,
        serde_json::to_vec_pretty(&serde_json::json!({"source_rgb":source,"packed":packed_dir,
        "source_spectral":spectral,"cpu_only":true,"direct_sun_disk":false,"cases":cases}))?,
    )?;
    fs::write(
        out.with_extension("queries.json"),
        serde_json::to_vec_pretty(&points)?,
    )?;
    let colors:Vec<_>=(0..points.len()).map(|i|serde_json::json!({"rgb":reference[i],"packed":decoded[i],"spectral":teacher.as_ref().map(|t|t[i])})).collect();
    fs::write(
        out.with_extension("colors.json"),
        serde_json::to_vec(&colors)?,
    )?;
    println!("{}", out.display());
    Ok(())
}
