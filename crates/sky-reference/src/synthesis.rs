//! CPU datasets for fitting a separate production LUT. Geometry is compiled
//! once, bands are streamed, and independent query chunks run on CPU threads.
use crate::{
    Result,
    asset::Manifest,
    mapping::{State, sample_radiance_state},
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb::rec2020_weights,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplePoint {
    pub altitude_km: f32,
    pub sun_elevation_deg: f32,
    pub view_elevation_deg: f32,
    pub relative_azimuth_deg: f32,
}
impl SamplePoint {
    pub fn packed(self) -> Result<[f32; 4]> {
        if [
            self.altitude_km,
            self.sun_elevation_deg,
            self.view_elevation_deg,
            self.relative_azimuth_deg,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || self.altitude_km < 0.0
            || !(-90.0..=90.0).contains(&self.sun_elevation_deg)
            || !(-90.0..=90.0).contains(&self.view_elevation_deg)
        {
            return Err(
                "sample needs finite angles, nonnegative altitude and elevations in [-90,90]"
                    .into(),
            );
        }
        let e = self.view_elevation_deg.to_radians();
        let s = self.sun_elevation_deg.to_radians();
        let nu = e.sin() * s.sin()
            + e.cos()
                * s.cos()
                * self
                    .relative_azimuth_deg
                    .rem_euclid(360.0)
                    .to_radians()
                    .cos();
        Ok([self.altitude_km, e.sin(), s.sin(), nu.clamp(-1.0, 1.0)])
    }
}
enum PreparedQuery {
    Vacuum,
    Reference(ReferenceStencil),
    General(State),
}

/// Diffuse sky or ground radiance, without the visible solar disc. Every band
/// is interpolated before RGB conversion: neither PCHIP nor log interpolation
/// commutes with spectral integration. No GPU adapter is created.
pub fn sample_rec2020(source: &Path, points: &[SamplePoint]) -> Result<Vec<[f32; 3]>> {
    if points.is_empty() {
        return Err("synthesis needs at least one query".into());
    }
    let m = Manifest::open(source)?;
    if !m.complete() || m.rgb.is_some() {
        return Err("synthesis requires a complete spectral reference, before RGB export".into());
    }
    let g = m.geometry;
    let c = &m.config;
    let heights = radius_nodes(g, c);
    let prepared: Vec<_> = points
        .iter()
        .map(|p| {
            let [h, mu, mu_s, nu] = p.packed()?;
            let state = State {
                altitude_km: h,
                mu,
                mu_s,
                nu,
                ground: g.hits_ground(h, mu),
            };
            Ok(match g.atmosphere_entry(state) {
                None => PreparedQuery::Vacuum,
                Some((s, _)) if c.mapping.is_reference() => {
                    PreparedQuery::Reference(ReferenceStencil::new(g, c, s, Some(&heights)))
                }
                Some((s, _)) => PreparedQuery::General(s),
            })
        })
        .collect::<Result<_>>()?;
    let mut rgb = vec![[0.0_f32; 3]; points.len()];
    let mut compensation = vec![[0.0_f32; 3]; points.len()];
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(16);
    let chunk = points.len().div_ceil(workers).max(256);
    for (i, w) in rec2020_weights(&m).iter().enumerate() {
        let band = m.read_band(source, i)?;
        std::thread::scope(|scope| {
            for ((queries, values), errors) in prepared
                .chunks(chunk)
                .zip(rgb.chunks_mut(chunk))
                .zip(compensation.chunks_mut(chunk))
            {
                let band = &band;
                scope.spawn(move || {
                    for ((q, value), error) in queries.iter().zip(values).zip(errors) {
                        let radiance = match q {
                            PreparedQuery::Vacuum => 0.0,
                            PreparedQuery::Reference(stencil) => {
                                stencil.sample_with(|index| band.radiance[index])
                            }
                            PreparedQuery::General(s) => sample_radiance_state(
                                &band.radiance,
                                g,
                                c,
                                *s,
                                &band.scattering_cosines,
                            ),
                        };
                        for j in 0..3 {
                            let y = radiance * w[j] - error[j];
                            let sum = value[j] + y;
                            error[j] = (sum - value[j]) - y;
                            value[j] = sum;
                        }
                    }
                });
            }
        });
        eprintln!("CPU synthesis band {}/{}", i + 1, m.bands.len());
    }
    if rgb.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite dataset radiance".into());
    }
    Ok(rgb)
}
pub fn export(source: &Path, points: &[SamplePoint], output: &Path) -> Result<()> {
    if output.exists() {
        return Err("dataset output already exists".into());
    }
    let m = Manifest::open(source)?;
    let values = sample_rec2020(source, points)?;
    fs::create_dir_all(output)?;
    let path = output.join("samples.f32.part");
    let mut writer = BufWriter::new(fs::File::create(&path)?);
    for value in values.iter().flatten() {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(path, output.join("samples.f32"))?;
    fs::write(
        output.join("queries.json"),
        serde_json::to_vec_pretty(points)?,
    )?;
    let metadata = serde_json::json!({"kind":"reference_atmosphere_samples_v1","samples":points.len(),"format":"little-endian-f32-rgb","color_space":"linear Rec.2020 / solar D65",
        "includes_direct_sun_disk":false,"includes_ground":m.includes_ground,"source_model":m.model_fingerprint_fnv1a64,"source_solver":m.solver,
        "source_mapping":m.coordinate_mapping,"source_config":m.config,"source_band_checksums":m.records.iter().map(|r|&r.as_ref().unwrap().checksum_fnv1a64).collect::<Vec<_>>()});
    fs::write(
        output.join("dataset.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}
