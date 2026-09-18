//! Generate the compact directional source entirely from the frozen reference.
use clap::Parser;
use glam::{Vec3, Vec4};
use half::f16;
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result, anisotropic as sh,
    asset::{Manifest, fingerprint},
    four_wave::{KIND, Resource, checksum},
    mapping::{State, unit},
    model::Model,
    quadrature,
    reference_mapping::{self as mapping, ReferenceStencil, radius_nodes},
};
use std::{
    f32::consts::PI,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 64)]
    angular_mu: usize,
    #[arg(long, default_value_t = 128)]
    angular_phi: usize,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() || a.angular_mu < 16 || a.angular_phi < 32 {
        return Err("use a new output directory and sufficient angular samples".into());
    }
    let start = Instant::now();
    let m = Manifest::open(&a.source)?;
    let candidate: serde_json::Value = serde_json::from_slice(&fs::read(&a.candidate)?)?;
    let checksums: Vec<_> = m
        .records
        .iter()
        .map(|r| {
            r.as_ref()
                .map(|r| r.checksum_fnv1a64.clone())
                .ok_or("incomplete reference")
        })
        .collect::<std::result::Result<_, _>>()?;
    if candidate["model"] != m.model_fingerprint_fnv1a64
        || candidate["source_band_checksums"] != serde_json::to_value(&checksums)?
    {
        return Err("candidate/reference mismatch".into());
    }
    let indices: [usize; 4] = serde_json::from_value(candidate["indices"].clone())?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != m.model_fingerprint_fnv1a64 {
        return Err("model mismatch".into());
    }
    let mut bands = Vec::new();
    for i in indices {
        bands.push(m.read_band(&a.source, i)?);
    }
    let heights = radius_nodes(m.geometry, &m.config);
    let degrees: Vec<_> = heights
        .iter()
        .map(|&h| if h <= 35.0 { 16 } else { 2 })
        .collect();
    let ns = m.config.scattering[2];
    let sphere = quadrature::sphere(a.angular_mu, a.angular_phi);
    let radial = quadrature::gauss_legendre(a.angular_mu);
    let done = AtomicUsize::new(0);
    let mut offsets = Vec::new();
    let mut words = 0;
    for &degree in &degrees {
        offsets.push(words);
        words += ns * sh::count(degree) * 2;
    }
    eprintln!(
        "projecting {} height/Sun states, four wavelengths, CPU only",
        heights.len() * ns
    );
    let projected: Vec<_> = (0..heights.len() * ns)
        .into_par_iter()
        .map(|node| {
            let hi = node / ns;
            let h = heights[hi];
            let degree = degrees[hi];
            let count = sh::count(degree);
            let mu_s = mapping::solar_cosine(m.geometry, h, unit(node % ns, ns));
            let sun = Vec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s);
            let horizon = m.geometry.horizon(h);
            let mut moments = vec![Vec4::ZERO; count];
            let mut correction = moments.clone();
            let mut basis = vec![0.0; count];
            let sun_weight = |v: Vec3| {
                let d = 0.0004 + (1.0 - v.dot(sun)).max(0.0);
                0.01 / (0.01 + d * d)
            };
            let mut sample = |v: Vec3, w: f32| {
                let s = State {
                    altitude_km: h,
                    mu: v.z,
                    mu_s,
                    nu: v.dot(sun),
                    ground: m.geometry.hits_ground(h, v.z),
                };
                let stencil = ReferenceStencil::new(m.geometry, &m.config, s, Some(&heights));
                let light = Vec4::from_array(std::array::from_fn(|k| {
                    stencil.sample_with(|i| bands[k].radiance[i])
                        / (bands[k].info.upper_nm - bands[k].info.lower_nm)
                }));
                sh::cosine_sh(v, degree, &mut basis);
                for i in 0..count {
                    let term = light * (basis[i] * w) - correction[i];
                    let next = moments[i] + term;
                    correction[i] = (next - moments[i]) - term;
                    moments[i] = next;
                }
            };
            for &(v, w) in &sphere {
                let v = quadrature::rotate(v, sun);
                sample(v, w * sun_weight(v));
            }
            for &(u, w) in &radial {
                for (lo, hi) in [(-1.0, horizon), (horizon, 1.0)] {
                    let mu = lo + (hi - lo) * u;
                    let r = (1.0 - mu * mu).max(0.0).sqrt();
                    for j in 0..a.angular_phi {
                        let phi = 2.0 * PI * (j as f32 + 0.5) / a.angular_phi as f32;
                        let v = Vec3::new(r * phi.cos(), r * phi.sin(), mu);
                        sample(
                            v,
                            w * (hi - lo) * 2.0 * PI / a.angular_phi as f32 * (1.0 - sun_weight(v)),
                        );
                    }
                }
            }
            let scale = moments[0].max(Vec4::splat(1e-30));
            let mut packed = Vec::with_capacity(count * 2);
            for v in moments {
                let v = (v / scale)
                    .to_array()
                    .map(|x| f16::from_f32(x).to_bits() as u32);
                packed.extend([v[0] | v[1] << 16, v[2] | v[3] << 16]);
            }
            let progress = done.fetch_add(1, Ordering::Relaxed) + 1;
            if progress % 1024 == 0 {
                eprintln!(
                    "{progress}/{} states, {:.1}s",
                    heights.len() * ns,
                    start.elapsed().as_secs_f32()
                );
            }
            (scale.to_array(), packed)
        })
        .collect();
    let mut aux: Vec<[f32; 4]> = projected.iter().map(|p| p.0).collect();
    let mut data = Vec::with_capacity(words);
    for (_, packed) in projected {
        data.extend(packed);
    }
    let mut aux_offsets = [0; 6];
    aux_offsets[1] = aux.len();
    for i in 0..heights.len() {
        aux.push([heights[i], degrees[i] as f32, offsets[i] as f32, 0.0]);
    }
    aux_offsets[2] = aux.len();
    let profile = &model.bands[indices[0]].profile;
    for (h, _) in profile {
        aux.push([*h, 0.0, 0.0, 0.0]);
        let c = indices.map(|i| model.bands[i].coefficients(*h));
        aux.push(c.map(|x| x.extinction));
        for species in 0..5 {
            aux.push(c.map(|x| x.scattering[species]));
        }
    }
    aux_offsets[3] = aux.len();
    for i in 0..4096 {
        aux.push(indices.map(|b| model.bands[b].phase[i]));
    }
    aux_offsets[4] = aux.len();
    let pm = indices.map(|b| sh::phase_moments(&model.bands[b], 16));
    for l in 0..=16 {
        for species in 0..5 {
            aux.push(std::array::from_fn(|k| pm[k][l][species]));
        }
    }
    aux_offsets[5] = aux.len();
    let ng = m.config.ground_sun_samples;
    for i in 0..ng {
        aux.push(std::array::from_fn(|k| {
            bands[k].ground_irradiance[i] / (bands[k].info.upper_nm - bands[k].info.lower_nm)
        }));
    }
    // Only the unoccluded Sun branch is needed. View transmittance comes from
    // the same segment integral as radiance, so no ground optical branch is stored.
    let optical = [m.config.optical_depth[0], 1024];
    let mut tau = Vec::new();
    for i in 0..optical[0] {
        for j in 0..optical[1] {
            let x = m.config.optical_depth[1] as f32 * 0.5
                + unit(j, optical[1]) * (m.config.optical_depth[1] / 2 - 1) as f32;
            let values = std::array::from_fn::<_, 4, _>(|k| {
                let v = sky_atmosphere_lut::mapping::sample(
                    &bands[k].optical_depth,
                    m.config.optical_depth,
                    [i as f32, x],
                )
                .min(80.0);
                f16::from_f32(v).to_bits() as u32
            });
            tau.extend([values[0] | values[1] << 16, values[2] | values[3] << 16]);
        }
    }
    let mut r = Resource {
        kind: KIND.into(),
        model: m.model_fingerprint_fnv1a64,
        source_checksums: checksums,
        wavelengths_nm: serde_json::from_value(candidate["wavelengths_nm"].clone())?,
        rgb_from_per_nm: serde_json::from_value(candidate["rec2020_from_per_nm"].clone())?,
        sun_per_nm: serde_json::from_value(candidate["sun_irradiance_per_nm"].clone())?,
        geometry: m.geometry,
        sun_radius: m.sun_radius_rad,
        albedo: m.config.ground_albedo,
        heights,
        degrees,
        offsets,
        sun_count: ns,
        optical,
        aux_offsets,
        profile_count: profile.len(),
        ground_count: ng,
        angular_quadrature: [a.angular_mu, a.angular_phi],
        payload_bytes: (data.len() + tau.len()) * 4 + aux.len() * 16,
        files: Vec::new(),
    };
    if r.payload_bytes > 16 * 1024 * 1024 {
        return Err("resource exceeds 16 MiB".into());
    }
    fs::create_dir_all(&a.out)?;
    for (name, bytes) in [
        ("moments.u32", bytemuck::cast_slice(&data)),
        ("aux.f32", bytemuck::cast_slice(&aux)),
        ("optical.u32", bytemuck::cast_slice(&tau)),
    ] {
        fs::write(a.out.join(name), bytes)?;
        r.files.push((name.into(), bytes.len(), checksum(bytes)));
    }
    fs::write(a.out.join("resource.json"), serde_json::to_vec_pretty(&r)?)?;
    eprintln!(
        "exported {:.3} MiB in {:.1} s",
        r.payload_bytes as f32 / 1048576.0,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
