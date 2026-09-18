//! Frozen cumulative-field source oracle: separate interpolation from quadrature.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    Result,
    asset::{BandLut, Manifest, fingerprint},
    mapping::State,
    model::{BandModel, Model, phase_weight},
    quadrature,
    ray_mapping::RayStencil,
    reference_mapping::{ReferenceStencil, radius_nodes},
};
use std::{collections::HashMap, fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 17)]
    band: usize,
    /// Move nearest height nodes to these profile boundaries, without adding nodes.
    #[arg(long, value_delimiter = ',')]
    height_anchors: Vec<f32>,
}
fn scattering(b: &BandModel, h: f32) -> f32 {
    b.coefficients(h).scattering.iter().sum()
}
fn source(l: &BandLut, b: &BandModel, heights: &[f32], sphere: &[(Vec3, f32)], s: State) -> f32 {
    let (ray, sun) = s.directions();
    let c = b.coefficients(s.altitude_km);
    let mut total = 0.0;
    let mut correction = 0.0;
    for &(v, w) in sphere {
        for (axis, other) in [(ray, sun), (sun, ray)] {
            let v = quadrature::rotate(v, axis);
            let a = 0.0004 + (1.0 - v.dot(axis)).max(0.0);
            let z = 0.0004 + (1.0 - v.dot(other)).max(0.0);
            let incoming = State {
                mu: v.z,
                nu: v.dot(sun),
                ground: l.geometry.hits_ground(s.altitude_km, v.z),
                ..s
            };
            let radiance = ReferenceStencil::new(l.geometry, &l.config, incoming, Some(heights))
                .sample_with(|i| l.radiance[i]);
            let value =
                radiance * phase_weight(c, b.phases(v.dot(ray))) * w * z * z / (a * a + z * z);
            let y = value - correction;
            let next = total + y;
            correction = (next - total) - y;
            total = next;
        }
    }
    total
}
fn main() -> Result<()> {
    let a = Args::parse();
    let m = Manifest::open(&a.source)?;
    let l = m.read_band(&a.source, a.band)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != m.model_fingerprint_fnv1a64 {
        return Err("model mismatch".into());
    }
    let b = &model.bands[a.band];
    let g = l.geometry;
    let heights = radius_nodes(g, &l.config);
    let mut source_config = l.config.clone();
    if !a.height_anchors.is_empty() {
        let mut nodes = heights.clone();
        nodes[0] = 0.0;
        *nodes.last_mut().unwrap() = g.top_height();
        let mut used = std::collections::HashSet::new();
        for anchor in a.height_anchors {
            if !b.profile.iter().any(|v| v.0 == anchor) || anchor <= 0.0 || anchor >= g.top_height()
            {
                return Err("height anchors must be interior profile boundaries".into());
            }
            let i = (1..nodes.len() - 1)
                .filter(|i| !used.contains(i))
                .min_by(|a, b| {
                    (nodes[*a] - anchor)
                        .abs()
                        .total_cmp(&(nodes[*b] - anchor).abs())
                })
                .unwrap();
            nodes[i] = anchor;
            used.insert(i);
        }
        nodes.sort_by(f32::total_cmp);
        source_config.scattering_altitudes_km = nodes;
        source_config.validate(m.bands.len())?;
    }
    let c = &source_config;
    let source_heights = radius_nodes(g, c);
    let sphere = quadrature::sphere(32, 64);
    let mut cache = HashMap::new();
    let mut rows = Vec::new();
    for (h, se, az, above) in [
        (0.2_f32, -6.0_f32, 0.0_f32, 0.5_f32),
        (2.0, -6.0, 180.0, 1.0),
        (30.0, -5.64, 0.0, 0.1),
    ] {
        let e = g.horizon(h).asin() + above.to_radians();
        let sun = se.to_radians();
        let az = az.to_radians();
        let s = State {
            altitude_km: h,
            mu: e.sin(),
            mu_s: sun.sin(),
            nu: e.sin() * sun.sin() + e.cos() * sun.cos() * az.cos(),
            ground: false,
        };
        let length = g.distance(h, s.mu, false);
        let mut trans = 1.0;
        for j in 0..32 {
            let dx = length / 32.0;
            let p = g.advanced(s, (j as f32 + 0.5) * dx);
            let ext = b.coefficients(p.altitude_km).extinction;
            let weight = trans * (-(-ext * dx).exp_m1()) / ext.max(1e-30);
            trans *= (-ext * dx).exp();
            let exact = source(&l, b, &heights, &sphere, p);
            let mut variants = Vec::new();
            for (local, linear) in [(false, false), (true, false), (false, true), (true, true)] {
                let stencil = if local {
                    RayStencil::new_local_source(g, c, p, Some(&source_heights))
                } else {
                    RayStencil::new(g, c, p, Some(&source_heights))
                };
                let stencil = if linear {
                    stencil.with_linear_radius()
                } else {
                    stencil
                };
                for normalized in [false, true] {
                    let value = stencil.sample_with(|i| {
                        let node = g.state_config(i, c);
                        let v = *cache
                            .entry(i)
                            .or_insert_with(|| source(&l, b, &heights, &sphere, node));
                        if normalized {
                            v / scattering(b, node.altitude_km).max(1e-30)
                        } else {
                            v
                        }
                    }) * if normalized {
                        scattering(b, p.altitude_km)
                    } else {
                        1.0
                    };
                    variants.push(value);
                }
            }
            rows.push(serde_json::json!({"pose":[h,se,az.to_degrees(),above],"step":j,"h":p.altitude_km,"weight":weight,"exact":exact,"variants":variants}));
        }
        eprintln!("pose h={h}, sun={se}: {} source nodes", cache.len());
    }
    fs::write(
        a.out,
        serde_json::to_vec_pretty(
            &serde_json::json!({"source":a.source,"source_grid_config":source_config,"band":a.band,"angular":[32,64],"variants":["ray","ray_normalized","local","local_normalized","ray_linear_height","ray_linear_height_normalized","local_linear_height","local_linear_height_normalized"],"rows":rows,"oracle_nodes":cache.len(),"note":"Frozen cumulative LUT. Source nodes and queries use identical quadrature, isolating source interpolation. Coarse 32-step path weights only rank errors, not a transport reference."}),
        )?,
    )?;
    Ok(())
}
