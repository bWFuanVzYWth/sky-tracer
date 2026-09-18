//! Sparse CPU node oracle: compare lookup designs without starting a bake.
use sky_atmosphere_lut::{
    Result,
    asset::{BandLut, Manifest},
    mapping::{State, log_mix, unit},
    model::Model,
    reference_mapping as m,
};
use std::{collections::HashMap, fs, path::Path};

#[path = "support/first_scattering.rs"]
mod first_scattering;
use first_scattering::first;

fn alternative(lut: &BandLut, s: State, mode: u32, fetch: &mut impl FnMut(usize) -> f32) -> f32 {
    let g = lut.geometry;
    let warp = mode / 32;
    let mode = mode % 32;
    if mode == 7 {
        // A research-only smooth overlap between the phase-preserving chart and
        // physical solar corners near the observer's geometric solar horizon.
        let d = (s.mu_s.asin() - g.horizon(s.altitude_km).asin())
            .abs()
            .to_degrees();
        let x = ((d - 1.0) / 2.0).clamp(0.0, 1.0);
        let w = x * x * (3.0 - 2.0 * x);
        if w <= 0.0 {
            return alternative(lut, s, 5 + warp * 32, fetch);
        }
        if w >= 1.0 {
            return alternative(lut, s, 6 + warp * 32, fetch);
        }
        return alternative(lut, s, 5 + warp * 32, fetch) * (1.0 - w)
            + alternative(lut, s, 6 + warp * 32, fetch) * w;
    }
    let c = &lut.config;
    let [nr, nm, ns, nn] = c.scattering;
    let rc = m::height_coord(g, s.altitude_km) * (nr - 1) as f32;
    let rlo = rc.floor() as usize;
    let rt = rc - rlo as f32;
    let hor = g.horizon(s.altitude_km);
    let delta = (s.mu - hor) / if s.ground { 1.0 + hor } else { 1.0 - hor };
    let az = ((s.nu - s.mu * s.mu_s)
        / ((1.0 - s.mu * s.mu) * (1.0 - s.mu_s * s.mu_s))
            .max(1e-30)
            .sqrt())
    .clamp(-1.0, 1.0);
    let mut result = 0.0;
    let mut radial = [0.0; 2];
    for (r, radial_value) in radial.iter_mut().enumerate() {
        let rw = if r == 0 { 1.0 - rt } else { rt };
        if rw <= 0.0 {
            continue;
        }
        let ri = (rlo + r).min(nr - 1);
        let h = m::height(g, unit(ri, nr));
        let hor = g.horizon(h);
        let mu = (hor + delta * if s.ground { 1.0 + hor } else { 1.0 - hor }).clamp(-1.0, 1.0);
        let target_sun = if mode >= 3 && 1.0 - s.mu * s.mu > 1e-7 {
            (mu * s.nu
                + ((1.0 - mu * mu) / (1.0 - s.mu * s.mu)).max(0.0).sqrt() * (s.mu_s - s.mu * s.nu))
                .clamp(-1.0, 1.0)
        } else {
            s.mu_s
        };
        let target_nu = if mode >= 3 {
            s.nu
        } else {
            mu * s.mu_s + ((1.0 - mu * mu) * (1.0 - s.mu_s * s.mu_s)).max(0.0).sqrt() * az
        };
        let target = State {
            altitude_km: h,
            mu,
            mu_s: target_sun,
            nu: target_nu,
            ..s
        };
        let az = ((target.nu - mu * target_sun)
            / ((1.0 - mu * mu) * (1.0 - target_sun * target_sun))
                .max(1e-30)
                .sqrt())
        .clamp(-1.0, 1.0);
        let sc = m::solar_coord(g, h, target_sun) * (ns - 1) as f32;
        let slo = sc.floor() as usize;
        let st = sc - slo as f32;
        let mut sv = [0.0; 2];
        for (j, v) in sv.iter_mut().enumerate() {
            let si = (slo + j).min(ns - 1);
            let mus = m::solar_cosine(g, h, unit(si, ns));
            let nu = mu * mus + ((1.0 - mu * mu) * (1.0 - mus * mus)).max(0.0).sqrt() * az;
            let mut corner = State {
                altitude_km: h,
                mu,
                mu_s: mus,
                nu,
                ..s
            };
            if mode == 4 || mode == 6 {
                corner = target;
            }
            let nc = phase_unit(m::phase_coord(g, corner), warp, true) * (nn - 1) as f32;
            let nlo = nc.floor() as usize;
            let nt = nc - nlo as f32;
            let mut nv = [0.0; 4];
            for (k, out) in nv.iter_mut().enumerate() {
                let ni = (nlo + k).saturating_sub(1).min(nn - 1);
                let target = if mode == 0 || mode >= 3 {
                    corner
                } else {
                    State {
                        nu: m::phase_cosine(g, corner, unit(ni, nn)),
                        ..corner
                    }
                };
                let mut mc = g.optical_cone_coord(target, nm);
                if mode == 2 {
                    let center = target.mu_s * target.nu;
                    let extent = ((1.0 - target.mu_s * target.mu_s)
                        * (1.0 - target.nu * target.nu))
                        .max(0.0)
                        .sqrt();
                    let lo = if s.ground {
                        center - extent
                    } else {
                        (center - extent).max(hor)
                    };
                    let hi = if s.ground {
                        (center + extent).min(hor)
                    } else {
                        center + extent
                    };
                    let scale = (16.0 / (g.bottom + h)).sqrt();
                    let warp = |x: f32| (x - hor).abs() / ((x - hor).abs() + scale);
                    let (a, b) = (warp(hi), warp(lo));
                    if (b - a).abs() > 1e-8 {
                        let u = (warp(mu) - a) / (b - a);
                        let ng = nm / 4;
                        mc = if s.ground {
                            u * (ng - 1) as f32
                        } else {
                            ng as f32 + u * (nm - ng - 1) as f32
                        };
                    }
                }
                let ng = nm / 4;
                let low = if s.ground { 0 } else { ng };
                let high = if s.ground { ng - 1 } else { nm - 1 };
                let mi = (mc.floor() as usize).clamp(low, high - 1);
                let mt = (mc - mi as f32).clamp(-8.0, 9.0);
                let a = fetch(((ri * nm + mi) * ns + si) * nn + ni);
                let b = fetch(((ri * nm + mi + 1) * ns + si) * nn + ni);
                *out = (a * (1.0 - mt) + b * mt).max(0.0);
            }
            *v = if warp >= 2 && nv.iter().all(|x| *x >= 0.0) {
                (m::cubic_phase(nv.map(|x| (x + 1e-30).ln()), nt).exp() - 1e-30).max(0.0)
            } else {
                m::cubic_phase(nv, nt)
            };
        }
        *radial_value = log_mix(sv[0], sv[1], st);
        result += rw * *radial_value;
    }
    if mode >= 5 && radial[0] > 0.0 && radial[1] > 0.0 {
        return log_mix(radial[0], radial[1], rt);
    }
    result
}

// Preserve exact chart boundaries while redistributing only the crossing region.
fn phase_unit(u: f32, warp: u32, inverse: bool) -> f32 {
    if warp.is_multiple_of(2) || !(0.5..=0.875).contains(&u) {
        return u;
    }
    let f = ((u - 0.5) / 0.375).clamp(0.0, 1.0);
    let fraction = if inverse {
        let t = 0.5 - 0.5 * (std::f32::consts::PI * f).cos();
        let a = t.cbrt();
        let b = (1.0 - t).cbrt();
        a / (a + b)
    } else {
        let a = f * f * f;
        let b = (1.0 - f) * (1.0 - f) * (1.0 - f);
        let t = a / (a + b);
        (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos() / std::f32::consts::PI
    };
    0.5 + 0.375 * fraction
}

fn main() -> Result<()> {
    let source = Path::new("out/lut_reference_v5c_first");
    let mut lut = Manifest::open(source)?.read_band(source, 17)?;
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let mut poses: Vec<[f32; 4]> =
        serde_json::from_slice(&fs::read("scripts/reference_quality_suspect_poses.json")?)?;
    for sun in [0.0, 20.0, 60.0, 85.0, 89.0] {
        for above in [0.05, 1.0, 3.0, 10.0, 20.9544] {
            poses.push([0.2, sun, 0.0, above]);
        }
    }
    let mut cache = HashMap::new();
    let mut rows = Vec::new();
    for [h, sun, az, above] in poses {
        let e = lut.geometry.horizon(h).asin() + above.to_radians();
        let se = sun.to_radians();
        let s = State {
            altitude_km: h,
            mu: e.sin(),
            mu_s: se.sin(),
            nu: e.sin() * se.sin() + e.cos() * se.cos() * az.to_radians().cos(),
            ground: false,
        };
        let Some((s, _)) = lut.geometry.atmosphere_entry(s) else {
            continue;
        };
        let direct = first(&lut, &model.bands[17], s);
        let original = m::sample_with(lut.geometry, &lut.config, s, |i| lut.radiance[i]);
        let mut virtual_node = |i| {
            *cache.entry(i).or_insert_with(|| {
                first(
                    &lut,
                    &model.bands[17],
                    lut.geometry.state_config(i, &lut.config),
                )
            })
        };
        let old_oracle = m::sample_with(lut.geometry, &lut.config, s, &mut virtual_node);
        let mut variants = Vec::new();
        for mode in 0..8 {
            let oracle = alternative(&lut, s, mode, &mut virtual_node);
            let baked = alternative(&lut, s, mode, &mut |i| lut.radiance[i]);
            variants.push(serde_json::json!({"mode":mode,"oracle":oracle,"baked":baked}));
        }
        let row = serde_json::json!({"h":h,"sun":sun,"az":az,"above":above,"direct":direct,"old_baked":original,"old_oracle":old_oracle,"variants":variants});
        eprintln!("{row}");
        rows.push(row);
    }
    fs::create_dir_all("out/lut_v6_design")?;
    fs::write(
        "out/lut_v6_design/interpolation.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"rows":rows,"oracle_nodes":cache.len(),"modes":["physical solar corners, shared cone coordinate","physical solar and phase projection","physical solar with bounded view extrapolation","ray-frame height transfer, physical solar corners","ray-frame height transfer, shared solar chart","mode 3 with positive log-height","mode 4 with positive log-height","smooth experimental overlap of modes 5 and 6"]}),
        )?,
    )?;
    let original = lut.config.clone();
    let mut allocation = Vec::new();
    for (name, dims, warp) in [
        ("base", [80, 32, 193, 257], 0u32),
        ("height_x2", [159, 32, 193, 257], 0),
        ("view_x2", [80, 64, 193, 257], 0),
        ("solar_x2", [80, 32, 385, 257], 0),
        ("phase_x2", [80, 32, 193, 513], 0),
        ("solar_x4", [80, 32, 769, 257], 0),
        ("cross_cubic", [80, 32, 193, 257], 1),
        ("cross_cubic_solar_x2", [80, 32, 385, 257], 1),
        ("log_phase", [80, 32, 193, 257], 2),
        ("log_phase_solar_x2", [80, 32, 385, 257], 2),
        ("log_phase_phase_x2", [80, 32, 193, 513], 2),
        ("log_phase_cross_cubic", [80, 32, 193, 257], 3),
    ] {
        lut.config.scattering = dims;
        let mut cache = HashMap::new();
        let mut rows = Vec::new();
        for (h, sun, above) in [
            (30.0_f32, -5.7_f32, 0.1_f32),
            (30.0, -5.64, 0.1),
            (30.0, -5.6, 0.1),
            (120.0, -11.02, 0.1),
            (120.0, -11.0, 0.1),
            (0.2, 20.0, 20.9544),
            (0.2, 85.0, 0.1),
        ] {
            let e = lut.geometry.horizon(h).asin() + above.to_radians();
            let se = sun.to_radians();
            let s = State {
                altitude_km: h,
                mu: e.sin(),
                mu_s: se.sin(),
                nu: (e - se).cos(),
                ground: false,
            };
            let direct = first(&lut, &model.bands[17], s);
            let mut fetch = |i| {
                *cache.entry(i).or_insert_with(|| {
                    first(&lut, &model.bands[17], {
                        let mut state = lut.geometry.state_config(i, &lut.config);
                        if !warp.is_multiple_of(2) {
                            state.nu = m::phase_cosine(
                                lut.geometry,
                                state,
                                phase_unit(unit(i % dims[3], dims[3]), warp, false),
                            );
                            state.mu = lut.geometry.optical_cone_view(
                                state,
                                i / (dims[2] * dims[3]) % dims[1],
                                dims[1],
                            );
                        }
                        state
                    })
                })
            };
            let candidate = alternative(&lut, s, 7 + warp * 32, &mut fetch);
            rows.push(serde_json::json!({"h":h,"sun":sun,"above":above,"direct":direct,"candidate":candidate,"relative_error":candidate/direct.max(1e-30)-1.0}));
        }
        eprintln!("allocation {name}: {rows:?}");
        allocation.push(
            serde_json::json!({"name":name,"dims":dims,"oracle_nodes":cache.len(),"rows":rows}),
        );
    }
    lut.config = original;
    fs::write(
        "out/lut_v6_design/allocation.json",
        serde_json::to_vec_pretty(&allocation)?,
    )?;
    Ok(())
}
