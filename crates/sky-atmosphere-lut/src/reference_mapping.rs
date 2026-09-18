//! Reference chart with fixed front / horizon-crossing / back phase regions.
//! Each region has fixed indices: a cone becoming tangent to the horizon never
//! crosses an arbitrary phase cell. Radiance (including phase) remains stored.
use crate::{
    config::{BakeConfig, SolarInterpolation},
    mapping::{Geometry, State, log_mix, unit},
};
use std::f32::consts::PI;

/// Harmonic-mean tangents preserve monotonicity and give matching derivatives
/// at adjacent phase cells. Constant padding handles the two domain endpoints.
pub fn cubic_phase(v: [f32; 4], t: f32) -> f32 {
    let slope = |a: f32, b: f32| {
        if a == 0.0 || b == 0.0 || a.is_sign_positive() != b.is_sign_positive() {
            return 0.0;
        }
        let lo = a.abs().min(b.abs());
        let hi = a.abs().max(b.abs());
        (2.0 * lo / (1.0 + lo / hi)).copysign(a)
    };
    let d = v[2] - v[1];
    let a = slope(v[1] - v[0], d);
    let b = slope(d, v[3] - v[2]);
    let t2 = t * t;
    let t3 = t2 * t;
    ((2.0 * t3 - 3.0 * t2 + 1.0) * v[1]
        + (t3 - 2.0 * t2 + t) * a
        + (-2.0 * t3 + 3.0 * t2) * v[2]
        + (t3 - t2) * b)
        .clamp(v[1].min(v[2]), v[1].max(v[2]))
}

pub fn log_cubic_phase(v: [f32; 4], t: f32) -> f32 {
    if v[1] == v[2] || t <= 0.0 {
        return v[1];
    }
    if t >= 1.0 {
        return v[2];
    }
    if v.iter().any(|x| *x < 0.0) {
        return cubic_phase(v, t);
    }
    (cubic_phase(v.map(|x| (x + 1e-30).ln()), t).exp() - 1e-30)
        .max(0.0)
        .clamp(v[1].min(v[2]), v[1].max(v[2]))
}

fn invert(f: impl Fn(f32) -> f32, u: f32, mut lo: f32, mut hi: f32) -> f32 {
    if u <= 0.0 {
        return lo;
    }
    if u >= 1.0 {
        return hi;
    }
    for _ in 0..24 {
        let mid = (lo + hi) * 0.5;
        if f(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

pub fn height_coord(g: Geometry, h: f32) -> f32 {
    let x = h.clamp(0.0, g.top_height()) / g.top_height();
    let scale = (0.01 / g.top_height()).sqrt();
    0.35 * (1.0 + x.sqrt() / scale).ln() / (1.0 + 1.0 / scale).ln() + 0.65 * x
}
pub fn height(g: Geometry, u: f32) -> f32 {
    // Invert in sqrt(height) to retain sub-metre nodes with f32 arithmetic.
    let root = invert(|x| height_coord(g, x * x), u, 0.0, g.top_height().sqrt());
    root * root
}
pub fn radius(g: Geometry, c: &BakeConfig, index: usize) -> f32 {
    c.scattering_altitudes_km
        .get(index)
        .copied()
        .unwrap_or_else(|| height(g, unit(index, c.scattering[0])))
}
pub fn radius_nodes(g: Geometry, c: &BakeConfig) -> Vec<f32> {
    (0..c.scattering[0]).map(|i| radius(g, c, i)).collect()
}
pub fn radius_coord(g: Geometry, c: &BakeConfig, h: f32) -> f32 {
    let nodes = &c.scattering_altitudes_km;
    if nodes.is_empty() {
        return height_coord(g, h) * (c.scattering[0] - 1) as f32;
    }
    let hi = nodes.partition_point(|v| *v <= h).clamp(1, nodes.len() - 1);
    if c.height_interpolation == crate::config::HeightInterpolation::ReferenceCdf {
        let a = height_coord(g, nodes[hi - 1]);
        let b = height_coord(g, nodes[hi]);
        return (hi - 1) as f32 + ((height_coord(g, h) - a) / (b - a)).clamp(0.0, 1.0);
    }
    (hi - 1) as f32 + ((h - nodes[hi - 1]) / (nodes[hi] - nodes[hi - 1])).clamp(0.0, 1.0)
}
pub fn radius_coord_linear(g: Geometry, c: &BakeConfig, h: f32) -> f32 {
    let coordinate = radius_coord(g, c, h);
    let i = (coordinate.floor() as usize).min(c.scattering[0] - 2);
    let a = radius(g, c, i);
    let b = radius(g, c, i + 1);
    i as f32 + ((h - a) / (b - a)).clamp(0.0, 1.0)
}
fn atan_cdf(e: f32, center: f32, scale: f32) -> f32 {
    let a = ((-PI * 0.5 - center) / scale).atan();
    (((e - center) / scale).atan() - a) / (((PI * 0.5 - center) / scale).atan() - a)
}
pub fn solar_coord(g: Geometry, h: f32, mu: f32) -> f32 {
    let e = mu.clamp(-1.0, 1.0).asin();
    let hor = g.horizon(h).asin();
    let d = PI * 0.5 - e;
    let cap_scale = 5.0_f32.to_radians();
    let cap = 1.0 - (d / (d + cap_scale)).sqrt() / (PI / (PI + cap_scale)).sqrt();
    (0.25 * (e / PI + 0.5)
        + 0.20 * atan_cdf(e, hor, 2.0_f32.to_radians())
        + 0.20 * atan_cdf(e, hor - 6.0_f32.to_radians(), 6.0_f32.to_radians())
        + 0.15 * atan_cdf(e, -hor, 0.5_f32.to_radians())
        + 0.20 * cap)
        .clamp(0.0, 1.0)
}
pub fn solar_cosine(g: Geometry, h: f32, u: f32) -> f32 {
    invert(|e| solar_coord(g, h, e.sin()), u, -PI * 0.5, PI * 0.5).sin()
}

// The two exact phase angles at which a solar cone touches the horizon.
fn bounds(g: Geometry, s: State) -> [f32; 4] {
    let alpha = s.mu_s.clamp(-1.0, 1.0).acos();
    let beta = g.horizon(s.altitude_km).acos();
    let a = (beta - alpha).abs();
    let b = (alpha + beta).min(2.0 * PI - alpha - beta).min(PI);
    let (lo, hi) = if s.ground {
        ((beta - alpha).max(0.0), (2.0 * PI - beta - alpha).min(PI))
    } else {
        ((alpha - beta).max(0.0), (alpha + beta).min(PI))
    };
    [lo, a.max(lo).min(hi), b.max(lo).min(hi), hi]
}
fn phase_cdf(theta: f32) -> f32 {
    // Keep forward resolution, with a uniform floor over the crossing region.
    0.65 * (1.0 + theta / 2.0_f32.to_radians()).ln() / 91.0_f32.ln() + 0.35 * theta / PI
}
const SPLITS: [f32; 4] = [0.0, 0.5, 0.875, 1.0];
pub fn phase_cosine(g: Geometry, s: State, u: f32) -> f32 {
    let b = bounds(g, s);
    phase_cosine_in_bounds(b, u)
}
fn phase_cosine_in_bounds(b: [f32; 4], u: f32) -> f32 {
    let k = if u < SPLITS[1] {
        0
    } else if u < SPLITS[2] {
        1
    } else {
        2
    };
    let fraction = ((u - SPLITS[k]) / (SPLITS[k + 1] - SPLITS[k])).clamp(0.0, 1.0);
    if b[k] == b[k + 1] {
        return b[k].cos();
    }
    let t = match k {
        0 => 1.0 - (1.0 - fraction) * (1.0 - fraction),
        1 => 0.5 - 0.5 * (PI * fraction).cos(),
        _ => fraction * fraction,
    };
    let lo = phase_cdf(b[k]);
    let hi = phase_cdf(b[k + 1]);
    let theta = if t <= 0.0 {
        b[k]
    } else if t >= 1.0 {
        b[k + 1]
    } else {
        invert(
            |x| (phase_cdf(x) - lo) / (hi - lo).max(1e-20),
            t,
            b[k],
            b[k + 1],
        )
    };
    theta.cos()
}

#[derive(Clone, Copy, Default)]
struct StencilRow {
    radius_weight: f32,
    solar_t: f32,
    view_t: f32,
    phase_t: f32,
    lower: [u32; 2],
    upper: [u32; 2],
    phase: [u32; 4],
}
/// Geometry is compiled once and reused for every wavelength of a sample.
#[derive(Clone)]
pub struct ReferenceStencil {
    rows: [StencilRow; 2],
    solar_interpolation: SolarInterpolation,
    ray: Option<Box<crate::ray_mapping::RayStencil>>,
}
impl ReferenceStencil {
    pub fn new(g: Geometry, c: &BakeConfig, s: State, heights: Option<&[f32]>) -> Self {
        if c.mapping == crate::config::CoordinateMapping::RayAlignedReference {
            return Self {
                rows: [StencilRow::default(); 2],
                solar_interpolation: c.solar_interpolation,
                ray: Some(Box::new(crate::ray_mapping::RayStencil::new(
                    g, c, s, heights,
                ))),
            };
        }
        let [nr, nm, ns, nn] = c.scattering;
        let rc = height_coord(g, s.altitude_km) * (nr - 1) as f32;
        let rlo = rc.floor() as usize;
        let rt = rc - rlo as f32;
        let hor = g.horizon(s.altitude_km);
        let delta = (s.mu - hor) / (if s.ground { 1.0 + hor } else { 1.0 - hor });
        let extent = ((1.0 - s.mu * s.mu) * (1.0 - s.mu_s * s.mu_s))
            .max(0.0)
            .sqrt();
        let az = ((s.nu - s.mu * s.mu_s) / extent.max(1e-20)).clamp(-1.0, 1.0);
        let mut rows = [StencilRow::default(); 2];
        for (r, row) in rows.iter_mut().enumerate() {
            let rw = if r == 0 { 1.0 - rt } else { rt };
            if rw <= 0.0 {
                continue;
            }
            let ri = (rlo + r).min(nr - 1);
            let h = heights.map_or_else(|| height(g, unit(ri, nr)), |v| v[ri]);
            // Anchor the two Earth boundary branches when changing radius. Recompute
            // solar coordinates at this radius, preserving the physical sun zenith.
            let target_hor = g.horizon(h);
            let mu = (target_hor
                + delta
                    * (if s.ground {
                        1.0 + target_hor
                    } else {
                        1.0 - target_hor
                    }))
            .clamp(-1.0, 1.0);
            let nu = mu * s.mu_s + ((1.0 - mu * mu) * (1.0 - s.mu_s * s.mu_s)).max(0.0).sqrt() * az;
            let point = State {
                altitude_km: h,
                mu,
                nu,
                ..s
            };
            let sc = solar_coord(g, h, s.mu_s) * (ns - 1) as f32;
            let slo = sc.floor() as usize;
            let st = sc - slo as f32;
            let nc = phase_coord(g, point) * (nn - 1) as f32;
            let nlo = nc.floor() as usize;
            let nt = nc - nlo as f32;
            let mc = g.optical_cone_coord(point, nm);
            let mi = mc.floor() as usize;
            let mt = mc - mi as f32;

            row.radius_weight = rw;
            row.solar_t = st;
            row.phase_t = nt;
            row.view_t = mt;
            row.phase = std::array::from_fn(|k| (nlo + k).saturating_sub(1).min(nn - 1) as u32);
            for j in 0..2 {
                let si = (slo + j).min(ns - 1);
                row.lower[j] = (((ri * nm + mi) * ns + si) * nn) as u32;
                row.upper[j] = (((ri * nm + (mi + 1).min(nm - 1)) * ns + si) * nn) as u32;
            }
        }
        Self {
            rows,
            solar_interpolation: c.solar_interpolation,
            ray: None,
        }
    }
    pub fn sample_with(&self, mut fetch: impl FnMut(usize) -> f32) -> f32 {
        if let Some(ray) = &self.ray {
            return ray.sample_with(fetch);
        }
        let mut value = 0.0;
        for row in &self.rows {
            if row.radius_weight <= 0.0 {
                continue;
            }
            let mut solar = [0.0; 2];
            for (j, result) in solar.iter_mut().enumerate() {
                let phase = row.phase.map(|ni| {
                    let a = fetch((row.lower[j] + ni) as usize);
                    let b = if row.view_t > 0.0 {
                        fetch((row.upper[j] + ni) as usize)
                    } else {
                        a
                    };
                    a * (1.0 - row.view_t) + b * row.view_t
                });
                *result = cubic_phase(phase, row.phase_t);
            }
            value += row.radius_weight
                * match self.solar_interpolation {
                    SolarInterpolation::Linear => {
                        solar[0] * (1.0 - row.solar_t) + solar[1] * row.solar_t
                    }
                    SolarInterpolation::LogRadiance => log_mix(solar[0], solar[1], row.solar_t),
                };
        }
        value
    }
}
pub fn sample_with(g: Geometry, c: &BakeConfig, s: State, fetch: impl FnMut(usize) -> f32) -> f32 {
    ReferenceStencil::new(g, c, s, None).sample_with(fetch)
}
pub fn phase_coord(g: Geometry, s: State) -> f32 {
    let b = bounds(g, s);
    let theta = s.nu.clamp(-1.0, 1.0).acos().clamp(b[0], b[3]);
    // A horizon point must stay in the crossing chart, including its endpoints.
    let at_horizon = (s.mu - g.horizon(s.altitude_km)).abs() < 1e-6;
    let k = if !at_horizon && theta < b[1] {
        0
    } else if !at_horizon && theta > b[2] {
        2
    } else {
        1
    };
    let a = phase_cdf(b[k]);
    let z = phase_cdf(b[k + 1]);
    let t = ((phase_cdf(theta) - a) / (z - a).max(1e-20)).clamp(0.0, 1.0);
    let fraction = match k {
        0 => 1.0 - (1.0 - t).sqrt(),
        1 => (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos() / PI,
        _ => t.sqrt(),
    };
    SPLITS[k] + fraction * (SPLITS[k + 1] - SPLITS[k])
}

/// Packed after the legacy 4096 phase samples and nn angular samples. Caching
/// inverse CDFs avoids doing bisection in every GPU transport lookup.
pub fn append_nodes(g: Geometry, c: &BakeConfig, packed: &mut Vec<f32>) {
    if !c.mapping.is_reference() {
        return;
    }
    let [_, _, ns, nn] = c.scattering;
    let heights = radius_nodes(g, c);
    let solar: Vec<_> = heights
        .iter()
        .flat_map(|&h| (0..ns).map(move |i| solar_cosine(g, h, unit(i, ns))))
        .collect();
    packed.extend_from_slice(&heights);
    packed.extend_from_slice(&solar);
    for (ri, &h) in heights.iter().enumerate() {
        for si in 0..ns {
            for ground in [false, true] {
                let s = State {
                    altitude_km: h,
                    mu: 0.0,
                    mu_s: solar[ri * ns + si],
                    nu: 0.0,
                    ground,
                };
                let b = bounds(g, s);
                packed.extend((0..nn).map(|ni| phase_cosine_in_bounds(b, unit(ni, nn))));
            }
        }
    }
}

/// Display needs cached radii, and the ray chart also needs solar corners.
/// It never uploads the full phase-node cache or active bake work list.
pub fn append_display_nodes(g: Geometry, c: &BakeConfig, packed: &mut Vec<f32>) {
    if c.mapping.is_reference() {
        let heights = radius_nodes(g, c);
        packed.extend_from_slice(&heights);
        if c.mapping == crate::config::CoordinateMapping::RayAlignedReference {
            for h in heights {
                packed.extend(
                    (0..c.scattering[2]).map(|si| solar_cosine(g, h, unit(si, c.scattering[2]))),
                );
            }
        }
    }
}
