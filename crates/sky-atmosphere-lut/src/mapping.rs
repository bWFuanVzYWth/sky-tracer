//! CPU lookup mirror of bake.wgsl. Height-based arithmetic avoids subtracting
//! planet-scale squared radii when computing short paths or grazing rays.
use crate::config::{BakeConfig, ConeMapping, CoordinateMapping, SolarInterpolation};
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub bottom: f32,
    pub top: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub altitude_km: f32,
    pub mu: f32,
    pub mu_s: f32,
    pub nu: f32,
    pub ground: bool,
}

impl Geometry {
    pub fn height_mapped(self, u: f32, mapping: CoordinateMapping) -> f32 {
        if mapping.is_reference() {
            crate::reference_mapping::height(self, u)
        } else {
            self.height(u)
        }
    }
    pub fn height_coord_mapped(self, h: f32, mapping: CoordinateMapping) -> f32 {
        if mapping.is_reference() {
            crate::reference_mapping::height_coord(self, h)
        } else {
            self.height_coord(h)
        }
    }
    /// Clip a ray from space to the atmospheric shell. Vacuum travel changes
    /// the local zenith and solar zenith, but not radiance or scattering angle.
    pub fn atmosphere_entry(self, s: State) -> Option<(State, f32)> {
        if s.altitude_km <= self.top_height() {
            return Some((s, 0.0));
        }
        let radius = self.bottom + s.altitude_km;
        let b = radius * s.mu;
        let c = (s.altitude_km - self.top_height()) * (radius + self.top);
        let discriminant = b * b - c;
        if b >= 0.0 || discriminant <= 0.0 {
            return None;
        }
        let d = c / (-b + discriminant.sqrt());
        let mu = ((b + d) / self.top).clamp(-1.0, 1.0);
        Some((
            State {
                altitude_km: self.top_height(),
                mu,
                mu_s: ((radius * s.mu_s + d * s.nu) / self.top).clamp(-1.0, 1.0),
                nu: s.nu,
                ground: self.hits_ground(self.top_height(), mu),
            },
            d,
        ))
    }
    pub fn coords_config(self, s: State, c: &BakeConfig) -> [f32; 4] {
        let mut coords = self.coords_mapped(s, c.scattering, c.mapping);
        if !c.scattering_altitudes_km.is_empty() {
            coords[0] = crate::reference_mapping::radius_coord(self, c, s.altitude_km);
        }
        if c.mapping != CoordinateMapping::Legacy && c.cone_mapping == ConeMapping::OpticalDistance
        {
            coords[1] = self.optical_cone_coord(s, c.scattering[1]);
        }
        coords
    }
    pub fn state_config(self, i: usize, c: &BakeConfig) -> State {
        let mut s = self.state_mapped(i, c.scattering, c.mapping);
        if !c.scattering_altitudes_km.is_empty() {
            let [_, nm, ns, nn] = c.scattering;
            s.altitude_km = crate::reference_mapping::radius(self, c, i / (nm * ns * nn));
            s.mu_s = self.solar_cosine_mapped(s.altitude_km, unit(i / nn % ns, ns), c.mapping);
            s.nu = crate::reference_mapping::phase_cosine(self, s, unit(i % nn, nn));
            s.mu = self.cone_view(s.altitude_km, s.mu_s, s.nu, i / (ns * nn) % nm, nm);
        }
        if c.mapping != CoordinateMapping::Legacy && c.cone_mapping == ConeMapping::OpticalDistance
        {
            s.mu = self.optical_cone_view(
                s,
                i / (c.scattering[2] * c.scattering[3]) % c.scattering[1],
                c.scattering[1],
            );
        }
        s
    }
    fn optical_cone_bounds(self, s: State) -> (f32, f32, f32) {
        let (center, extent, _) = self.cone_horizon(s.altitude_km, s.mu_s, s.nu);
        let hor = self.horizon(s.altitude_km);
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
        let scale = (16.0 / (self.bottom + s.altitude_km)).sqrt();
        let warp = |mu: f32| {
            let delta = (mu - hor).abs();
            delta / (delta + scale)
        };
        (warp(hi), warp(lo), scale)
    }
    pub fn optical_cone_view(self, s: State, index: usize, size: usize) -> f32 {
        let ng = (size / 4).max(2);
        let u = if s.ground {
            unit(index, ng)
        } else {
            unit(index - ng, size - ng)
        };
        let (a, b, scale) = self.optical_cone_bounds(s);
        let x = a + (b - a) * u;
        let delta = scale * x / (1.0 - x).max(1e-20);
        let (center, extent, _) = self.cone_horizon(s.altitude_km, s.mu_s, s.nu);
        (self.horizon(s.altitude_km) + if s.ground { -delta } else { delta })
            .clamp((center - extent).max(-1.0), (center + extent).min(1.0))
    }
    pub fn optical_cone_coord(self, s: State, size: usize) -> f32 {
        let ng = (size / 4).max(2);
        let (a, b, scale) = self.optical_cone_bounds(s);
        let delta = (s.mu - self.horizon(s.altitude_km)).abs();
        let x = delta / (delta + scale);
        let u = if (b - a).abs() > 1e-10 {
            ((x - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if s.ground {
            u * (ng - 1) as f32
        } else {
            ng as f32 + u * (size - ng - 1) as f32
        }
    }
    pub fn top_height(self) -> f32 {
        self.top - self.bottom
    }
    pub fn rho_squared(self, h: f32) -> f32 {
        h * (2.0 * self.bottom + h)
    }
    pub fn horizon(self, h: f32) -> f32 {
        -self.rho_squared(h).max(0.0).sqrt() / (self.bottom + h)
    }
    pub fn hits_ground(self, h: f32, mu: f32) -> bool {
        mu < 0.0 && mu <= self.horizon(h)
    }
    pub fn distance(self, h: f32, mu: f32, ground: bool) -> f32 {
        let b = (self.bottom + h) * mu;
        if ground {
            let c = self.rho_squared(h);
            let denominator = -b + (b * b - c).max(0.0).sqrt();
            if denominator > 0.0 {
                (c / denominator).max(0.0)
            } else {
                0.0
            }
        } else {
            let c = (self.top_height() - h) * (2.0 * self.bottom + self.top_height() + h);
            let root = (b * b + c).max(0.0).sqrt();
            if b > 0.0 {
                (c / (root + b)).max(0.0)
            } else {
                (-b + root).max(0.0)
            }
        }
    }
    pub fn height(self, x: f32) -> f32 {
        let rho2 = x * x * self.rho_squared(self.top_height());
        rho2 / ((self.bottom * self.bottom + rho2).sqrt() + self.bottom)
    }
    pub fn height_coord(self, h: f32) -> f32 {
        (self.rho_squared(h) / self.rho_squared(self.top_height()))
            .max(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
    }
    pub fn view(self, h: f32, index: usize, size: usize) -> (f32, bool) {
        let half = size / 2;
        let ground = index < half;
        let x = unit(index % half, half);
        let x = x.powi(3) / (x.powi(3) + (1.0 - x).powi(3));
        let horizon = self.horizon(h);
        (
            if ground {
                horizon - (1.0 + horizon) * x
            } else {
                horizon + (1.0 - horizon) * x
            },
            ground,
        )
    }
    pub fn view_coord(self, h: f32, mu: f32, ground: bool, size: usize) -> f32 {
        let horizon = self.horizon(h);
        let half = size / 2;
        let x = if ground {
            (horizon - mu) / (1.0 + horizon)
        } else {
            (mu - horizon) / (1.0 - horizon)
        };
        let x = x.clamp(0.0, 1.0);
        let x = x.cbrt() / (x.cbrt() + (1.0 - x).cbrt());
        x * (half - 1) as f32 + if ground { 0.0 } else { half as f32 }
    }
    pub fn state(self, index: usize, dims: [usize; 4]) -> State {
        self.state_mapped(index, dims, CoordinateMapping::Legacy)
    }
    pub fn state_mapped(self, index: usize, dims: [usize; 4], mapping: CoordinateMapping) -> State {
        let [_, nm, ns, nn] = dims;
        let h = self.height_mapped(unit(index / (nm * ns * nn), dims[0]), mapping);
        if mapping != CoordinateMapping::Legacy {
            let mu_s = self.solar_cosine_mapped(h, unit(index / nn % ns, ns), mapping);
            let mi = index / (ns * nn) % nm;
            let ground = mi < (nm / 4).max(2);
            let nu = if mapping.is_reference() {
                crate::reference_mapping::phase_cosine(
                    self,
                    State {
                        altitude_km: h,
                        mu: 0.0,
                        mu_s,
                        nu: 0.0,
                        ground,
                    },
                    unit(index % nn, nn),
                )
            } else {
                self.cone_cosine(h, mu_s, scattering_cosine(unit(index % nn, nn)), ground)
            };
            let mu = self.cone_view(h, mu_s, nu, mi, nm);
            return State {
                altitude_km: h,
                mu,
                mu_s,
                nu,
                ground,
            };
        }
        let (mu, ground) = self.view(h, index / (ns * nn) % nm, nm);
        let mu_s = sun_cosine(unit(index / nn % ns, ns));
        let extent = ((1.0 - mu * mu) * (1.0 - mu_s * mu_s)).max(0.0).sqrt();
        let nu = (1.0 - 2.0 * unit(index % nn, nn).powi(3))
            .clamp(mu * mu_s - extent, mu * mu_s + extent);
        State {
            altitude_km: h,
            mu,
            mu_s,
            nu,
            ground,
        }
    }
    pub fn coords(self, s: State, dims: [usize; 4]) -> [f32; 4] {
        self.coords_mapped(s, dims, CoordinateMapping::Legacy)
    }
    pub fn coords_mapped(self, s: State, dims: [usize; 4], mapping: CoordinateMapping) -> [f32; 4] {
        if mapping != CoordinateMapping::Legacy {
            return [
                self.height_coord_mapped(s.altitude_km, mapping) * (dims[0] - 1) as f32,
                self.cone_coord(s, dims[1]),
                self.solar_coord_mapped(s.altitude_km, s.mu_s, mapping) * (dims[2] - 1) as f32,
                (if mapping.is_reference() {
                    crate::reference_mapping::phase_coord(self, s)
                } else {
                    scattering_coord(s.nu)
                }) * (dims[3] - 1) as f32,
            ];
        }
        [
            self.height_coord(s.altitude_km) * (dims[0] - 1) as f32,
            self.view_coord(s.altitude_km, s.mu, s.ground, dims[1]),
            sun_coord(s.mu_s) * (dims[2] - 1) as f32,
            ((1.0 - s.nu.clamp(-1.0, 1.0)) * 0.5).cbrt() * (dims[3] - 1) as f32,
        ]
    }
    pub fn solar_cosine(self, h: f32, u: f32) -> f32 {
        let hor = self.horizon(h);
        let x = 2.0 * u - 1.0;
        (hor + x * x.abs() * if x < 0.0 { 1.0 + hor } else { 1.0 - hor }).clamp(-1.0, 1.0)
    }
    pub fn solar_coord(self, h: f32, mu: f32) -> f32 {
        let hor = self.horizon(h);
        let delta = mu - hor;
        let x = delta / if delta < 0.0 { 1.0 + hor } else { 1.0 - hor };
        (0.5 + 0.5 * x.abs().sqrt().copysign(x)).clamp(0.0, 1.0)
    }
    pub fn solar_cosine_mapped(self, h: f32, u: f32, mapping: CoordinateMapping) -> f32 {
        if mapping.is_reference() {
            return crate::reference_mapping::solar_cosine(self, h, u);
        }
        if mapping != CoordinateMapping::SunAlignedAngular {
            return self.solar_cosine(h, u);
        }
        // A fixed normalized mapping lets later refinement subdivide both
        // twilight and zenith intervals, preserving consistency over the domain.
        let index = u * 96.0;
        if index <= 62.0 {
            return self.solar_cosine(h, index / 64.0);
        }
        let theta = self.solar_cosine(h, 62.0 / 64.0).acos();
        let x = (index - 62.0) / 34.0;
        (theta * (1.0 - x) * (1.0 - x)).cos()
    }
    pub fn solar_coord_mapped(self, h: f32, mu: f32, mapping: CoordinateMapping) -> f32 {
        if mapping.is_reference() {
            return crate::reference_mapping::solar_coord(self, h, mu);
        }
        if mapping != CoordinateMapping::SunAlignedAngular {
            return self.solar_coord(h, mu);
        }
        let old = self.solar_coord(h, mu) * 64.0;
        if old <= 62.0 {
            return old / 96.0;
        }
        let theta = self.solar_cosine(h, 62.0 / 64.0).acos();
        let x = 1.0 - (mu.clamp(-1.0, 1.0).acos() / theta).clamp(0.0, 1.0).sqrt();
        (62.0 + x * 34.0) / 96.0
    }
    fn cone_horizon(self, h: f32, mu_s: f32, nu: f32) -> (f32, f32, f32) {
        let center = mu_s * nu;
        let extent = ((1.0 - mu_s * mu_s) * (1.0 - nu * nu)).max(0.0).sqrt();
        let angle = ((self.horizon(h) - center) / extent.max(1e-20))
            .clamp(-1.0, 1.0)
            .acos();
        (center, extent, angle)
    }
    pub fn cone_cosine(self, h: f32, mu_s: f32, nu: f32, ground: bool) -> f32 {
        let alpha = mu_s.clamp(-1.0, 1.0).acos();
        let beta = self.horizon(h).acos();
        let (lo, hi) = if ground {
            (
                (beta - alpha).max(0.0),
                (2.0 * std::f32::consts::PI - beta - alpha).min(std::f32::consts::PI),
            )
        } else {
            (
                (alpha - beta).max(0.0),
                (alpha + beta).min(std::f32::consts::PI),
            )
        };
        nu.clamp(-1.0, 1.0).acos().clamp(lo, hi).cos()
    }
    pub fn cone_view(self, h: f32, mu_s: f32, nu: f32, index: usize, size: usize) -> f32 {
        let ng = (size / 4).max(2);
        let (center, extent, horizon) = self.cone_horizon(h, mu_s, nu);
        let angle = if index < ng {
            let u = unit(index, ng);
            horizon + (std::f32::consts::PI - horizon) * u * u
        } else {
            let u = unit(index - ng, size - ng);
            horizon * (2.0 * u - u * u)
        };
        (center + extent * angle.cos()).clamp(-1.0, 1.0)
    }
    pub fn cone_coord(self, s: State, size: usize) -> f32 {
        let ng = (size / 4).max(2);
        let (center, extent, horizon) = self.cone_horizon(s.altitude_km, s.mu_s, s.nu);
        let angle = ((s.mu - center) / extent.max(1e-20))
            .clamp(-1.0, 1.0)
            .acos();
        if s.ground {
            ((angle - horizon) / (std::f32::consts::PI - horizon).max(1e-20))
                .clamp(0.0, 1.0)
                .sqrt()
                * (ng - 1) as f32
        } else {
            ng as f32
                + (1.0 - (1.0 - angle / horizon.max(1e-20)).clamp(0.0, 1.0).sqrt())
                    * (size - ng - 1) as f32
        }
    }
    pub fn advanced(self, s: State, d: f32) -> State {
        let radius = self.bottom + s.altitude_km;
        let rho2 = self.rho_squared(s.altitude_km) + d * (2.0 * radius * s.mu + d);
        let h = (rho2 / ((self.bottom * self.bottom + rho2).max(0.0).sqrt() + self.bottom))
            .clamp(0.0, self.top_height());
        State {
            altitude_km: h,
            mu: ((radius * s.mu + d) / (self.bottom + h)).clamp(-1.0, 1.0),
            mu_s: ((radius * s.mu_s + d * s.nu) / (self.bottom + h)).clamp(-1.0, 1.0),
            ..s
        }
    }
}

impl State {
    pub fn directions(self) -> (Vec3, Vec3) {
        let sin_v = (1.0 - self.mu * self.mu).max(0.0).sqrt();
        let view = Vec3::new(sin_v, 0.0, self.mu);
        let sx = if sin_v > 1e-6 {
            (self.nu - self.mu * self.mu_s) / sin_v
        } else {
            (1.0 - self.mu_s * self.mu_s).max(0.0).sqrt()
        }
        .clamp(
            -(1.0 - self.mu_s * self.mu_s).max(0.0).sqrt(),
            (1.0 - self.mu_s * self.mu_s).max(0.0).sqrt(),
        );
        (
            view,
            Vec3::new(
                sx,
                (1.0 - self.mu_s * self.mu_s - sx * sx).max(0.0).sqrt(),
                self.mu_s,
            )
            .normalize(),
        )
    }
}

pub fn unit(i: usize, size: usize) -> f32 {
    i as f32 / (size - 1) as f32
}
pub fn sun_cosine(x: f32) -> f32 {
    (((2.0 * x - 1.0) * 20.0_f32.asinh()).sinh() / 20.0).clamp(-1.0, 1.0)
}
pub fn sun_coord(mu: f32) -> f32 {
    (0.5 + 0.5 * (20.0 * mu).asinh() / 20.0_f32.asinh()).clamp(0.0, 1.0)
}

// Mixture CDF resolves the forward peak, anti-solar cap, and grazing sunlight
// around 90 degrees. A small nu error shifts solar tangent height on long rays.
pub fn scattering_cosine(u: f32) -> f32 {
    let mut lo = 0.0;
    let mut hi = std::f32::consts::PI;
    for _ in 0..22 {
        let mid = (lo + hi) * 0.5;
        if scattering_angle_coord(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ((lo + hi) * 0.5).cos()
}
pub fn scattering_coord(nu: f32) -> f32 {
    scattering_angle_coord(nu.clamp(-1.0, 1.0).acos())
}
fn scattering_angle_coord(theta: f32) -> f32 {
    let scale = 2.0_f32.to_radians();
    let pi = std::f32::consts::PI;
    (0.6 * (1.0 + theta / scale).ln() / 91.0_f32.ln()
        + 0.15 * (1.0 - (1.0 + (pi - theta) / scale).ln() / 91.0_f32.ln())
        + 0.25 * (((theta - pi * 0.5) / scale).atan() + 45.0_f32.atan()) / (2.0 * 45.0_f32.atan()))
    .clamp(0.0, 1.0)
}

pub fn sample<const N: usize>(values: &[f32], dims: [usize; N], coords: [f32; N]) -> f32 {
    let mut lo = [0; N];
    let mut hi = [0; N];
    let mut t = [0.0; N];
    for a in 0..N {
        let x = coords[a].clamp(0.0, (dims[a] - 1) as f32);
        lo[a] = x.floor() as usize;
        hi[a] = (lo[a] + 1).min(dims[a] - 1);
        t[a] = x - lo[a] as f32;
    }
    let mut result = 0.0;
    for corner in 0..(1 << N) {
        let mut index = 0;
        let mut weight = 1.0;
        for a in 0..N {
            let upper = corner & (1 << a) != 0;
            index = index * dims[a] + if upper { hi[a] } else { lo[a] };
            weight *= if upper { t[a] } else { 1.0 - t[a] };
        }
        if weight > 0.0 {
            result += weight * values[index];
        }
    }
    result
}

/// Interpolation of positive solar attenuation in log amplitude. Storage stays
/// linear f32 radiance; the floor only regularizes exactly shadowed corners.
pub fn sample_radiance(
    values: &[f32],
    dims: [usize; 4],
    coords: [f32; 4],
    mode: SolarInterpolation,
) -> f32 {
    if mode == SolarInterpolation::Linear {
        return sample(values, dims, coords);
    }
    let s = coords[2].clamp(0.0, (dims[2] - 1) as f32);
    let lo = s.floor();
    let hi = (lo + 1.0).min((dims[2] - 1) as f32);
    let mut c = coords;
    c[2] = lo;
    let a = sample(values, dims, c);
    c[2] = hi;
    let b = sample(values, dims, c);
    log_mix(a, b, s - lo)
}

/// Reproject the physical view into each height/solar/phase corner's cone.
/// A normalized cone coordinate is not shared between those corners: its
/// horizon-clipped interval can change topology within one interpolation cell.
pub fn sample_radiance_state(
    values: &[f32],
    geometry: Geometry,
    config: &BakeConfig,
    state: State,
    scattering_cosines: &[f32],
) -> f32 {
    if config.mapping.is_reference() {
        return crate::reference_mapping::sample_with(geometry, config, state, |i| values[i]);
    }
    let c = geometry.coords_config(state, config);
    let dims = config.scattering;
    if config.mapping == CoordinateMapping::Legacy {
        return sample_radiance(values, dims, c, config.solar_interpolation);
    }
    let lo =
        std::array::from_fn::<_, 4, _>(|i| c[i].clamp(0.0, (dims[i] - 1) as f32).floor() as usize);
    let t = std::array::from_fn::<_, 4, _>(|i| (c[i] - lo[i] as f32).clamp(0.0, 1.0));
    let mut solar = [0.0; 2];
    for (upper_s, value) in solar.iter_mut().enumerate() {
        let si = (lo[2] + upper_s).min(dims[2] - 1);
        for upper_r in 0..2 {
            let ri = (lo[0] + upper_r).min(dims[0] - 1);
            let h = geometry.height(unit(ri, dims[0]));
            let mu_s = geometry.solar_cosine_mapped(h, unit(si, dims[2]), config.mapping);
            for upper_n in 0..2 {
                let ni = (lo[3] + upper_n).min(dims[3] - 1);
                let nu = geometry.cone_cosine(h, mu_s, scattering_cosines[ni], state.ground);
                let corner = State {
                    altitude_km: h,
                    mu_s,
                    nu,
                    ..state
                };
                let mc = match config.cone_mapping {
                    ConeMapping::OpticalDistance => geometry.optical_cone_coord(corner, dims[1]),
                    ConeMapping::AngleSquare => geometry.cone_coord(corner, dims[1]),
                };
                let mi = mc.floor() as usize;
                let mh = (mi + 1).min(dims[1] - 1);
                let mt = mc - mi as f32;
                let index = |m| ((ri * dims[1] + m) * dims[2] + si) * dims[3] + ni;
                let radiance = values[index(mi)] * (1.0 - mt) + values[index(mh)] * mt;
                let rw = if upper_r == 0 { 1.0 - t[0] } else { t[0] };
                let nw = if upper_n == 0 { 1.0 - t[3] } else { t[3] };
                *value += radiance * rw * nw;
            }
        }
    }
    match config.solar_interpolation {
        SolarInterpolation::Linear => solar[0] * (1.0 - t[2]) + solar[1] * t[2],
        SolarInterpolation::LogRadiance => log_mix(solar[0], solar[1], t[2]),
    }
}

pub fn log_mix(a: f32, b: f32, t: f32) -> f32 {
    // Signed RGB components can occur outside the chosen output gamut.
    if a < 0.0 || b < 0.0 {
        return a * (1.0 - t) + b * t;
    }
    if a == b || t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let floor = 1e-30;
    (((a + floor).ln() * (1.0 - t) + (b + floor).ln() * t).exp() - floor).max(0.0)
}
