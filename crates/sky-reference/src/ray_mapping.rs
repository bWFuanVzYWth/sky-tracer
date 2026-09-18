//! Ray-frame interpolation for the reference grid. Older assets keep v5 lookup.
use crate::{
    config::{BakeConfig, PhaseInterpolation, SolarInterpolation, ViewInterpolation},
    mapping::{Geometry, State, log_mix, unit},
    reference_mapping as m,
};

#[derive(Clone, Copy, Default)]
struct Column {
    lower: usize,
    upper: usize,
    before: usize,
    after: usize,
    cubic_view: bool,
    phase: [usize; 4],
    view_t: f32,
    phase_t: f32,
    log: bool,
}
impl Column {
    fn new(g: Geometry, c: &BakeConfig, s: State, ri: usize, si: usize) -> Self {
        let [_, nm, ns, nn] = c.scattering;
        let nc = m::phase_coord(g, s) * (nn - 1) as f32;
        let ni = nc.floor() as usize;
        let mc = g.optical_cone_coord(s, nm);
        let mi = mc.floor() as usize;
        let ng = (nm / 4).max(2);
        let (begin, end) = if s.ground { (0, ng - 1) } else { (ng, nm - 1) };
        Self {
            lower: ((ri * nm + mi) * ns + si) * nn,
            upper: ((ri * nm + (mi + 1).min(nm - 1)) * ns + si) * nn,
            before: ((ri * nm + mi.saturating_sub(1).max(begin)) * ns + si) * nn,
            after: ((ri * nm + (mi + 2).min(end)) * ns + si) * nn,
            cubic_view: c.view_interpolation == ViewInterpolation::MonotoneCubic,
            phase: std::array::from_fn(|k| (ni + k).saturating_sub(1).min(nn - 1)),
            view_t: mc - mi as f32,
            phase_t: nc - ni as f32,
            log: c.phase_interpolation == PhaseInterpolation::LogRadiance,
        }
    }
    fn sample(&self, fetch: &mut impl FnMut(usize) -> f32) -> f32 {
        let values = self.phase.map(|n| {
            let a = fetch(self.lower + n);
            if self.view_t <= 0.0 {
                a
            } else {
                let b = fetch(self.upper + n);
                if self.cubic_view {
                    let before = if self.before == self.lower {
                        2.0 * a - b
                    } else {
                        fetch(self.before + n)
                    };
                    let after = if self.after == self.upper {
                        2.0 * b - a
                    } else {
                        fetch(self.after + n)
                    };
                    m::cubic_phase([before, a, b, after], self.view_t)
                } else {
                    a * (1.0 - self.view_t) + b * self.view_t
                }
            }
        });
        if self.log {
            m::log_cubic_phase(values, self.phase_t)
        } else {
            m::cubic_phase(values, self.phase_t)
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Row {
    charts: [[Column; 2]; 2],
    solar_t: f32,
    active: bool,
}
#[derive(Clone)]
pub struct RayStencil {
    rows: [Row; 2],
    radius_t: f32,
    chart_blend: f32,
    log: bool,
    linear_radius: bool,
}
impl RayStencil {
    pub fn with_linear_radius(mut self) -> Self {
        self.linear_radius = true;
        self
    }
    pub fn new(g: Geometry, c: &BakeConfig, s: State, heights: Option<&[f32]>) -> Self {
        Self::build(g, c, s, heights, false)
    }
    /// CPU source-field diagnostic: hold the local direction angles across height.
    /// Unlike boundary radiance, a volume source has no outgoing horizon discontinuity.
    pub fn new_local_source(
        g: Geometry,
        c: &BakeConfig,
        s: State,
        heights: Option<&[f32]>,
    ) -> Self {
        Self::build(g, c, s, heights, true)
    }
    fn build(
        g: Geometry,
        c: &BakeConfig,
        s: State,
        heights: Option<&[f32]>,
        local_source: bool,
    ) -> Self {
        let [nr, _, ns, _] = c.scattering;
        let rc = if local_source {
            m::radius_coord_linear(g, c, s.altitude_km)
        } else {
            m::radius_coord(g, c, s.altitude_km)
        };
        let rlo = rc.floor() as usize;
        let rt = rc - rlo as f32;
        let hor = g.horizon(s.altitude_km);
        let delta = (s.mu - hor) / if s.ground { 1.0 + hor } else { 1.0 - hor };
        let d = (s.mu_s.clamp(-1.0, 1.0).asin() - hor.asin())
            .abs()
            .to_degrees();
        let x = ((d - 1.0) / 2.0).clamp(0.0, 1.0);
        let blend = x * x * (3.0 - 2.0 * x);
        let mut rows = [Row::default(); 2];
        for (r, row) in rows.iter_mut().enumerate() {
            if (r == 0 && rt >= 1.0) || (r == 1 && rt <= 0.0) {
                continue;
            }
            let ri = (rlo + r).min(nr - 1);
            let h = heights.map_or_else(|| m::radius(g, c, ri), |hs| hs[ri]);
            let hor = g.horizon(h);
            let mu = (hor + delta * if s.ground { 1.0 + hor } else { 1.0 - hor }).clamp(-1.0, 1.0);
            let mus = if 1.0 - s.mu * s.mu > 1e-7 {
                (mu * s.nu
                    + ((1.0 - mu * mu) / (1.0 - s.mu * s.mu)).max(0.0).sqrt()
                        * (s.mu_s - s.mu * s.nu))
                    .clamp(-1.0, 1.0)
            } else {
                s.mu_s
            };
            let mut point = State {
                altitude_km: h,
                mu,
                mu_s: mus,
                ..s
            };
            if local_source {
                point = State {
                    altitude_km: h,
                    ground: g.hits_ground(h, s.mu),
                    ..s
                };
            }
            let (mu, mus) = (point.mu, point.mu_s);
            let az = ((point.nu - mu * mus)
                / ((1.0 - mu * mu) * (1.0 - mus * mus)).max(1e-30).sqrt())
            .clamp(-1.0, 1.0);
            let sc = m::solar_coord(g, h, mus) * (ns - 1) as f32;
            let slo = sc.floor() as usize;
            row.active = true;
            row.solar_t = sc - slo as f32;
            for (chart, columns) in row.charts.iter_mut().enumerate() {
                if (chart == 0 && blend >= 1.0) || (chart == 1 && blend <= 0.0) {
                    continue;
                }
                for (j, column) in columns.iter_mut().enumerate() {
                    let si = (slo + j).min(ns - 1);
                    let corner = if chart == 0 {
                        let mus = m::solar_cosine(g, h, unit(si, ns));
                        State {
                            mu_s: mus,
                            nu: mu * mus
                                + ((1.0 - mu * mu) * (1.0 - mus * mus)).max(0.0).sqrt() * az,
                            ..point
                        }
                    } else {
                        point
                    };
                    *column = Column::new(g, c, corner, ri, si);
                }
            }
        }
        Self {
            rows,
            radius_t: rt,
            chart_blend: blend,
            log: c.solar_interpolation == SolarInterpolation::LogRadiance,
            linear_radius: false,
        }
    }
    pub fn sample_with(&self, mut fetch: impl FnMut(usize) -> f32) -> f32 {
        let mut radial = [0.0; 2];
        for (r, row) in self.rows.iter().enumerate() {
            if !row.active {
                continue;
            }
            for chart in 0..2 {
                let weight = if chart == 0 {
                    1.0 - self.chart_blend
                } else {
                    self.chart_blend
                };
                if weight <= 0.0 {
                    continue;
                }
                let a = row.charts[chart][0].sample(&mut fetch);
                let b = row.charts[chart][1].sample(&mut fetch);
                radial[r] += weight
                    * if self.log {
                        log_mix(a, b, row.solar_t)
                    } else {
                        a * (1.0 - row.solar_t) + b * row.solar_t
                    };
            }
        }
        if self.log && !self.linear_radius && radial[0] > 0.0 && radial[1] > 0.0 {
            log_mix(radial[0], radial[1], self.radius_t)
        } else {
            radial[0] * (1.0 - self.radius_t) + radial[1] * self.radius_t
        }
    }
}
