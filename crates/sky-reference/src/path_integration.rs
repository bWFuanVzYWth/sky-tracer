//! CPU mirror of the optional, frozen realtime height allocation for ray cells.
use crate::{
    config::RayStepMapping,
    mapping::{Geometry, State},
};

const HEIGHT_SCALE_KM: f32 = 0.25;

fn log1p(x: f32) -> f32 {
    if x < 0.001 {
        x * (1.0 - x * 0.5 + x * x / 3.0)
    } else {
        (1.0 + x).ln()
    }
}
fn expm1(x: f32) -> f32 {
    if x < 0.001 {
        x * (1.0 + x * 0.5 + x * x / 6.0)
    } else {
        x.exp() - 1.0
    }
}

struct PathPlan {
    geometry: Geometry,
    state: State,
    count: usize,
    length: f32,
    middle: f32,
    hmin: f32,
    before: f32,
    after: f32,
    split: usize,
}
impl PathPlan {
    fn new(g: Geometry, s: State, count: usize, mapping: RayStepMapping) -> Self {
        assert!(count > 0, "path integration requires positive steps");
        let length = g.distance(s.altitude_km, s.mu, s.ground);
        let mut plan = Self {
            geometry: g,
            state: s,
            count,
            length,
            middle: 0.0,
            hmin: 0.0,
            before: 0.0,
            after: 0.0,
            split: 0,
        };
        if mapping == RayStepMapping::UniformDistance || length <= 0.0 || count < 2 {
            return plan;
        }
        plan.middle = (-(g.bottom + s.altitude_km) * s.mu).clamp(0.0, length);
        plan.hmin = g.advanced(s, plan.middle).altitude_km;
        let hend = g.advanced(s, length).altitude_km;
        plan.before = log1p((s.altitude_km - plan.hmin).max(0.0) / HEIGHT_SCALE_KM);
        plan.after = log1p((hend - plan.hmin).max(0.0) / HEIGHT_SCALE_KM);
        if plan.before + plan.after < 0.00001 {
            plan.before = 0.0;
            plan.after = 0.0;
            return plan;
        }
        plan.split = (count as f32 * plan.before / (plan.before + plan.after)).round() as usize;
        plan.split = if plan.middle <= 0.0 {
            0
        } else if plan.middle >= length {
            count
        } else {
            plan.split.clamp(1, count - 1)
        };
        let a = HEIGHT_SCALE_KM * expm1(plan.before / plan.split.max(1) as f32);
        let b = HEIGHT_SCALE_KM * expm1(plan.after / (count - plan.split).max(1) as f32);
        if (plan.before > 0.0 && plan.hmin + a == plan.hmin)
            || (plan.after > 0.0 && plan.hmin + b == plan.hmin)
        {
            plan.before = 0.0;
            plan.after = 0.0;
        }
        plan
    }
    fn is_log(&self) -> bool {
        self.before + self.after > 0.0
    }
    fn edge(&self, i: usize, previous: f32) -> f32 {
        if i >= self.count {
            return self.length;
        }
        if !self.is_log() {
            return self.length * i as f32 / self.count as f32;
        }
        if i == self.split {
            return self.middle;
        }
        let incoming = i < self.split;
        let lh = if incoming {
            self.before * (1.0 - i as f32 / self.split as f32)
        } else {
            self.after * (i - self.split) as f32 / (self.count - self.split) as f32
        };
        let h = self.hmin + HEIGHT_SCALE_KM * expm1(lh);
        let b = (self.geometry.bottom + self.state.altitude_km) * self.state.mu;
        let c = (self.state.altitude_km - h)
            * (2.0 * self.geometry.bottom + self.state.altitude_km + h);
        let root = (b * b - c).max(0.0).sqrt();
        let edge = if incoming {
            c / (-b + root).max(1e-20)
        } else if b > 0.0 {
            -c / (root + b).max(1e-20)
        } else {
            -b + root
        };
        edge.clamp(previous, self.length)
    }
}

/// Full boundary-to-boundary ray partition. Zero-length rays have zero edges.
/// Counts and geometry must be valid, as enforced by the bake configuration.
pub fn path_edges(g: Geometry, s: State, steps: usize, mapping: RayStepMapping) -> Vec<f32> {
    let plan = PathPlan::new(g, s, steps, mapping);
    let mut edges = Vec::with_capacity(steps + 1);
    edges.push(0.0);
    for i in 1..=steps {
        edges.push(plan.edge(i, edges[i - 1]));
    }
    edges
}

/// (midpoint distance, cell width), retaining the old uniform expressions.
pub(crate) fn path_cells(
    g: Geometry,
    s: State,
    steps: usize,
    mapping: RayStepMapping,
) -> Vec<(f32, f32)> {
    let plan = PathPlan::new(g, s, steps, mapping);
    if !plan.is_log() {
        let dx = plan.length / steps as f32;
        return (0..steps).map(|j| ((j as f32 + 0.5) * dx, dx)).collect();
    }
    let mut previous = 0.0;
    (1..=steps)
        .map(|i| {
            let edge = plan.edge(i, previous);
            let cell = ((edge + previous) * 0.5, edge - previous);
            previous = edge;
            cell
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_cells_preserve_original_arithmetic() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        let s = State {
            altitude_km: 0.2,
            mu: 0.01,
            mu_s: 0.1,
            nu: 0.5,
            ground: false,
        };
        for steps in [1, 64, 192, 256, 768] {
            let dx = g.distance(s.altitude_km, s.mu, s.ground) / steps as f32;
            for (j, (travel, width)) in path_cells(g, s, steps, RayStepMapping::UniformDistance)
                .into_iter()
                .enumerate()
            {
                assert_eq!(width.to_bits(), dx.to_bits());
                assert_eq!(travel.to_bits(), ((j as f32 + 0.5) * dx).to_bits());
            }
        }
    }
}
