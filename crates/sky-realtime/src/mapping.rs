//! Offline-fitted coordinates. Inversion happens only on the CPU at setup or
//! observer changes; lookup uses a few arithmetic operations and cached terms.
use crate::geometry::Geometry;
use std::{f32::consts::PI, sync::OnceLock};

struct Calibration {
    values: serde_json::Value,
    source: String,
}
static CALIBRATION: OnceLock<Calibration> = OnceLock::new();

fn calibration() -> &'static Calibration {
    CALIBRATION.get_or_init(|| Calibration {
        values: serde_json::from_str(CALIBRATION_JSON).expect("embedded mapping calibration"),
        source: CALIBRATION_JSON.to_owned(),
    })
}

/// Freeze an experiment's calibration before creating any renderer or mapping.
/// Once initialized it cannot change, including while a GPU solution is cached.
/// The application reads files; the core only validates the supplied JSON.
pub fn load_calibration_json(source: &str) -> crate::Result<()> {
    let values: serde_json::Value = serde_json::from_str(source)?;
    validate_calibration(&values)?;
    CALIBRATION
        .set(Calibration {
            values,
            source: source.to_owned(),
        })
        .map_err(|_| "mapping calibration is already initialized; use a fresh process".into())
}

/// Exact JSON used by both CPU node generation and GPU lookup coefficients.
pub fn calibration_json() -> &'static str {
    &calibration().source
}

fn validate_calibration(v: &serde_json::Value) -> crate::Result<()> {
    let array = |value: &serde_json::Value, length: usize, name: &str| -> crate::Result<Vec<f32>> {
        let a = value
            .as_array()
            .ok_or_else(|| format!("mapping {name} must be an array"))?;
        if a.len() != length {
            return Err(format!("mapping {name} needs {length} values").into());
        }
        a.iter()
            .map(|x| {
                let f = x
                    .as_f64()
                    .ok_or_else(|| format!("mapping {name} must contain numbers"))?
                    as f32;
                if !f.is_finite() {
                    return Err(format!("mapping {name} must contain finite f32 values").into());
                }
                Ok(f)
            })
            .collect()
    };
    let heights_len = v["source_height"].as_array().map_or(0, Vec::len);
    if !(8..=128).contains(&heights_len) {
        return Err("mapping source_height needs 8..=128 nodes".into());
    }
    let heights = array(&v["source_height"], heights_len, "source_height")?;
    if heights[0] != 0.0
        || *heights.last().unwrap() != 120.0
        || !heights.windows(2).all(|h| h[1] > h[0])
    {
        return Err("mapping heights must increase strictly from 0 to 120 km".into());
    }
    let weights = |value: &serde_json::Value, count: usize, name: &str| -> crate::Result<()> {
        let w = array(value, count, name)?;
        if w.iter().any(|x| *x < 0.0) || w[0] < 0.001 || (w.iter().sum::<f32>() - 1.0).abs() > 1e-5
        {
            return Err(format!(
                "mapping {name} needs nonnegative unit-sum weights and a positive uniform component"
            )
            .into());
        }
        Ok(())
    };
    let widths = |value: &serde_json::Value, name: &str| -> crate::Result<()> {
        let w = array(value, 3, name)?;
        if w.iter().any(|x| !(1e-6..=PI).contains(x)) {
            return Err(format!("mapping {name} widths must be in [1e-6, pi] radians").into());
        }
        Ok(())
    };
    weights(&v["solar_weights"], 5, "solar_weights")?;
    widths(&v["solar_widths"], "solar_widths")?;
    for name in ["low", "upper", "space", "ground", "ground_near"] {
        weights(&v["sky"][name]["weights"], 4, name)?;
        widths(&v["sky"][name]["widths"], name)?;
    }
    for (name, lo, hi) in [("phase_weight", 0.0, 1.0), ("cone_warp", -1.0, 0.99)] {
        let value = v[name]
            .as_f64()
            .ok_or_else(|| format!("missing numeric mapping {name}"))? as f32;
        if !value.is_finite() || !(lo..=hi).contains(&value) {
            return Err(format!("mapping {name} is outside [{lo}, {hi}]").into());
        }
    }
    let anchors = array(&v["optical"]["anchors_km"], 7, "optical.anchors_km")?;
    let indices = array(&v["optical"]["indices"], 7, "optical.indices")?;
    if anchors != [0.0, 1.0, 2.0, 11.0, 12.0, 35.0, 120.0]
        || indices[0] != 0.0
        || indices[6] != 255.0
        || !indices.windows(2).all(|x| x[1] > x[0])
    {
        return Err(
            "mapping optical anchors must retain material boundaries and increasing indices 0..255"
                .into(),
        );
    }
    Ok(())
}

fn fit() -> &'static serde_json::Value {
    &calibration().values
}
fn numbers(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
pub fn inverse(f: impl Fn(f32) -> f32, u: f32, mut lo: f32, mut hi: f32) -> f32 {
    if u <= 0.0 {
        return lo;
    }
    if u >= 1.0 {
        return hi;
    }
    for _ in 0..28 {
        let mid = (lo + hi) * 0.5;
        if f(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}
pub fn heights(g: Geometry, n: u32) -> Vec<f32> {
    let nodes = numbers(&fit()["source_height"]);
    let mut result: Vec<_> = (0..n)
        .map(|i| {
            let x = i as f32 / (n - 1) as f32 * (nodes.len() - 1) as f32;
            let j = (x as usize).min(nodes.len() - 2);
            let t = x - j as f32;
            let h = nodes[j] * (1.0 - t) + nodes[j + 1] * t;
            if h <= 35.0 {
                h
            } else {
                35.0 + (h - 35.0) * (g.top_height() - 35.0) / 85.0
            }
        })
        .collect();
    // Preserve the material break and exact aerosol/Rayleigh boundary at every
    // supported resolution. Smaller diagnostic grids only pin the latter.
    let anchors: &[f32] = if n >= 32 {
        &[1.0, 2.0, 11.0, 12.0, 35.0]
    } else {
        &[35.0]
    };
    let mut used = vec![false; n as usize];
    used[0] = true;
    used[n as usize - 1] = true;
    for &anchor in anchors.iter().rev() {
        let j = (1..n as usize - 1)
            .filter(|&j| !used[j])
            .min_by(|&a, &b| {
                (result[a] - anchor)
                    .abs()
                    .total_cmp(&(result[b] - anchor).abs())
            })
            .unwrap();
        result[j] = anchor;
        used[j] = true;
    }
    result.sort_by(f32::total_cmp);
    result
}
fn soft(x: f32, w: f32) -> f32 {
    x / (x.abs() + w)
}
pub fn solar_coefficients(g: Geometry, h: f32) -> [[f32; 4]; 4] {
    let weights = numbers(&fit()["solar_weights"]);
    let widths = numbers(&fit()["solar_widths"]);
    let hor = g.horizon(h).asin();
    let centers = [hor, hor - 6.0_f32.to_radians(), -hor];
    let mut c = [[0.0; 4]; 4];
    c[0] = [
        weights[0] / PI,
        weights[0] * 0.5 + weights[4],
        weights[4] / (PI / (PI + 5.0_f32.to_radians())).sqrt(),
        0.0,
    ];
    for j in 0..3 {
        let a = soft(-PI * 0.5 - centers[j], widths[j]);
        let b = soft(PI * 0.5 - centers[j], widths[j]);
        let amplitude = weights[j + 1] / (b - a);
        c[0][1] -= amplitude * a;
        c[j + 1] = [centers[j], widths[j], amplitude, 0.0];
    }
    c
}
pub fn solar_coord(e: f32, c: &[[f32; 4]; 4]) -> f32 {
    let d = (PI * 0.5 - e).max(0.0);
    let mut y = c[0][0] * e + c[0][1] - c[0][2] * (d / (d + 5.0_f32.to_radians())).sqrt();
    for v in &c[1..] {
        y += v[2] * soft(e - v[0], v[1]);
    }
    y.clamp(0.0, 1.0)
}
pub fn phase_coord(nu: f32, fitted: bool) -> f32 {
    let a = if fitted {
        fit()["phase_weight"].as_f64().unwrap() as f32
    } else {
        0.5
    };
    a * nu.clamp(-1.0, 1.0).acos() / PI + (1.0 - a) * ((1.0 - nu) * 0.5).max(0.0).cbrt()
}
pub fn angular_parameters() -> [f32; 4] {
    [
        fit()["phase_weight"].as_f64().unwrap() as f32,
        fit()["cone_warp"].as_f64().unwrap() as f32,
        0.0,
        0.0,
    ]
}
pub fn cone_coord(c: f32) -> f32 {
    c + angular_parameters()[1] * c * (1.0 - c) * (2.0 * c - 1.0)
}

/// Segment coefficients map sqrt(height) to normalized texture coordinate.
/// Anchor indices are rounded before inversion, keeping layer boundaries on
/// texel centers even when diagnostics change the resolution.
pub fn optical_segments(top: f32, n: u32) -> [[f32; 4]; 6] {
    let mut heights = numbers(&fit()["optical"]["anchors_km"]);
    heights[6] = top;
    let indices = numbers(&fit()["optical"]["indices"]);
    std::array::from_fn(|j| {
        let a = (indices[j] / 255.0 * (n - 1) as f32).round() / (n - 1) as f32;
        let b = (indices[j + 1] / 255.0 * (n - 1) as f32).round() / (n - 1) as f32;
        let slope = (b - a) / (heights[j + 1].sqrt() - heights[j].sqrt());
        [
            heights[j + 1].sqrt(),
            slope,
            a - slope * heights[j].sqrt(),
            b,
        ]
    })
}
pub fn optical_height(u: f32, segs: &[[f32; 4]; 6]) -> f32 {
    let s = segs.iter().find(|s| u <= s[3]).unwrap_or(&segs[5]);
    ((u - s[2]) / s[1].max(1e-20)).powi(2)
}

pub struct SkyChart {
    pub bounds: [f32; 2],
    pub centers: [f32; 3],
    pub weights: [f32; 4],
    pub widths: [f32; 3],
    pub fitted: bool,
}
impl SkyChart {
    pub fn new(g: Geometry, h: f32, sun: f32, chart: usize, fitted: bool) -> Self {
        let hor = g.horizon(h).asin();
        let space = h >= g.top_height();
        let outer = if space {
            -((g.bottom + g.top_height()) / (g.bottom + h))
                .clamp(0.0, 1.0)
                .acos()
        } else {
            hor
        };
        let bounds = match chart {
            0 => [hor, if space { outer } else { 0.0 }],
            1 => [0.0, PI * 0.5],
            _ => [-PI * 0.5, hor],
        };
        let group = if chart == 2 {
            "ground"
        } else if space {
            "space"
        } else if chart == 1 {
            "upper"
        } else {
            "low"
        };
        let mut weights: [f32; 4] = if fitted {
            numbers(&fit()["sky"][group]["weights"]).try_into().unwrap()
        } else {
            [0.15, 0.4, 0.35, 0.1]
        };
        let mut widths: [f32; 3] = if fitted {
            numbers(&fit()["sky"][group]["widths"]).try_into().unwrap()
        } else {
            [PI / 180.0, 0.7 * PI / 180.0, 0.5 * PI / 180.0]
        };
        if fitted && chart == 2 && !space {
            // Near-ground grazing rays have a sharp distance-to-surface cusp.
            // Fit this chart separately, then blend across altitude on the CPU.
            let smooth = |x: f32| {
                let x = x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            };
            let t = 1.0 - smooth((h - 1.0) / 3.0);
            let w = numbers(&fit()["sky"]["ground_near"]["weights"]);
            let widths_low = numbers(&fit()["sky"]["ground_near"]["widths"]);
            for j in 0..4 {
                weights[j] = weights[j] * (1.0 - t) + w[j] * t;
            }
            for j in 0..3 {
                widths[j] = widths[j] * (1.0 - t) + widths_low[j] * t;
            }
        }
        Self {
            bounds,
            // Legacy WGSL used a strict > test for the outer CDF center but
            // >= for chart bounds. Keep its CPU inverse consistent at top.
            centers: [
                hor,
                sun,
                if !fitted && h == g.top_height() {
                    hor
                } else {
                    outer
                },
            ],
            weights,
            widths,
            fitted,
        }
    }
    pub fn forward(&self, e: f32) -> f32 {
        let [lo, hi] = self.bounds;
        let e = e.clamp(lo, hi);
        let mut y = self.weights[0] * (e - lo) / (hi - lo).max(1e-10);
        for j in 0..3 {
            if self.fitted && (self.centers[j] < lo || self.centers[j] > hi) {
                // Algebraic softsign difference: no subtraction of nearly
                // identical saturated values for a Sun far outside a thin arc.
                y += self.weights[j + 1] * (e - lo) / (hi - lo).max(1e-10)
                    * ((hi - self.centers[j]).abs() + self.widths[j])
                    / ((e - self.centers[j]).abs() + self.widths[j]);
                continue;
            }
            let f = |d| {
                if self.fitted {
                    soft(d, self.widths[j])
                } else {
                    (d / self.widths[j]).atan()
                }
            };
            let a = f(lo - self.centers[j]);
            let b = f(hi - self.centers[j]);
            y += self.weights[j + 1] * (f(e - self.centers[j]) - a) / (b - a).max(1e-10);
        }
        y.clamp(0.0, 1.0)
    }
    pub fn coefficients(&self) -> [[f32; 4]; 4] {
        let [lo, hi] = self.bounds;
        let slope = self.weights[0] / (hi - lo).max(1e-10);
        let mut c = [[0.0; 4]; 4];
        c[0] = [slope, 0.0, lo, hi];
        for j in 0..3 {
            let a = soft(lo - self.centers[j], self.widths[j]);
            let b = soft(hi - self.centers[j], self.widths[j]);
            let outside = self.centers[j] < lo || self.centers[j] > hi;
            let scale = if outside {
                self.weights[j + 1] * ((hi - self.centers[j]).abs() + self.widths[j])
                    / (hi - lo).max(1e-10)
            } else {
                self.weights[j + 1] / (b - a).max(1e-10)
            };
            c[j + 1] = [self.centers[j], self.widths[j], scale, a];
        }
        c
    }
}
pub fn sky_cache(
    g: Geometry,
    h: f32,
    sun: f32,
    size: u32,
    fitted: bool,
) -> (Vec<[f32; 4]>, [u32; 4]) {
    sky_cache_for_view(g, h, sun, size, fitted, true)
}

/// The ground rows may be omitted only when projection cannot query them.
/// Sky texel coordinates and both sky charts retain their original layout.
pub fn sky_cache_for_view(
    g: Geometry,
    h: f32,
    sun: f32,
    size: u32,
    fitted: bool,
    include_ground: bool,
) -> (Vec<[f32; 4]>, [u32; 4]) {
    let sky_rows = size * 3 / 4;
    let lower = if fitted && h >= g.top_height() {
        sky_rows
    } else {
        sky_rows / 2
    };
    let counts = [lower, sky_rows - lower, size - sky_rows];
    let mut coefficients = Vec::new();
    let mut rows = Vec::new();
    for chart in 0..3 {
        if chart == 2 && !include_ground {
            coefficients.extend([[0.0; 4]; 4]);
            continue;
        }
        let c = SkyChart::new(g, h, sun, chart, fitted);
        coefficients.extend(c.coefficients());
        for y in 0..counts[chart] {
            let u = y as f32 / (counts[chart] - 1) as f32;
            let e = inverse(|e| c.forward(e), u, c.bounds[0], c.bounds[1]);
            rows.push([e.sin(), e, 0.0, 0.0]);
        }
    }
    coefficients.extend(rows);
    (coefficients, [size, size, lower, sky_rows])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_rejects_nonmonotone_and_singular_coordinates() {
        let mut v: serde_json::Value = serde_json::from_str(CALIBRATION_JSON).unwrap();
        validate_calibration(&v).unwrap();
        v["sky"]["space"]["widths"][0] = 0.0.into();
        assert!(validate_calibration(&v).is_err());
        v = serde_json::from_str(CALIBRATION_JSON).unwrap();
        v["source_height"][3] = v["source_height"][2].clone();
        assert!(validate_calibration(&v).is_err());
        v = serde_json::from_str(CALIBRATION_JSON).unwrap();
        v["sky"]["low"]["weights"][0] = (-0.1).into();
        assert!(validate_calibration(&v).is_err());
    }
    #[test]
    fn omitted_ground_rows_preserve_all_sky_coordinates() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        for h in [0.0, 0.2, 12.0, 108.0, 120.0, 400.0] {
            for sun in [-16.0f32, -6.0, 0.0, 85.0] {
                let (full, layout) = sky_cache(g, h, sun.to_radians(), 256, true);
                let (sky, sky_layout) =
                    sky_cache_for_view(g, h, sun.to_radians(), 256, true, false);
                assert_eq!(layout, sky_layout);
                assert_eq!(sky.len(), 12 + layout[3] as usize);
                assert_eq!(&sky[..8], &full[..8]);
                assert_eq!(&sky[12..], &full[12..sky.len()]);
            }
        }
    }
    #[test]
    fn fitted_nodes_and_inverses_cover_domain() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        for n in [4, 32, 48, 72, 96, 128] {
            let hs = heights(g, n);
            assert_eq!(hs[0], 0.0);
            assert_eq!(*hs.last().unwrap(), 120.0);
            assert!(hs.contains(&35.0));
            assert!(hs.windows(2).all(|w| w[1] > w[0]));
            for h in hs {
                let c = solar_coefficients(g, h);
                for j in 0..176 {
                    let u = j as f32 / 175.0;
                    let e = inverse(|e| solar_coord(e, &c), u, -PI * 0.5, PI * 0.5);
                    assert!((solar_coord(e, &c) - u).abs() < 2e-5);
                }
            }
        }
        for n in [128, 256, 512] {
            let segs = optical_segments(120.0, n);
            for i in 0..n {
                let u = i as f32 / (n - 1) as f32;
                let h = optical_height(u, &segs);
                let s = segs.iter().find(|s| h.sqrt() <= s[0]).unwrap_or(&segs[5]);
                assert!((h.sqrt() * s[1] + s[2] - u).abs() < 2e-6);
            }
        }
        for h in [0.0, 0.2, 35.0, 108.0, 119.99, 120.0, 400.0, 36000.0] {
            let (rows, layout) = sky_cache(g, h, 0.82, 256, true);
            assert_eq!(rows.len(), 268);
            assert_eq!(layout[3], 192);
            assert!(rows.iter().flatten().all(|x| x.is_finite()));
            for k in 0..3 {
                let c = SkyChart::new(g, h, 0.82, k, true);
                let packed = c.coefficients();
                if c.bounds[1] - c.bounds[0] < 1e-6 {
                    continue;
                }
                for i in 0..193 {
                    let u = i as f32 / 192.0;
                    let e = inverse(|e| c.forward(e), u, c.bounds[0], c.bounds[1]);
                    let mut y = packed[0][0] * (e - c.bounds[0]);
                    for t in &packed[1..] {
                        y += if t[0] < c.bounds[0] || t[0] > c.bounds[1] {
                            (e - c.bounds[0]) * t[2] / ((e - t[0]).abs() + t[1])
                        } else {
                            t[2] * (soft(e - t[0], t[1]) - t[3])
                        };
                    }
                    assert!((y - u).abs() < 1e-3, "h={h}, chart={k}, u={u}, y={y}");
                }
            }
        }
    }
}

pub const CALIBRATION_JSON: &str = include_str!("../configs/mapping_fit.json");
