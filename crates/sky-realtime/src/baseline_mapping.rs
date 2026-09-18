use crate::geometry::Geometry;
use std::f32::consts::PI;
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
