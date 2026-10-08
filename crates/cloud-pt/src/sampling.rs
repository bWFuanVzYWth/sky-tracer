//! Deterministic CPU sampling. Every film sample has its own random stream.

use glam::DVec3;
use std::f64::consts::{PI, TAU};

/// PCG-XSH-RR 64/32, with a separate odd stream increment.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Independent of render scheduling, tile layout, or samples per batch.
    pub fn for_sample(seed: u64, pixel: u64, sample: u64) -> Self {
        let key = splitmix64(
            seed ^ splitmix64(pixel) ^ splitmix64(sample.wrapping_add(0x9e37_79b9_7f4a_7c15)),
        );
        Self::new(key, splitmix64(key ^ 0xda3e_39cb_94b9_5bdb))
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.increment);
        let bits = (((old >> 18) ^ old) >> 27) as u32;
        bits.rotate_right((old >> 59) as u32)
    }

    /// Strictly between zero and one, including after floating point rounding.
    /// 52 bits plus a half-bin offset are exactly representable in f64.
    pub fn open01(&mut self) -> f64 {
        let bits = ((u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())) >> 12;
        (bits as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// HG with the cosine between propagation directions. Positive g is forward.
pub fn henyey_greenstein(cos_theta: f64, g: f64) -> f64 {
    let cosine = cos_theta.clamp(-1.0, 1.0);
    let denominator = if g >= 0.0 {
        (1.0 - g).powi(2) + 2.0 * g * (1.0 - cosine)
    } else {
        (1.0 + g).powi(2) - 2.0 * g * (1.0 + cosine)
    };
    (1.0 - g) * (1.0 + g) / (4.0 * PI * denominator * denominator.sqrt())
}

/// Exact inverse HG CDF, using a stable polynomial for g near zero.
/// No isotropic approximation is substituted for small, nonzero g.
pub fn sample_hg_cosine(u: f64, g: f64) -> f64 {
    if g.abs() < 0.01 {
        let a = 2.0 * u - 1.0;
        let numerator = a + 0.5 * g * (a * a + 3.0) + g * g * a + 0.5 * g * g * g * (a * a - 1.0);
        (numerator / (1.0 + g * a).powi(2)).clamp(-1.0, 1.0)
    } else {
        let q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        ((1.0 + g * g - q * q) / (2.0 * g)).clamp(-1.0, 1.0)
    }
}

pub fn sample_henyey_greenstein(direction: DVec3, g: f64, rng: &mut Pcg32) -> DVec3 {
    let cosine = sample_hg_cosine(rng.open01(), g);
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    let azimuth = TAU * rng.open01();
    let (tangent, bitangent) = basis(direction);
    (direction * cosine + (tangent * azimuth.cos() + bitangent * azimuth.sin()) * sine).normalize()
}

pub fn sample_cosine_hemisphere(normal: DVec3, rng: &mut Pcg32) -> DVec3 {
    let radius_squared = rng.open01();
    let radius = radius_squared.sqrt();
    let azimuth = TAU * rng.open01();
    let (tangent, bitangent) = basis(normal);
    (tangent * (radius * azimuth.cos())
        + bitangent * (radius * azimuth.sin())
        + normal * (1.0 - radius_squared).sqrt())
    .normalize()
}

fn basis(normal: DVec3) -> (DVec3, DVec3) {
    let helper = if normal.z.abs() < 0.999 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let tangent = helper.cross(normal).normalize();
    (tangent, normal.cross(tangent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_stream_is_reproducible_and_open() {
        let mut a = Pcg32::for_sample(7, 31, 129);
        let mut b = Pcg32::for_sample(7, 31, 129);
        let mut c = Pcg32::for_sample(7, 31, 130);
        assert_ne!(a.next_u32(), c.next_u32());
        b.next_u32();
        for _ in 0..10_000 {
            let u = a.open01();
            assert_eq!(u, b.open01());
            assert!(u > 0.0 && u < 1.0);
        }
    }

    #[test]
    fn hg_samples_match_first_and_second_moments() {
        // HG has E[cos(theta)]=g and E[cos(theta)^2]=(1+2g^2)/3.
        let count = 20_000;
        for g in [-0.877, -0.001, 0.0, 0.001, 0.877] {
            let mut first = 0.0;
            let mut second = 0.0;
            for i in 0..count {
                let c = sample_hg_cosine((i as f64 + 0.5) / count as f64, g);
                first += c;
                second += c * c;
            }
            assert!((first / count as f64 - g).abs() < 1e-6);
            assert!((second / count as f64 - (1.0 + 2.0 * g * g) / 3.0).abs() < 2e-6);
        }
    }

    #[test]
    fn hg_small_g_is_not_replaced_with_isotropic() {
        assert!(sample_hg_cosine(0.5, 1e-5) > 1e-5);
        assert!((henyey_greenstein(1.0, 0.0) * 4.0 * PI - 1.0).abs() < 1e-15);
        assert!(henyey_greenstein(1.0, 0.877) > henyey_greenstein(-1.0, 0.877));
    }
}
