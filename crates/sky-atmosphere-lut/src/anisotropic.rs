//! CPU reference for directional multiple-scattering source compression.
//!
//! The incident field contains diffuse sky AND the ground, but no solar disk.
//! Convolving that field gives orders >= 2; the direct solar source is separate.
//! All arithmetic is f32. Reflection symmetry in the up/Sun plane removes the
//! sine harmonics, but does not make multiple scattering isotropic.
use crate::model::{BandModel, Coefficients};
use glam::Vec3;
use std::f32::consts::PI;

pub fn count(degree: usize) -> usize {
    (degree + 1) * (degree + 2) / 2
}
pub fn index(l: usize, m: usize) -> usize {
    l * (l + 1) / 2 + m
}

/// Orthonormal real cosine SH, z=radial up, x=tangent toward the Sun.
/// No Condon-Shortley sign (consistent on both projection and reconstruction).
pub fn cosine_sh(direction: Vec3, degree: usize, out: &mut [f32]) {
    assert_eq!(out.len(), count(degree));
    let z = direction.z.clamp(-1.0, 1.0);
    let radial = direction.x.hypot(direction.y);
    let (cp, sp) = if radial > 1e-10 {
        (direction.x / radial, direction.y / radial)
    } else {
        (1.0, 0.0)
    };
    let mut diagonal = (4.0 * PI).sqrt().recip();
    let (mut cm, mut sm) = (1.0, 0.0);
    for m in 0..=degree {
        if m > 0 {
            diagonal *= if m == 1 {
                3.0_f32.sqrt() * radial
            } else {
                ((2 * m + 1) as f32 / (2 * m) as f32).sqrt() * radial
            };
            (cm, sm) = (cm * cp - sm * sp, sm * cp + cm * sp);
        }
        out[index(m, m)] = diagonal * cm;
        let (mut previous2, mut previous) = (0.0, diagonal);
        for l in m + 1..=degree {
            let value = if l == m + 1 {
                ((2 * m + 3) as f32).sqrt() * z * previous
            } else {
                let lf = l as f32;
                let mf = m as f32;
                let p = lf - 1.0;
                let a = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
                let b = ((p * p - mf * mf) / (4.0 * p * p - 1.0)).sqrt();
                a * (z * previous - b * previous2)
            };
            out[index(l, m)] = value * cm;
            (previous2, previous) = (previous, value);
        }
    }
}

/// a_l = 2 pi integral p(mu) P_l(mu) dmu for each scattering species.
/// A linear phase table in cubic-cosine coordinates is integrated in that same
/// coordinate, retaining narrow forward peaks rather than sampling uniform mu.
pub fn phase_moments(model: &BandModel, degree: usize) -> Vec<[f32; 5]> {
    let mut result = vec![[0.0; 5]; degree + 1];
    let n = 16_384;
    let mut correction = result.clone();
    for i in 0..n {
        let u = (i as f32 + 0.5) / n as f32;
        let mu = 1.0 - 2.0 * u * u * u;
        let phase = model.phases(mu);
        let weight = 12.0 * PI * u * u / n as f32;
        let (mut previous2, mut previous) = (0.0, 1.0);
        for l in 0..=degree {
            let value = if l == 0 {
                1.0
            } else {
                ((2 * l - 1) as f32 * mu * previous - (l - 1) as f32 * previous2) / l as f32
            };
            for s in 1..5 {
                let term = phase[s] * weight * value - correction[l][s];
                let next = result[l][s] + term;
                correction[l][s] = (next - result[l][s]) - term;
                result[l][s] = next;
            }
            (previous2, previous) = (previous, value);
        }
    }
    result[0][0] = 1.0;
    if degree >= 2 {
        result[2][0] = 0.1;
    }
    result
}

/// Evaluate the physical source per unit distance. Negative truncated values
/// are deliberately retained here so a validator can detect ringing.
pub fn source(
    coefficients: Coefficients,
    moments: &[f32],
    phase: &[[f32; 5]],
    basis: &[f32],
    degree: usize,
) -> f32 {
    let mut sum = 0.0;
    for (l, p) in phase.iter().enumerate().take(degree + 1) {
        let weight: f32 = p
            .iter()
            .zip(coefficients.scattering)
            .map(|(a, s)| a * s)
            .sum();
        for m in 0..=l {
            let i = index(l, m);
            sum += weight * moments[i] * basis[i];
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orthonormal_projection_and_rayleigh_convolution() {
        let degree = 6;
        let mut moments = vec![0.0; count(degree)];
        let mut basis = moments.clone();
        for (v, w) in crate::quadrature::sphere(64, 128) {
            cosine_sh(v, degree, &mut basis);
            // Contains l=0, l=1 and l=2, m=0,1,2. Rayleigh removes l=1.
            let light = 2.0
                + 0.2 * basis[index(1, 1)]
                + 0.3 * basis[index(2, 0)]
                + 0.4 * basis[index(2, 2)];
            for (p, y) in moments.iter_mut().zip(&basis) {
                *p += light * y * w;
            }
        }
        let mut phase = vec![[0.0; 5]; degree + 1];
        phase[0][0] = 1.0;
        phase[2][0] = 0.1;
        let c = Coefficients {
            scattering: [1.0, 0.0, 0.0, 0.0, 0.0],
            extinction: 1.0,
        };
        for v in [
            Vec3::Z,
            Vec3::X,
            Vec3::Y,
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            cosine_sh(v, degree, &mut basis);
            let expected = 2.0 + 0.03 * basis[index(2, 0)] + 0.04 * basis[index(2, 2)];
            assert!((source(c, &moments, &phase, &basis, degree) - expected).abs() < 2e-5);
        }
    }
}
