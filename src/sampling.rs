use std::fmt;
use std::str::FromStr;

use crate::math::{PI, TAU, Vec3, orthonormal_basis};
use crate::phase::{MiePhaseTable, rayleigh_phase};

const HALTON_BASES: [u32; 64] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerKind {
    Random,
    RandomizedQmc,
}

impl fmt::Display for SamplerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => write!(f, "random"),
            Self::RandomizedQmc => write!(f, "rqmc"),
        }
    }
}

impl FromStr for SamplerKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "random" | "prng" => Ok(Self::Random),
            "rqmc" | "qmc" | "halton" => Ok(Self::RandomizedQmc),
            _ => Err(format!(
                "unknown sampler `{s}`; expected `rqmc` or `random`"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerState {
    seed: u64,
    stream: u64,
    sample_index: u64,
    dimension: u32,
    state: u64,
    kind: SamplerKind,
}

impl SamplerState {
    pub fn new(seed: u64) -> Self {
        Self::for_sample(seed, 0, 0, SamplerKind::Random)
    }

    pub fn for_sample(seed: u64, sample_index: u64, stream: u64, kind: SamplerKind) -> Self {
        let mixed = mix64(
            seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03)
                ^ sample_index.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        Self {
            seed,
            stream,
            sample_index,
            dimension: 0,
            state: mixed,
            kind,
        }
    }

    pub fn fork(&self, stream: u64) -> Self {
        Self::for_sample(
            self.seed,
            self.sample_index,
            self.stream ^ stream.wrapping_mul(0xA24B_AED4_963E_E407),
            self.kind,
        )
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut x = self.state;
        x ^= x >> 18;
        ((x >> 27) as u32).rotate_right((x >> 59) as u32)
    }

    pub fn next_f32(&mut self) -> f32 {
        match self.kind {
            SamplerKind::Random => self.next_random_f32(),
            SamplerKind::RandomizedQmc => {
                let dimension = self.dimension;
                self.dimension = self.dimension.wrapping_add(1);
                randomized_halton(self.sample_index, dimension, self.seed ^ self.stream)
                    .unwrap_or_else(|| self.next_random_f32())
            }
        }
    }

    fn next_random_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }
}

pub fn sample_isotropic(rng: &mut SamplerState) -> (Vec3, f32) {
    let z = 1.0 - 2.0 * rng.next_f32();
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = TAU * rng.next_f32();
    let dir = Vec3::new(r * phi.cos(), z, r * phi.sin());
    (dir, 1.0 / (4.0 * PI))
}

pub fn sample_rayleigh_phase(axis: Vec3, rng: &mut SamplerState) -> (Vec3, f32) {
    let mu = invert_rayleigh_cdf(rng.next_f32());
    let dir = direction_from_axis_mu(axis, mu, rng.next_f32());
    (dir, rayleigh_phase(mu))
}

pub fn sample_mie_phase(
    axis: Vec3,
    table: &MiePhaseTable,
    species_index: usize,
    band_index: usize,
    rng: &mut SamplerState,
) -> (Vec3, f32) {
    let (mu, pdf) = table.sample_mu_pdf(species_index, band_index, rng.next_f32());
    (direction_from_axis_mu(axis, mu, rng.next_f32()), pdf)
}

pub fn sample_uniform_cone(axis: Vec3, angular_radius: f32, rng: &mut SamplerState) -> (Vec3, f32) {
    let cos_max = angular_radius.cos();
    let cos_theta = 1.0 - rng.next_f32() * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * rng.next_f32();
    let (t, b) = orthonormal_basis(axis);
    let local = t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + axis * cos_theta;
    let pdf = 1.0 / (TAU * (1.0 - cos_max));
    (local.normalized(), pdf)
}

pub fn direction_in_cone(dir: Vec3, axis: Vec3, angular_radius: f32) -> bool {
    dir.normalized().dot(axis.normalized()) >= angular_radius.cos()
}

fn direction_from_axis_mu(axis: Vec3, mu: f32, xi_phi: f32) -> Vec3 {
    let mu = mu.clamp(-1.0, 1.0);
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    let phi = TAU * xi_phi;
    let axis = axis.normalized();
    let (t, b) = orthonormal_basis(axis);
    (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + axis * mu).normalized()
}

fn invert_rayleigh_cdf(xi: f32) -> f32 {
    let y = 4.0 * xi.clamp(0.0, 1.0) - 2.0;
    (2.0 * (y.asinh() / 3.0).sinh()).clamp(-1.0, 1.0)
}

fn randomized_halton(sample_index: u64, dimension: u32, seed: u64) -> Option<f32> {
    let base = *HALTON_BASES.get(dimension as usize)?;
    let value = radical_inverse(sample_index, base);
    let shift = hash_to_unit_float(mix64(
        seed ^ (dimension as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    ));
    Some(fract(value + shift))
}

fn radical_inverse(mut index: u64, base: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut reversed = 0.0;
    let mut inv = inv_base;
    while index > 0 {
        let digit = index % base as u64;
        reversed += digit as f32 * inv;
        index /= base as u64;
        inv *= inv_base;
    }
    reversed
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn hash_to_unit_float(x: u64) -> f32 {
    ((x >> 40) as u32 as f32) * (1.0 / 16_777_216.0)
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_cone_contains_sampled_direction() {
        let mut rng = SamplerState::new(7);
        let axis = Vec3::Z;
        let (dir, pdf) = sample_uniform_cone(axis, 0.01, &mut rng);
        assert!(direction_in_cone(dir, axis, 0.01));
        assert!(pdf.is_finite() && pdf > 0.0);
    }

    #[test]
    fn rayleigh_phase_sample_has_matching_pdf() {
        let mut rng = SamplerState::new(9);
        let axis = Vec3::Z;
        let (dir, pdf) = sample_rayleigh_phase(axis, &mut rng);
        let mu = dir.dot(axis).clamp(-1.0, 1.0);
        assert!((pdf - rayleigh_phase(mu)).abs() < 1.0e-5);
    }

    #[test]
    fn rayleigh_inverse_hits_distribution_endpoints() {
        assert!((invert_rayleigh_cdf(0.0) + 1.0).abs() < 1.0e-6);
        assert!(invert_rayleigh_cdf(0.5).abs() < 1.0e-6);
        assert!((invert_rayleigh_cdf(1.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn randomized_qmc_is_reproducible_and_uniform() {
        let mut sum = 0.0;
        for sample in 0..128 {
            let mut a = SamplerState::for_sample(11, sample, 3, SamplerKind::RandomizedQmc);
            let mut b = SamplerState::for_sample(11, sample, 3, SamplerKind::RandomizedQmc);
            let va = a.next_f32();
            let vb = b.next_f32();
            assert_eq!(va, vb);
            assert!((0.0..1.0).contains(&va));
            sum += va;
        }
        assert!((sum / 128.0 - 0.5).abs() < 0.02);
    }
}
