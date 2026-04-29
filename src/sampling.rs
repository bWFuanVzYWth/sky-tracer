use crate::math::{PI, TAU, Vec3, orthonormal_basis};

#[derive(Clone, Copy, Debug)]
pub struct SamplerState {
    state: u64,
}

impl SamplerState {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn fork(self, stream: u64) -> Self {
        Self::new(self.state ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03))
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
}
