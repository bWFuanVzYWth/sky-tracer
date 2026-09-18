use glam::Vec3;
use std::f32::consts::{PI, TAU};

/// Gauss-Legendre nodes and weights on [0,1].
pub fn gauss_legendre(n: usize) -> Vec<(f32, f32)> {
    assert!(n > 0);
    let mut nodes = vec![(0.0, 0.0); n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f32 + 0.75) / (n as f32 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..64 {
            let mut a = 1.0;
            let mut b = 0.0;
            for j in 1..=n {
                let c = b;
                b = a;
                a = ((2 * j - 1) as f32 * z * b - (j - 1) as f32 * c) / j as f32;
            }
            derivative = n as f32 * (z * a - b) / (z * z - 1.0);
            let step = a / derivative;
            z -= step;
            if step.abs() < 2e-7 {
                break;
            }
        }
        let weight = 1.0 / ((1.0 - z * z) * derivative * derivative);
        nodes[i] = ((1.0 - z) * 0.5, weight);
        nodes[n - 1 - i] = ((1.0 + z) * 0.5, weight);
    }
    nodes
}

/// Local directions around +Z and solid-angle weights. A cubic cosine warp
/// resolves the aerosol forward peak without dropping any of the sphere.
pub fn sphere(n_mu: usize, n_phi: usize) -> Vec<(Vec3, f32)> {
    ring_quadrature(n_mu, n_phi, |u| (1.0 - 2.0 * u.powi(3), 6.0 * u * u))
}

pub fn hemisphere(n_mu: usize, n_phi: usize) -> Vec<(Vec3, f32)> {
    ring_quadrature(n_mu, n_phi, |u| (u, 1.0))
}

/// Weights sum to one. Multiplying by band solar irradiance implements a
/// uniform finite sun disk with band-integrated irradiance in the reference model.
pub fn sun_disk(n_mu: usize, n_phi: usize, radius: f32) -> Vec<(Vec3, f32)> {
    ring_quadrature(n_mu, n_phi, |u| (1.0 - u * (1.0 - radius.cos()), 1.0 / TAU))
}

fn ring_quadrature(n_mu: usize, n_phi: usize, map: impl Fn(f32) -> (f32, f32)) -> Vec<(Vec3, f32)> {
    let mut samples = Vec::with_capacity(n_mu * n_phi);
    for (u, w) in gauss_legendre(n_mu) {
        let (mu, jacobian) = map(u);
        let sin = (1.0 - mu * mu).max(0.0).sqrt();
        for j in 0..n_phi {
            let phi = TAU * (j as f32 + 0.5) / n_phi as f32;
            samples.push((
                Vec3::new(sin * phi.cos(), sin * phi.sin(), mu),
                w * jacobian * TAU / n_phi as f32,
            ));
        }
    }
    samples
}

pub fn rotate(local: Vec3, axis: Vec3) -> Vec3 {
    let helper = if axis.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let x = helper.cross(axis).normalize();
    x * local.x + axis.cross(x) * local.y + axis * local.z
}
