use std::f32::consts::PI;

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
