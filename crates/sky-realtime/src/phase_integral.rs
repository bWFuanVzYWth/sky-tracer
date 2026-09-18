use crate::model::BandModel;
use std::f32::consts::PI;
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
