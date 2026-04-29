use crate::atmosphere::{PHASE_BINS, SPECIES_COUNT};
use crate::math::PI;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatteringMode {
    Rayleigh,
    Aerosol { species_index: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct PhaseFrame {
    pub mu: f32,
    pub band_index: usize,
    pub mode: ScatteringMode,
}

#[derive(Clone, Debug)]
pub struct MiePhaseTable {
    values: Vec<f32>,
    cdf: Vec<f32>,
    bin_pdf: Vec<f32>,
    band_count: usize,
}

impl MiePhaseTable {
    pub fn new(values: Vec<f32>, band_count: usize) -> Self {
        assert_eq!(values.len(), SPECIES_COUNT * band_count * PHASE_BINS);
        let (cdf, bin_pdf) = build_sampling_tables(&values, band_count);
        Self {
            values,
            cdf,
            bin_pdf,
            band_count,
        }
    }

    pub fn evaluate(&self, species_index: usize, band_index: usize, mu: f32) -> f32 {
        let u = ((1.0 - mu.clamp(-1.0, 1.0)) * 0.5).cbrt();
        let f = u * PHASE_BINS as f32 - 0.5;
        let i0 = f.floor().clamp(0.0, (PHASE_BINS - 1) as f32) as usize;
        let i1 = (i0 + 1).min(PHASE_BINS - 1);
        let t = (f - i0 as f32).clamp(0.0, 1.0);
        let p0 = self.value(species_index, band_index, i0);
        let p1 = self.value(species_index, band_index, i1);
        (p0 * (1.0 - t) + p1 * t).max(0.0)
    }

    pub fn value(&self, species_index: usize, band_index: usize, bin: usize) -> f32 {
        let idx = ((species_index * self.band_count + band_index) * PHASE_BINS) + bin;
        self.values[idx]
    }

    pub fn sample_mu_pdf(&self, species_index: usize, band_index: usize, xi: f32) -> (f32, f32) {
        let base = (species_index * self.band_count + band_index) * PHASE_BINS;
        let cdf = &self.cdf[base..base + PHASE_BINS];
        let xi = xi.clamp(0.0, 1.0 - f32::EPSILON);
        let bin = cdf
            .partition_point(|value| *value <= xi)
            .min(PHASE_BINS - 1);
        let prev = if bin == 0 { 0.0 } else { cdf[bin - 1] };
        let next = cdf[bin];
        let t = if next > prev {
            (xi - prev) / (next - prev)
        } else {
            0.5
        };
        let mu = lerp(mu_edge(bin), mu_edge(bin + 1), t);
        (mu, self.bin_pdf[base + bin].max(1.0e-12))
    }

    pub fn sampling_pdf(&self, species_index: usize, band_index: usize, mu: f32) -> f32 {
        let base = (species_index * self.band_count + band_index) * PHASE_BINS;
        let bin = mu_to_bin(mu);
        self.bin_pdf[base + bin].max(1.0e-12)
    }

    pub fn band_count(&self) -> usize {
        self.band_count
    }
}

pub trait ScalarPhase {
    fn eval_scalar(&self, frame: PhaseFrame) -> f32;
}

impl ScalarPhase for MiePhaseTable {
    fn eval_scalar(&self, frame: PhaseFrame) -> f32 {
        match frame.mode {
            ScatteringMode::Rayleigh => rayleigh_phase(frame.mu),
            ScatteringMode::Aerosol { species_index } => {
                self.evaluate(species_index, frame.band_index, frame.mu)
            }
        }
    }
}

pub fn rayleigh_phase(mu: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + mu * mu)
}

fn build_sampling_tables(values: &[f32], band_count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cdf = vec![0.0; values.len()];
    let mut bin_pdf = vec![0.0; values.len()];

    for species in 0..SPECIES_COUNT {
        for band in 0..band_count {
            let base = (species * band_count + band) * PHASE_BINS;
            let norm = phase_norm(values, base).max(1.0e-20);
            let mut cumulative = 0.0;
            for bin in 0..PHASE_BINS {
                let value = values[base + bin].max(0.0);
                let delta_mu = (mu_edge(bin) - mu_edge(bin + 1)).abs();
                cumulative += 2.0 * PI * value * delta_mu / norm;
                cdf[base + bin] = cumulative.min(1.0);
                bin_pdf[base + bin] = value / norm;
            }
            cdf[base + PHASE_BINS - 1] = 1.0;
        }
    }

    (cdf, bin_pdf)
}

fn phase_norm(values: &[f32], base: usize) -> f32 {
    let mut sum = 0.0;
    for bin in 0..PHASE_BINS {
        let delta_mu = (mu_edge(bin) - mu_edge(bin + 1)).abs();
        sum += 2.0 * PI * values[base + bin].max(0.0) * delta_mu;
    }
    sum
}

fn mu_edge(bin_edge: usize) -> f32 {
    let u = bin_edge as f32 / PHASE_BINS as f32;
    1.0 - 2.0 * u * u * u
}

fn mu_to_bin(mu: f32) -> usize {
    let u = ((1.0 - mu.clamp(-1.0, 1.0)) * 0.5).cbrt();
    ((u * PHASE_BINS as f32).floor() as usize).min(PHASE_BINS - 1)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_phase_is_positive() {
        assert!(rayleigh_phase(-1.0) > 0.0);
        assert!(rayleigh_phase(0.0) > 0.0);
        assert!(rayleigh_phase(1.0) > 0.0);
    }

    #[test]
    fn uniform_mie_sampling_has_isotropic_pdf() {
        let table = MiePhaseTable::new(vec![1.0; SPECIES_COUNT * PHASE_BINS], 1);
        let (mu, pdf) = table.sample_mu_pdf(0, 0, 0.5);
        assert!((-1.0..=1.0).contains(&mu));
        assert!((pdf - 1.0 / (4.0 * PI)).abs() < 1.0e-3);
        assert!((table.sampling_pdf(0, 0, mu) - pdf).abs() < 1.0e-6);
    }
}
