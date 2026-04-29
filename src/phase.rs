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
    band_count: usize,
}

impl MiePhaseTable {
    pub fn new(values: Vec<f32>, band_count: usize) -> Self {
        assert_eq!(values.len(), SPECIES_COUNT * band_count * PHASE_BINS);
        Self { values, band_count }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rayleigh_phase_is_positive() {
        assert!(rayleigh_phase(-1.0) > 0.0);
        assert!(rayleigh_phase(0.0) > 0.0);
        assert!(rayleigh_phase(1.0) > 0.0);
    }
}
