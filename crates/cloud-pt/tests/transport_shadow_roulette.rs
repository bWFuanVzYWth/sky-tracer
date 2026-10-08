//! CPU ratio-tracking tests. This file never requests a GPU device or window.

use cloud_pt::{
    sampling::Pcg32,
    transport::{Bounds, DensityField, Ray, TransportSettings, transmittance},
};
use glam::DVec3;

struct Shadow {
    length: f64,
}
impl DensityField for Shadow {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: DVec3::ZERO,
            max: DVec3::new(self.length, 1.0, 1.0),
        }
    }
    fn density_world(&self, point: DVec3) -> f64 {
        if point.cmpge(self.bounds().min).all() && point.cmple(self.bounds().max).all() {
            0.1
        } else {
            0.0
        }
    }
    fn max_density(&self) -> f64 {
        1.0
    }
}

fn settings(enabled: bool) -> TransportSettings {
    TransportSettings {
        shadow_roulette: enabled,
        extinction_scale: 1.0,
        ground: None,
        ..TransportSettings::default()
    }
}
fn ray() -> Ray {
    Ray {
        origin: DVec3::new(-1.0, 0.5, 0.5),
        direction: DVec3::X,
    }
}

#[test]
fn dark_ratio_estimates_with_and_without_roulette_match_beer_lambert() {
    const COUNT: u64 = 20_000;
    let expected = (-10.0_f64).exp();
    for enabled in [false, true] {
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut nonzero = 0;
        for index in 0..COUNT {
            let value = transmittance(
                &Shadow { length: 100.0 },
                ray(),
                f64::INFINITY,
                &settings(enabled),
                &mut Pcg32::for_sample(0x5AD0_1111, u64::from(enabled), index),
            )
            .unwrap();
            nonzero += u64::from(value > 0.0);
            let delta = value - mean;
            mean += delta / (index + 1) as f64;
            m2 += delta * (value - mean);
        }
        let standard_error = (m2 / (COUNT - 1) as f64 / COUNT as f64).sqrt();
        assert!(
            (mean - expected).abs() <= 5.0 * standard_error,
            "roulette={enabled}: mean {mean}, expected {expected}, SE {standard_error}"
        );
        assert!(
            nonzero > COUNT / 10,
            "fixed half-survival keeps fractional estimates"
        );
        eprintln!("tau=10, roulette={enabled}: mean {mean:.8e}, SE {standard_error:.3e}");
    }
}

#[test]
fn bright_shadows_keep_identical_samples_and_random_streams() {
    let field = Shadow { length: 30.0 }; // Optical thickness 3, above the trigger.
    for index in 0..1024 {
        let mut no_roulette_rng = Pcg32::for_sample(0xB12_16117, 0, index);
        let mut roulette_rng = no_roulette_rng.clone();
        let a = transmittance(
            &field,
            ray(),
            f64::INFINITY,
            &settings(false),
            &mut no_roulette_rng,
        )
        .unwrap();
        let b = transmittance(
            &field,
            ray(),
            f64::INFINITY,
            &settings(true),
            &mut roulette_rng,
        )
        .unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        assert_eq!(no_roulette_rng.next_u32(), roulette_rng.next_u32());
    }
}
