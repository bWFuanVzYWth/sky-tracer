use crate::atmosphere::{AerosolProfilePoint, AtmosphericProfilePoint, SPECIES_COUNT, SceneData};
use crate::geometry::altitude_km;
use crate::math::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct MediumCoefficients {
    pub rayleigh_scattering_km_inv: f32,
    pub aerosol_scattering_km_inv: [f32; SPECIES_COUNT],
    pub aerosol_absorption_km_inv: [f32; SPECIES_COUNT],
    pub ozone_absorption_km_inv: f32,
}

impl MediumCoefficients {
    pub fn scattering_total(self) -> f32 {
        self.rayleigh_scattering_km_inv + self.aerosol_scattering_km_inv.iter().sum::<f32>()
    }

    pub fn absorption_total(self) -> f32 {
        self.ozone_absorption_km_inv + self.aerosol_absorption_km_inv.iter().sum::<f32>()
    }

    pub fn extinction_total(self) -> f32 {
        self.scattering_total() + self.absorption_total()
    }
}

pub fn coefficients_at(scene: &SceneData, position: Vec3, band_index: usize) -> MediumCoefficients {
    let altitude = altitude_km(scene.planet, position);
    if !(0.0..=scene.planet.atmosphere_radius_km - scene.planet.ground_radius_km)
        .contains(&altitude)
    {
        return MediumCoefficients::default();
    }

    let atm = interpolate_atmosphere(&scene.atmospheric_profile, altitude);
    let aerosol = interpolate_aerosol(&scene.aerosol_profile, altitude);
    let band = scene.bands[band_index];

    let rayleigh = atm.air_cm3 * rayleigh_cross_section_m2(band.center_nm) * 1.0e9;
    let ozone = atm.ozone_cm3 * band.ozone_cross_section_cm2 * 1.0e5;

    let mut aerosol_sca = [0.0; SPECIES_COUNT];
    let mut aerosol_abs = [0.0; SPECIES_COUNT];
    for species in 0..SPECIES_COUNT {
        let optics = scene.aerosol_optics[band_index][species];
        aerosol_sca[species] = aerosol.mass_g_m3[species] * optics.scattering_km_inv_per_g_m3;
        aerosol_abs[species] = aerosol.mass_g_m3[species] * optics.absorption_km_inv_per_g_m3;
    }

    MediumCoefficients {
        rayleigh_scattering_km_inv: rayleigh.max(0.0),
        aerosol_scattering_km_inv: aerosol_sca,
        aerosol_absorption_km_inv: aerosol_abs,
        ozone_absorption_km_inv: ozone.max(0.0),
    }
}

pub fn compute_majorants(scene: &SceneData) -> Vec<f32> {
    let top = scene.planet.atmosphere_radius_km - scene.planet.ground_radius_km;
    scene
        .bands
        .iter()
        .enumerate()
        .map(|(band_index, _)| {
            let mut max_ext: f32 = 0.0;
            for i in 0..=512 {
                let altitude = top * i as f32 / 512.0;
                let pos = Vec3::new(0.0, scene.planet.ground_radius_km + altitude, 0.0);
                max_ext = max_ext.max(coefficients_at(scene, pos, band_index).extinction_total());
            }
            (max_ext * 1.25).max(1.0e-6)
        })
        .collect()
}

pub fn rayleigh_cross_section_m2(wavelength_nm: f32) -> f32 {
    5.8e-31 * (550.0 / wavelength_nm).powi(4)
}

fn interpolate_atmosphere(
    profile: &[AtmosphericProfilePoint],
    altitude_km: f32,
) -> AtmosphericProfilePoint {
    let (lo, hi, t) = bracket(profile, altitude_km, |p| p.altitude_km);
    AtmosphericProfilePoint {
        altitude_km,
        temperature_k: lerp(lo.temperature_k, hi.temperature_k, t),
        air_cm3: lerp(lo.air_cm3, hi.air_cm3, t),
        ozone_cm3: lerp(lo.ozone_cm3, hi.ozone_cm3, t),
    }
}

fn interpolate_aerosol(profile: &[AerosolProfilePoint], altitude_km: f32) -> AerosolProfilePoint {
    let (lo, hi, t) = bracket(profile, altitude_km, |p| p.altitude_km);
    let mut mass = [0.0; SPECIES_COUNT];
    for (i, value) in mass.iter_mut().enumerate() {
        *value = lerp(lo.mass_g_m3[i], hi.mass_g_m3[i], t).max(0.0);
    }
    AerosolProfilePoint {
        altitude_km,
        mass_g_m3: mass,
    }
}

fn bracket<T: Copy>(items: &[T], x: f32, key: impl Fn(T) -> f32) -> (T, T, f32) {
    assert!(!items.is_empty());
    let hi_idx = items.partition_point(|item| key(*item) < x);

    if hi_idx == 0 {
        return (items[0], items[0], 0.0);
    }
    if hi_idx >= items.len() {
        let last = items[items.len() - 1];
        return (last, last, 0.0);
    }

    let lo = items[hi_idx - 1];
    let hi = items[hi_idx];
    let klo = key(lo);
    let khi = key(hi);
    let t = if khi > klo {
        (x - klo) / (khi - klo)
    } else {
        0.0
    };
    (lo, hi, t.clamp(0.0, 1.0))
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::{
        AerosolOptics, AtmosphericProfilePoint, GROUND_ALBEDO, PHASE_BINS, Sun,
    };
    use crate::geometry::Planet;
    use crate::phase::MiePhaseTable;
    use crate::spectrum::SpectralBand;

    #[test]
    fn coefficients_are_non_negative_and_add_up() {
        let planet = Planet::earth_reference();
        let bands = vec![SpectralBand {
            index: 0,
            center_nm: 550.0,
            lower_nm: 540.0,
            upper_nm: 560.0,
            solar_irradiance_w_m2: 1.0,
            ozone_cross_section_cm2: 1.0e-21,
        }];
        let optics = vec![
            [AerosolOptics {
                scattering_km_inv_per_g_m3: 1.0,
                absorption_km_inv_per_g_m3: 0.5,
            }; SPECIES_COUNT],
        ];
        let scene = SceneData {
            planet,
            sun: Sun::from_degrees(0.0, 0.0),
            bands,
            atmospheric_profile: vec![AtmosphericProfilePoint {
                altitude_km: 0.0,
                temperature_k: 288.0,
                air_cm3: 2.5e19,
                ozone_cm3: 1.0e12,
            }],
            aerosol_profile: vec![AerosolProfilePoint {
                altitude_km: 0.0,
                mass_g_m3: [1.0e-6; SPECIES_COUNT],
            }],
            aerosol_optics: optics,
            phase_table: MiePhaseTable::new(vec![1.0; SPECIES_COUNT * PHASE_BINS], 1),
            majorants_km_inv: vec![1.0],
        };
        let c = coefficients_at(&scene, Vec3::new(0.0, planet.ground_radius_km, 0.0), 0);
        assert!(c.rayleigh_scattering_km_inv > 0.0);
        assert!(c.extinction_total() >= c.scattering_total());
        assert!((GROUND_ALBEDO - 0.3).abs() < f32::EPSILON);
    }
}
