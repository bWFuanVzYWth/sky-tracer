//! Pack the existing tabulated medium for f32 interpolation and GPU transport.
//! Source coefficients retain their original precision and spectral bandwidth.
use crate::physics::atmosphere::{PHASE_BINS, SPECIES_COUNT, SceneData};
use crate::{Result, geometry::Geometry};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BandInfo {
    pub center_nm: f32,
    pub lower_nm: f32,
    pub upper_nm: f32,
    pub solar_irradiance_w_m2: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Coefficients {
    pub scattering: [f32; 5],
    pub extinction: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BandModel {
    pub info: BandInfo,
    pub profile: Vec<(f32, Coefficients)>,
    pub phase: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Model {
    pub geometry: Geometry,
    pub sun_radius: f32,
    pub bands: Vec<BandModel>,
}

impl Model {
    /// Construct this solver's independently owned default Earth model.
    pub fn earth() -> Result<Self> {
        Self::earth_with_aerosol_scale(1.0)
    }
    pub fn earth_with_aerosol_scale(scale: f32) -> Result<Self> {
        if !scale.is_finite() || !(0.0..=16.0).contains(&scale) {
            return Err("invalid aerosol multiplier".into());
        }
        let mut scene = crate::physics::data::load_scene_data(
            &crate::physics::data::default_data_dir(),
            0.0,
            0.0,
        )
        .map_err(|e| e.to_string())?;
        for point in &mut scene.aerosol_profile {
            for mass in &mut point.mass_g_m3 {
                *mass *= scale;
            }
        }
        Self::from_scene(&scene)
    }

    pub fn from_scene(scene: &SceneData) -> Result<Self> {
        let count = scene.bands.len();
        if count == 0
            || scene.aerosol_optics.len() != count
            || scene.rayleigh_cross_sections_m2.len() != count
            || scene.phase_table.band_count() != count
            || scene.atmospheric_profile.is_empty()
            || scene.aerosol_profile.is_empty()
        {
            return Err("incomplete scene spectral or medium data".into());
        }
        let geometry = Geometry {
            bottom: scene.planet.ground_radius_km,
            top: scene.planet.atmosphere_radius_km,
        };
        if !geometry.bottom.is_finite()
            || !geometry.top.is_finite()
            || geometry.bottom <= 0.0
            || geometry.top <= geometry.bottom
            || !scene.sun.angular_radius_rad.is_finite()
            || scene.sun.angular_radius_rad <= 0.0
            || !(0.0..0.1).contains(&scene.sun.angular_radius_rad)
        {
            return Err("invalid planet or solar angular radius".into());
        }
        let atm: Vec<_> = scene
            .atmospheric_profile
            .iter()
            .map(|p| (p.altitude_km, [p.air_cm3, p.ozone_cm3]))
            .collect();
        let aerosol: Vec<_> = scene
            .aerosol_profile
            .iter()
            .map(|p| (p.altitude_km, p.mass_g_m3.map(f32::from)))
            .collect();
        validate_profile(&atm)?;
        validate_profile(&aerosol)?;
        let mut heights: Vec<_> = atm
            .iter()
            .map(|p| p.0)
            .chain(aerosol.iter().map(|p| p.0))
            .filter(|&h| h >= 0.0 && h <= geometry.top - geometry.bottom)
            .collect();
        heights.extend([0.0, geometry.top - geometry.bottom]);
        heights.sort_by(f32::total_cmp);
        heights.dedup();
        let mut bands = Vec::with_capacity(count);
        for (index, band) in scene.bands.iter().enumerate() {
            if !band.center_nm.is_finite()
                || !band.lower_nm.is_finite()
                || !band.upper_nm.is_finite()
                || band.lower_nm >= band.upper_nm
                || band.center_nm < band.lower_nm
                || band.center_nm > band.upper_nm
                || !band.solar_irradiance_w_m2.is_finite()
                || band.solar_irradiance_w_m2 < 0.0
                || !band.ozone_cross_section_cm2.is_finite()
                || band.ozone_cross_section_cm2 < 0.0
                || !scene.rayleigh_cross_sections_m2[index].is_finite()
                || scene.rayleigh_cross_sections_m2[index] < 0.0
            {
                return Err(format!("invalid spectral band {index}").into());
            }
            for o in scene.aerosol_optics[index] {
                if [o.scattering_km_inv_per_g_m3, o.absorption_km_inv_per_g_m3]
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err("invalid aerosol optics".into());
                }
            }
            let profile = heights
                .iter()
                .map(|&h| {
                    let air = interpolate(&atm, h);
                    let mass = interpolate(&aerosol, h);
                    let mut c = Coefficients::default();
                    c.scattering[0] = air[0] * scene.rayleigh_cross_sections_m2[index] * 1e9;
                    c.extinction = c.scattering[0] + air[1] * band.ozone_cross_section_cm2 * 1e5;
                    for species in 0..SPECIES_COUNT {
                        let optics = scene.aerosol_optics[index][species];
                        c.scattering[species + 1] =
                            mass[species] * optics.scattering_km_inv_per_g_m3;
                        c.extinction += c.scattering[species + 1]
                            + mass[species] * optics.absorption_km_inv_per_g_m3;
                    }
                    (h, c)
                })
                .collect();
            let phase: Vec<_> = (0..SPECIES_COUNT)
                .flat_map(|s| (0..PHASE_BINS).map(move |b| scene.phase_table.value(s, index, b)))
                .collect();
            if phase.iter().any(|p| !p.is_finite() || *p < 0.0) {
                return Err("invalid phase samples".into());
            }
            bands.push(BandModel {
                info: BandInfo {
                    center_nm: band.center_nm,
                    lower_nm: band.lower_nm,
                    upper_nm: band.upper_nm,
                    solar_irradiance_w_m2: band.solar_irradiance_w_m2,
                },
                profile,
                phase,
            });
        }
        Ok(Self {
            geometry,
            sun_radius: scene.sun.angular_radius_rad,
            bands,
        })
    }
}

impl BandModel {
    pub fn coefficients(&self, altitude: f32) -> Coefficients {
        let (a, b, t) = bracket(&self.profile, altitude);
        Coefficients {
            scattering: std::array::from_fn(|i| a.scattering[i] * (1.0 - t) + b.scattering[i] * t),
            extinction: a.extinction * (1.0 - t) + b.extinction * t,
        }
    }

    pub fn phases(&self, mu: f32) -> [f32; 5] {
        let mu = mu.clamp(-1.0, 1.0);
        let f = ((1.0 - mu) * 0.5).cbrt() * PHASE_BINS as f32 - 0.5;
        let lo = f.floor().clamp(0.0, (PHASE_BINS - 1) as f32) as usize;
        let hi = (lo + 1).min(PHASE_BINS - 1);
        let t = (f - lo as f32).clamp(0.0, 1.0);
        std::array::from_fn(|s| {
            if s == 0 {
                3.0 * (1.0 + mu * mu) / (16.0 * std::f32::consts::PI)
            } else {
                self.phase[(s - 1) * PHASE_BINS + lo] * (1.0 - t)
                    + self.phase[(s - 1) * PHASE_BINS + hi] * t
            }
        })
    }
}

pub fn phase_weight(c: Coefficients, phases: [f32; 5]) -> f32 {
    c.scattering.iter().zip(phases).map(|(a, b)| a * b).sum()
}

fn validate_profile<const N: usize>(p: &[(f32, [f32; N])]) -> Result<()> {
    if p.iter()
        .any(|(h, v)| !h.is_finite() || v.iter().any(|x| !x.is_finite() || *x < 0.0))
        || p.windows(2).any(|w| w[0].0 >= w[1].0)
    {
        return Err(
            "profiles must have finite nonnegative values and strictly increasing heights".into(),
        );
    }
    Ok(())
}

fn interpolate<const N: usize>(p: &[(f32, [f32; N])], x: f32) -> [f32; N] {
    let (a, b, t) = bracket(p, x);
    std::array::from_fn(|i| a[i] * (1.0 - t) + b[i] * t)
}

fn bracket<T: Copy>(p: &[(f32, T)], x: f32) -> (T, T, f32) {
    let hi = p.partition_point(|v| v.0 < x);
    if hi == 0 {
        return (p[0].1, p[0].1, 0.0);
    }
    if hi == p.len() {
        return (p[hi - 1].1, p[hi - 1].1, 0.0);
    }
    let (h0, a) = p[hi - 1];
    let (h1, b) = p[hi];
    (a, b, ((x - h0) / (h1 - h0)).clamp(0.0, 1.0))
}
