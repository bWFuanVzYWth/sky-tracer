use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::atmosphere::{
    AerosolOptics, AerosolProfilePoint, AtmosphericProfilePoint, PHASE_BINS, SPECIES_COUNT,
    SPECIES_NAMES, SceneData, Sun,
};
use crate::geometry::Planet;
use crate::medium::{compute_majorant_grid, compute_majorants, rayleigh_cross_section_m2};
use crate::phase::MiePhaseTable;
use crate::spectrum::{BAND_COUNT, SpectralBand};

pub fn load_scene_data(
    data_dir: &Path,
    sun_elevation_deg: f32,
    sun_azimuth_deg: f32,
) -> Result<SceneData, Box<dyn Error>> {
    let bands = load_bands(&data_dir.join("bands.csv"))?;
    let atmospheric_profile = load_atmospheric_profile(&data_dir.join("atmosphere_profile.csv"))?;
    let aerosol_profile = load_aerosol_profile(&data_dir.join("aerosol_profile.csv"))?;
    let aerosol_optics = load_aerosol_optics(&data_dir.join("aerosol_optics.csv"), bands.len())?;
    let phase_table = load_mie_phase(&data_dir.join("mie_phase.csv"), bands.len())?;
    let sun = Sun::from_degrees(sun_elevation_deg, sun_azimuth_deg);
    let rayleigh_cross_sections_m2 = bands
        .iter()
        .map(|band| rayleigh_cross_section_m2(band.center_nm))
        .collect();
    let solar_radiance_w_m2_sr = bands
        .iter()
        .map(|band| band.solar_irradiance_w_m2 / sun.solid_angle_sr)
        .collect();

    let mut scene = SceneData {
        planet: Planet::earth_reference(),
        sun,
        bands,
        rayleigh_cross_sections_m2,
        solar_radiance_w_m2_sr,
        atmospheric_profile,
        aerosol_profile,
        aerosol_optics,
        phase_table,
        majorants_km_inv: Vec::new(),
        majorant_grid: crate::atmosphere::MajorantGrid::new(120.0, 1, 1),
    };
    scene.majorant_grid = compute_majorant_grid(&scene, 64);
    scene.majorants_km_inv = compute_majorants(&scene);
    Ok(scene)
}

fn load_bands(path: &Path) -> Result<Vec<SpectralBand>, Box<dyn Error>> {
    let mut bands = Vec::new();
    for cols in csv_rows(path)? {
        if cols.len() != 6 {
            return Err(format!("{}: expected 6 columns", path.display()).into());
        }
        bands.push(SpectralBand {
            index: parse(&cols[0])?,
            center_nm: parse(&cols[1])?,
            lower_nm: parse(&cols[2])?,
            upper_nm: parse(&cols[3])?,
            solar_irradiance_w_m2: parse(&cols[4])?,
            ozone_cross_section_cm2: parse(&cols[5])?,
        });
    }
    if bands.len() != BAND_COUNT {
        return Err(format!("expected {BAND_COUNT} bands, found {}", bands.len()).into());
    }
    Ok(bands)
}

fn load_atmospheric_profile(path: &Path) -> Result<Vec<AtmosphericProfilePoint>, Box<dyn Error>> {
    let mut profile = Vec::new();
    for cols in csv_rows(path)? {
        if cols.len() != 4 {
            return Err(format!("{}: expected 4 columns", path.display()).into());
        }
        profile.push(AtmosphericProfilePoint {
            altitude_km: parse(&cols[0])?,
            temperature_k: parse(&cols[1])?,
            air_cm3: parse(&cols[2])?,
            ozone_cm3: parse(&cols[3])?,
        });
    }
    profile.sort_by(|a, b| a.altitude_km.total_cmp(&b.altitude_km));
    Ok(profile)
}

fn load_aerosol_profile(path: &Path) -> Result<Vec<AerosolProfilePoint>, Box<dyn Error>> {
    let mut profile = Vec::new();
    for cols in csv_rows(path)? {
        if cols.len() != 1 + SPECIES_COUNT {
            return Err(
                format!("{}: expected {} columns", path.display(), 1 + SPECIES_COUNT).into(),
            );
        }
        let mut mass = [0.0; SPECIES_COUNT];
        for i in 0..SPECIES_COUNT {
            mass[i] = parse(&cols[i + 1])?;
        }
        profile.push(AerosolProfilePoint {
            altitude_km: parse(&cols[0])?,
            mass_g_m3: mass,
        });
    }
    profile.sort_by(|a, b| a.altitude_km.total_cmp(&b.altitude_km));
    Ok(profile)
}

fn load_aerosol_optics(
    path: &Path,
    band_count: usize,
) -> Result<Vec<[AerosolOptics; SPECIES_COUNT]>, Box<dyn Error>> {
    let mut optics = vec![[AerosolOptics::default(); SPECIES_COUNT]; band_count];
    let mut seen = vec![[false; SPECIES_COUNT]; band_count];
    for cols in csv_rows(path)? {
        if cols.len() != 4 {
            return Err(format!("{}: expected 4 columns", path.display()).into());
        }
        let species = species_index(&cols[0])?;
        let band: usize = parse(&cols[1])?;
        optics[band][species] = AerosolOptics {
            scattering_km_inv_per_g_m3: parse(&cols[2])?,
            absorption_km_inv_per_g_m3: parse(&cols[3])?,
        };
        seen[band][species] = true;
    }
    assert_all_seen(path, &seen)?;
    Ok(optics)
}

fn load_mie_phase(path: &Path, band_count: usize) -> Result<MiePhaseTable, Box<dyn Error>> {
    let mut values = vec![0.0; SPECIES_COUNT * band_count * PHASE_BINS];
    let mut seen = vec![false; values.len()];
    for cols in csv_rows(path)? {
        if cols.len() != 4 {
            return Err(format!("{}: expected 4 columns", path.display()).into());
        }
        let species = species_index(&cols[0])?;
        let band: usize = parse(&cols[1])?;
        let bin: usize = parse(&cols[2])?;
        let idx = ((species * band_count + band) * PHASE_BINS) + bin;
        values[idx] = parse(&cols[3])?;
        seen[idx] = true;
    }
    if let Some(missing) = seen.iter().position(|x| !*x) {
        return Err(format!("{}: missing phase entry index {missing}", path.display()).into());
    }
    Ok(MiePhaseTable::new(values, band_count))
}

fn csv_rows(path: &Path) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        rows.push(trimmed.split(',').map(|s| s.trim().to_owned()).collect());
    }
    Ok(rows)
}

fn parse<T>(s: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    Ok(s.parse::<T>()?)
}

fn species_index(name: &str) -> Result<usize, Box<dyn Error>> {
    SPECIES_NAMES
        .iter()
        .position(|x| *x == name)
        .ok_or_else(|| format!("unknown species {name}").into())
}

fn assert_all_seen(path: &Path, seen: &[[bool; SPECIES_COUNT]]) -> Result<(), Box<dyn Error>> {
    for (band, species_seen) in seen.iter().enumerate() {
        for (species, ok) in species_seen.iter().enumerate() {
            if !ok {
                return Err(format!(
                    "{}: missing optics for band {band}, species {}",
                    path.display(),
                    SPECIES_NAMES[species]
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_data_loads() {
        let scene = load_scene_data(&repository_data_dir(), 0.0, 0.0).expect("data should load");
        assert_eq!(scene.bands.len(), BAND_COUNT);
        assert_eq!(scene.rayleigh_cross_sections_m2.len(), BAND_COUNT);
        assert_eq!(scene.solar_radiance_w_m2_sr.len(), BAND_COUNT);
        assert_eq!(scene.phase_table.band_count(), BAND_COUNT);
        assert_eq!(scene.majorants_km_inv.len(), BAND_COUNT);
        assert!(
            scene
                .majorants_km_inv
                .iter()
                .all(|x| x.is_finite() && *x > 0.0)
        );
        assert_eq!(scene.majorant_grid.band_count, BAND_COUNT);
        assert!(scene.majorant_grid.layer_count > 1);
    }

    #[test]
    fn baked_mie_phase_is_normalized() {
        let scene = load_scene_data(&repository_data_dir(), 0.0, 0.0).expect("data should load");
        for species in 0..SPECIES_COUNT {
            for band in 0..BAND_COUNT {
                let integral = integrate_phase(&scene, species, band);
                assert!(
                    (integral - 1.0).abs() < 0.005,
                    "species={species} band={band} integral={integral}"
                );
            }
        }
    }

    fn integrate_phase(scene: &SceneData, species: usize, band: usize) -> f32 {
        let mut sum = 0.0;
        for i in 0..PHASE_BINS - 1 {
            let mu0 = cube_mu(i);
            let mu1 = cube_mu(i + 1);
            let p0 = scene.phase_table.value(species, band, i);
            let p1 = scene.phase_table.value(species, band, i + 1);
            sum += 0.5 * (p0 + p1) * (mu0 - mu1).abs() * std::f32::consts::TAU;
        }
        sum
    }

    fn cube_mu(i: usize) -> f32 {
        let u = (i as f32 + 0.5) / PHASE_BINS as f32;
        1.0 - 2.0 * u * u * u
    }

    #[test]
    fn layered_majorants_dominate_extinction_samples() {
        let scene = load_scene_data(&repository_data_dir(), 0.0, 0.0).expect("data should load");
        for band in 0..BAND_COUNT {
            for i in 0..=240 {
                let altitude = scene.majorant_grid.top_altitude_km * i as f32 / 240.0;
                let pos =
                    crate::math::Vec3::new(0.0, scene.planet.ground_radius_km + altitude, 0.0);
                let extinction =
                    crate::medium::coefficients_at(&scene, pos, band).extinction_total();
                let layer = scene.majorant_grid.layer_for_altitude(altitude);
                let majorant = scene.majorant_grid.get(band, layer);
                let minorant = scene.majorant_grid.minorant(band, layer);
                assert!(
                    extinction <= majorant * 1.001,
                    "band={band} altitude={altitude} extinction={extinction} majorant={majorant}"
                );
                assert!(
                    minorant <= extinction * 1.001 + 1.0e-8,
                    "band={band} altitude={altitude} extinction={extinction} minorant={minorant}"
                );
            }
        }
    }

    fn repository_data_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("data")
    }
}
