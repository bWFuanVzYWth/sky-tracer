use rayon::prelude::*;

use crate::atmosphere::{GROUND_ALBEDO, SPECIES_COUNT, SceneData};
use crate::config::RenderConfig;
use crate::film::Film;
use crate::geometry::{
    BoundaryKind, RAY_EPSILON_KM, next_boundary, segment_to_sun_or_space, surface_normal,
};
use crate::math::{INV_PI, Ray, TAU, Vec3};
use crate::medium::{MediumCoefficients, coefficients_at};
use crate::phase::{PhaseFrame, ScalarPhase, ScatteringMode, rayleigh_phase};
use crate::sampling::{
    SamplerState, direction_in_cone, sample_mie_phase, sample_rayleigh_phase, sample_uniform_cone,
};
use crate::spectrum::BAND_COUNT;

type PixelSpectrum = [f32; BAND_COUNT];

pub fn render(scene: &SceneData, config: &RenderConfig) -> Film {
    assert_eq!(scene.bands.len(), BAND_COUNT);
    if can_use_azimuth_symmetry(config) {
        render_with_azimuth_symmetry(scene, config)
    } else {
        render_full(scene, config)
    }
}

fn render_full(scene: &SceneData, config: &RenderConfig) -> Film {
    let pixels: Vec<PixelSpectrum> = (0..config.width * config.height)
        .into_par_iter()
        .map(|pixel| {
            let x = pixel % config.width;
            let y = pixel / config.width;
            render_pixel(scene, config, x, y)
        })
        .collect();

    let mut film = Film::new(config.width, config.height, scene.bands.len());
    for (pixel, spectrum) in pixels.into_iter().enumerate() {
        film.set_pixel_spectrum(pixel, &spectrum);
    }
    film
}

fn render_with_azimuth_symmetry(scene: &SceneData, config: &RenderConfig) -> Film {
    let representative_width = config.width.div_ceil(2);

    let pixels: Vec<(usize, usize, PixelSpectrum)> = (0..representative_width * config.height)
        .into_par_iter()
        .map(|representative| {
            let x = representative % representative_width;
            let y = representative / representative_width;
            let mirror_x = mirror_azimuth_x(config.width, x);
            let pixel = y * config.width + x;
            let mirror_pixel = y * config.width + mirror_x;
            (pixel, mirror_pixel, render_pixel(scene, config, x, y))
        })
        .collect();

    let mut film = Film::new(config.width, config.height, scene.bands.len());
    for (pixel, mirror_pixel, spectrum) in pixels {
        film.set_pixel_spectrum(pixel, &spectrum);
        if mirror_pixel != pixel {
            film.set_pixel_spectrum(mirror_pixel, &spectrum);
        }
    }
    film
}

fn render_pixel(scene: &SceneData, config: &RenderConfig, x: usize, y: usize) -> PixelSpectrum {
    let mut sum = [0.0; BAND_COUNT];
    let pixel_seed = config.seed
        ^ ((x as u64).wrapping_mul(0x9E37_79B9))
        ^ ((y as u64).wrapping_mul(0xD1B5_4A32));
    let mut rng = SamplerState::new(pixel_seed);

    for sample in 0..config.spp {
        let u = (x as f32 + rng.next_f32()) / config.width as f32;
        let v = (y as f32 + rng.next_f32()) / config.height as f32;
        let ray = camera_ray(scene, config, u, v);
        for (band, value) in sum.iter_mut().enumerate() {
            let mut band_rng = rng.fork((sample as u64) << 32 ^ band as u64);
            *value += trace_band(scene, config, ray, band, &mut band_rng);
        }
    }

    let inv_spp = 1.0 / config.spp as f32;
    for value in &mut sum {
        *value = (*value * inv_spp).max(0.0);
    }
    sum
}

pub fn camera_ray(scene: &SceneData, config: &RenderConfig, u: f32, v: f32) -> Ray {
    let azimuth = (u - 0.5) * TAU;
    let elevation = (0.5 - v) * std::f32::consts::PI;
    let cos_e = elevation.cos();
    let dir = Vec3::new(
        cos_e * azimuth.sin(),
        elevation.sin(),
        cos_e * azimuth.cos(),
    );
    Ray::new(
        crate::geometry::observer_position(scene.planet, config.observer_altitude_km),
        dir.normalized(),
    )
}

pub fn can_use_azimuth_symmetry(config: &RenderConfig) -> bool {
    if !config.use_azimuth_symmetry {
        return false;
    }

    // The x <-> width-1-x mirror is exactly unbiased only when the sun lies in
    // the panorama's x=0 vertical plane. Other azimuths need a shifted mirror
    // mapping or full rendering to preserve the per-pixel integration domain.
    let azimuth_rad = config.sun_azimuth_deg.to_radians();
    azimuth_rad.sin().abs() < 1.0e-6
}

fn mirror_azimuth_x(width: usize, x: usize) -> usize {
    width - 1 - x
}

pub fn trace_band(
    scene: &SceneData,
    config: &RenderConfig,
    initial_ray: Ray,
    band_index: usize,
    rng: &mut SamplerState,
) -> f32 {
    let mut ray = initial_ray;
    let mut throughput = 1.0_f32;
    let mut radiance = 0.0_f32;
    let mut had_scatter = false;

    for depth in 0..config.max_depth {
        let Some(boundary) = next_boundary(ray, scene.planet) else {
            break;
        };

        let majorant = scene.majorants_km_inv[band_index];
        let sampled = sample_real_collision(scene, ray, boundary.t_km, band_index, majorant, rng);

        match sampled {
            Some((t, coeffs)) => {
                let pos = ray.at(t);
                let scattering = coeffs.scattering_total();
                let extinction = coeffs.extinction_total();
                if extinction <= 0.0 || scattering <= 0.0 {
                    break;
                }

                if rng.next_f32() > scattering / extinction {
                    break;
                }

                let mode = choose_scattering_mode(coeffs, rng);
                radiance +=
                    throughput * direct_sun_at_scatter(scene, pos, ray.dir, band_index, mode, rng);
                had_scatter = true;

                let (new_dir, pdf) =
                    sample_scattering_direction(scene, mode, ray.dir, band_index, rng);
                let mu = ray.dir.dot(new_dir).clamp(-1.0, 1.0);
                let phase = phase_value(scene, mode, band_index, mu);
                throughput *= phase / pdf;

                if depth > 3 {
                    let q = throughput.clamp(0.05, 0.95);
                    if rng.next_f32() > q {
                        break;
                    }
                    throughput /= q;
                }

                ray = Ray::new(pos + new_dir * RAY_EPSILON_KM, new_dir);
            }
            None => match boundary.kind {
                BoundaryKind::AtmosphereExit => {
                    if !had_scatter
                        && direction_in_cone(
                            ray.dir,
                            scene.sun.direction,
                            scene.sun.angular_radius_rad,
                        )
                    {
                        radiance += throughput * scene.solar_radiance_w_m2_sr(band_index);
                    }
                    break;
                }
                BoundaryKind::Ground => {
                    let pos = ray.at(boundary.t_km);
                    radiance += throughput * ground_radiance(scene, pos, band_index, rng);
                    break;
                }
            },
        }

        if !radiance.is_finite() || !throughput.is_finite() {
            return 0.0;
        }
    }

    radiance.max(0.0)
}

fn sample_real_collision(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    majorant: f32,
    rng: &mut SamplerState,
) -> Option<(f32, MediumCoefficients)> {
    let mut t = 0.0;
    while t < t_max {
        let u = (1.0 - rng.next_f32()).max(1.0e-7);
        t += -u.ln() / majorant;
        if t >= t_max {
            return None;
        }
        let coeffs = coefficients_at(scene, ray.at(t), band_index);
        let accept = (coeffs.extinction_total() / majorant).clamp(0.0, 1.0);
        if rng.next_f32() < accept {
            return Some((t, coeffs));
        }
    }
    None
}

pub fn estimate_transmittance_ratio(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    rng: &mut SamplerState,
) -> f32 {
    let majorant = scene.majorants_km_inv[band_index];
    let mut t = 0.0;
    let mut weight = 1.0_f32;
    while t < t_max {
        let u = (1.0 - rng.next_f32()).max(1.0e-7);
        t += -u.ln() / majorant;
        if t >= t_max {
            break;
        }
        let coeffs = coefficients_at(scene, ray.at(t), band_index);
        weight *= (1.0 - coeffs.extinction_total() / majorant).clamp(0.0, 1.0);
        if weight <= 0.0 {
            return 0.0;
        }
    }
    weight.clamp(0.0, 1.0)
}

fn direct_sun_at_scatter(
    scene: &SceneData,
    pos: Vec3,
    view_dir: Vec3,
    band_index: usize,
    mode: ScatteringMode,
    rng: &mut SamplerState,
) -> f32 {
    let (sun_dir, pdf) =
        sample_uniform_cone(scene.sun.direction, scene.sun.angular_radius_rad, rng);
    let Some(t_sun) = segment_to_sun_or_space(pos, sun_dir, scene.planet) else {
        return 0.0;
    };
    let trans = estimate_transmittance_ratio(
        scene,
        Ray::new(pos + sun_dir * RAY_EPSILON_KM, sun_dir),
        t_sun,
        band_index,
        rng,
    );
    let phase = phase_value(
        scene,
        mode,
        band_index,
        view_dir.dot(sun_dir).clamp(-1.0, 1.0),
    );
    scene.solar_radiance_w_m2_sr(band_index) * trans * phase / pdf
}

fn ground_radiance(scene: &SceneData, pos: Vec3, band_index: usize, rng: &mut SamplerState) -> f32 {
    let n = surface_normal(pos);
    let (sun_dir, pdf) =
        sample_uniform_cone(scene.sun.direction, scene.sun.angular_radius_rad, rng);
    let cos_sun = n.dot(sun_dir).max(0.0);
    if cos_sun <= 0.0 {
        return 0.0;
    }
    let Some(t_sun) = segment_to_sun_or_space(pos, sun_dir, scene.planet) else {
        return 0.0;
    };
    let trans = estimate_transmittance_ratio(
        scene,
        Ray::new(pos + sun_dir * RAY_EPSILON_KM, sun_dir),
        t_sun,
        band_index,
        rng,
    );
    GROUND_ALBEDO * INV_PI * scene.solar_radiance_w_m2_sr(band_index) * trans * cos_sun / pdf
}

fn choose_scattering_mode(coeffs: MediumCoefficients, rng: &mut SamplerState) -> ScatteringMode {
    let scattering = coeffs.scattering_total();
    let mut xi = rng.next_f32() * scattering;
    if xi < coeffs.rayleigh_scattering_km_inv {
        return ScatteringMode::Rayleigh;
    }
    xi -= coeffs.rayleigh_scattering_km_inv;
    for species in 0..SPECIES_COUNT {
        if xi < coeffs.aerosol_scattering_km_inv[species] {
            return ScatteringMode::Aerosol {
                species_index: species,
            };
        }
        xi -= coeffs.aerosol_scattering_km_inv[species];
    }
    ScatteringMode::Rayleigh
}

fn sample_scattering_direction(
    scene: &SceneData,
    mode: ScatteringMode,
    axis: Vec3,
    band_index: usize,
    rng: &mut SamplerState,
) -> (Vec3, f32) {
    match mode {
        ScatteringMode::Rayleigh => sample_rayleigh_phase(axis, rng),
        ScatteringMode::Aerosol { species_index } => {
            sample_mie_phase(axis, &scene.phase_table, species_index, band_index, rng)
        }
    }
}

fn phase_value(scene: &SceneData, mode: ScatteringMode, band_index: usize, mu: f32) -> f32 {
    match mode {
        ScatteringMode::Rayleigh => rayleigh_phase(mu),
        ScatteringMode::Aerosol { .. } => scene.phase_table.eval_scalar(PhaseFrame {
            mu,
            band_index,
            mode,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azimuth_symmetry_is_enabled_only_for_exact_mirror_plane() {
        assert!(can_use_azimuth_symmetry(&RenderConfig {
            sun_azimuth_deg: 0.0,
            ..RenderConfig::default()
        }));
        assert!(can_use_azimuth_symmetry(&RenderConfig {
            sun_azimuth_deg: 180.0,
            ..RenderConfig::default()
        }));
        assert!(!can_use_azimuth_symmetry(&RenderConfig {
            sun_azimuth_deg: 90.0,
            ..RenderConfig::default()
        }));
        assert!(!can_use_azimuth_symmetry(&RenderConfig {
            sun_azimuth_deg: 0.0,
            use_azimuth_symmetry: false,
            ..RenderConfig::default()
        }));
    }
}
