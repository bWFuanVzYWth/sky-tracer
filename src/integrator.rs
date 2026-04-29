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
use crate::sampling::{SamplerState, direction_in_cone, sample_isotropic, sample_uniform_cone};

pub fn render(scene: &SceneData, config: &RenderConfig) -> Film {
    let pixels: Vec<Vec<f32>> = (0..config.width * config.height)
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

fn render_pixel(scene: &SceneData, config: &RenderConfig, x: usize, y: usize) -> Vec<f32> {
    let mut sum = vec![0.0; scene.bands.len()];
    let pixel_seed = config.seed
        ^ ((x as u64).wrapping_mul(0x9E37_79B9))
        ^ ((y as u64).wrapping_mul(0xD1B5_4A32));
    let mut rng = SamplerState::new(pixel_seed);

    for sample in 0..config.spp {
        let u = (x as f32 + rng.next_f32()) / config.width as f32;
        let v = (y as f32 + rng.next_f32()) / config.height as f32;
        let ray = camera_ray(scene, u, v);
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

pub fn camera_ray(scene: &SceneData, u: f32, v: f32) -> Ray {
    let azimuth = (u - 0.5) * TAU;
    let elevation = (0.5 - v) * std::f32::consts::PI;
    let cos_e = elevation.cos();
    let dir = Vec3::new(
        cos_e * azimuth.sin(),
        elevation.sin(),
        cos_e * azimuth.cos(),
    );
    Ray::new(
        crate::geometry::observer_position(scene.planet),
        dir.normalized(),
    )
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

                let (new_dir, pdf) = sample_isotropic(rng);
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
                    if direction_in_cone(ray.dir, scene.sun.direction, scene.sun.angular_radius_rad)
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
    let cos_sun = n.dot(scene.sun.direction).max(0.0);
    if cos_sun <= 0.0 {
        return 0.0;
    }
    let Some(t_sun) = segment_to_sun_or_space(pos, scene.sun.direction, scene.planet) else {
        return 0.0;
    };
    let trans = estimate_transmittance_ratio(
        scene,
        Ray::new(
            pos + scene.sun.direction * RAY_EPSILON_KM,
            scene.sun.direction,
        ),
        t_sun,
        band_index,
        rng,
    );
    GROUND_ALBEDO * INV_PI * scene.bands[band_index].solar_irradiance_w_m2 * trans * cos_sun
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
