use rayon::prelude::*;

use crate::atmosphere::{GROUND_ALBEDO, SPECIES_COUNT, SceneData};
use crate::config::{
    CollisionEstimator, RenderConfig, SpectralCorrelation, TransmittanceEstimator,
};
use crate::film::Film;
use crate::geometry::{
    BoundaryKind, RAY_EPSILON_KM, altitude_km, intersect_sphere, next_boundary,
    segment_to_sun_or_space, surface_normal,
};
use crate::math::{INV_PI, Ray, TAU, Vec3};
use crate::medium::{MediumCoefficients, coefficients_at};
use crate::phase::{PhaseFrame, ScalarPhase, ScatteringMode, rayleigh_phase};
use crate::sampling::{
    SamplerState, direction_in_cone, sample_mie_phase, sample_rayleigh_phase, sample_uniform_cone,
};
use crate::spectrum::BAND_COUNT;

type PixelSpectrum = [f32; BAND_COUNT];

const SCATTER_LIGHT_STREAM: u64 = 0x51C4_77E2_0D1A_1137;
const GROUND_LIGHT_STREAM: u64 = 0x9A6C_63D5_3B8F_4A11;

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

    for sample in 0..config.spp {
        let mut rng =
            SamplerState::for_sample(config.seed, sample as u64, pixel_seed, config.sampler);
        let u = (x as f32 + rng.next_f32()) / config.width as f32;
        let v = (y as f32 + rng.next_f32()) / config.height as f32;
        let ray = camera_ray(scene, config, u, v);
        for (band, value) in sum.iter_mut().enumerate() {
            let mut band_rng = rng.fork(band_stream(config.spectral_correlation, sample, band));
            *value += trace_band(scene, config, ray, band, &mut band_rng);
        }
    }

    let inv_spp = 1.0 / config.spp as f32;
    for value in &mut sum {
        *value = (*value * inv_spp).max(0.0);
    }
    sum
}

fn band_stream(correlation: SpectralCorrelation, sample: usize, band: usize) -> u64 {
    match correlation {
        SpectralCorrelation::Common => (sample as u64) << 32,
        SpectralCorrelation::Independent => ((sample as u64) << 32) ^ band as u64,
    }
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
    let mut last_phase_pdf: Option<f32> = None;

    for depth in 0..config.max_depth {
        let Some(boundary) = next_boundary(ray, scene.planet) else {
            break;
        };

        let sampled = sample_real_collision(scene, ray, boundary.t_km, band_index, rng);

        match sampled {
            Some((t, coeffs)) => {
                let pos = ray.at(t);
                let scattering = coeffs.scattering_total();
                let extinction = coeffs.extinction_total();
                if extinction <= 0.0 || scattering <= 0.0 {
                    break;
                }

                let albedo = (scattering / extinction).clamp(0.0, 1.0);
                if !survive_collision(config.collision_estimator, albedo, &mut throughput, rng) {
                    break;
                }

                let mode = choose_scattering_mode(coeffs, rng);
                let mut light_rng = rng.fork(depth_stream(SCATTER_LIGHT_STREAM, depth));
                radiance += throughput
                    * direct_sun_at_scatter(
                        scene,
                        config,
                        pos,
                        ray.dir,
                        band_index,
                        coeffs,
                        &mut light_rng,
                    );

                let (new_dir, pdf) =
                    sample_scattering_direction(scene, mode, ray.dir, band_index, rng);
                let mu = ray.dir.dot(new_dir).clamp(-1.0, 1.0);
                let phase = phase_value(scene, mode, band_index, mu);
                throughput *= phase / pdf;
                last_phase_pdf = Some(pdf);

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
                        let weight = last_phase_pdf
                            .map(|phase_pdf| balance_heuristic(phase_pdf, sun_light_pdf(scene)))
                            .unwrap_or(1.0);
                        radiance += throughput * scene.solar_radiance_w_m2_sr(band_index) * weight;
                    }
                    break;
                }
                BoundaryKind::Ground => {
                    let pos = ray.at(boundary.t_km);
                    let mut light_rng = rng.fork(depth_stream(GROUND_LIGHT_STREAM, depth));
                    radiance += throughput
                        * ground_radiance(scene, config, pos, band_index, &mut light_rng);
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

fn depth_stream(tag: u64, depth: usize) -> u64 {
    tag ^ (depth as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn survive_collision(
    estimator: CollisionEstimator,
    albedo: f32,
    throughput: &mut f32,
    rng: &mut SamplerState,
) -> bool {
    match estimator {
        CollisionEstimator::Analog => rng.next_f32() <= albedo,
        CollisionEstimator::Weighted => {
            *throughput *= albedo;
            *throughput > 0.0
        }
    }
}

fn sample_real_collision(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    rng: &mut SamplerState,
) -> Option<(f32, MediumCoefficients)> {
    let mut t = 0.0;
    while t < t_max {
        let segment_end = next_majorant_segment_end(scene, ray, t, t_max);
        let majorant = segment_majorant(scene, ray, t, segment_end, band_index);
        while t < segment_end {
            let u = (1.0 - rng.next_f32()).max(1.0e-7);
            t += -u.ln() / majorant;
            if t >= segment_end {
                break;
            }
            let coeffs = coefficients_at(scene, ray.at(t), band_index);
            debug_assert!(
                coeffs.extinction_total() <= majorant * 1.001,
                "layer majorant underestimates extinction"
            );
            let accept = (coeffs.extinction_total() / majorant).clamp(0.0, 1.0);
            if rng.next_f32() < accept {
                return Some((t, coeffs));
            }
        }
        t = segment_end;
    }
    None
}

pub fn estimate_transmittance(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    rng: &mut SamplerState,
    estimator: TransmittanceEstimator,
) -> f32 {
    match estimator {
        TransmittanceEstimator::Ratio => {
            estimate_transmittance_ratio(scene, ray, t_max, band_index, rng)
        }
        TransmittanceEstimator::ResidualRatio => {
            estimate_transmittance_residual_ratio(scene, ray, t_max, band_index, rng)
        }
    }
}

pub fn estimate_transmittance_ratio(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    rng: &mut SamplerState,
) -> f32 {
    let mut t = 0.0;
    let mut weight = 1.0_f32;
    while t < t_max {
        let segment_end = next_majorant_segment_end(scene, ray, t, t_max);
        let majorant = segment_majorant(scene, ray, t, segment_end, band_index);
        while t < segment_end {
            let u = (1.0 - rng.next_f32()).max(1.0e-7);
            t += -u.ln() / majorant;
            if t >= segment_end {
                break;
            }
            let coeffs = coefficients_at(scene, ray.at(t), band_index);
            debug_assert!(
                coeffs.extinction_total() <= majorant * 1.001,
                "layer majorant underestimates extinction"
            );
            weight *= (1.0 - coeffs.extinction_total() / majorant).clamp(0.0, 1.0);
            if weight <= 0.0 {
                return 0.0;
            }
        }
        t = segment_end;
    }
    weight.clamp(0.0, 1.0)
}

pub fn estimate_transmittance_residual_ratio(
    scene: &SceneData,
    ray: Ray,
    t_max: f32,
    band_index: usize,
    rng: &mut SamplerState,
) -> f32 {
    let mut t = 0.0;
    let mut weight = 1.0_f32;
    while t < t_max {
        let segment_start = t;
        let segment_end = next_majorant_segment_end(scene, ray, t, t_max);
        let (minorant, majorant) =
            segment_extinction_bounds(scene, ray, t, segment_end, band_index);
        let control = minorant.min(majorant);
        weight *= (-control * (segment_end - segment_start)).exp();

        let residual_majorant = (majorant - control).max(0.0);
        if residual_majorant > 1.0e-8 {
            while t < segment_end {
                let u = (1.0 - rng.next_f32()).max(1.0e-7);
                t += -u.ln() / residual_majorant;
                if t >= segment_end {
                    break;
                }
                let coeffs = coefficients_at(scene, ray.at(t), band_index);
                let extinction = coeffs.extinction_total();
                debug_assert!(
                    extinction <= majorant * 1.001,
                    "layer majorant underestimates extinction"
                );
                debug_assert!(
                    extinction + 1.0e-7 >= control,
                    "layer minorant overestimates extinction"
                );
                let residual_extinction = (extinction - control).clamp(0.0, residual_majorant);
                weight *= (1.0 - residual_extinction / residual_majorant).clamp(0.0, 1.0);
                if weight <= 0.0 {
                    return 0.0;
                }
            }
        }
        t = segment_end;
    }
    weight.clamp(0.0, 1.0)
}

fn segment_majorant(scene: &SceneData, ray: Ray, t0: f32, t1: f32, band_index: usize) -> f32 {
    segment_extinction_bounds(scene, ray, t0, t1, band_index).1
}

fn segment_extinction_bounds(
    scene: &SceneData,
    ray: Ray,
    t0: f32,
    t1: f32,
    band_index: usize,
) -> (f32, f32) {
    let (min_altitude, max_altitude) = segment_altitude_range(scene, ray, t0, t1);
    let min_layer = scene.majorant_grid.layer_for_altitude(min_altitude);
    let max_layer = scene.majorant_grid.layer_for_altitude(max_altitude);
    let minorant = (min_layer..=max_layer)
        .map(|layer| scene.majorant_grid.minorant(band_index, layer))
        .fold(f32::INFINITY, f32::min)
        .max(0.0);
    let majorant = (min_layer..=max_layer)
        .map(|layer| scene.majorant_grid.get(band_index, layer))
        .fold(0.0, f32::max)
        .max(1.0e-8);
    (minorant.min(majorant), majorant)
}

fn next_majorant_segment_end(scene: &SceneData, ray: Ray, t: f32, t_max: f32) -> f32 {
    let probe_t = (t + RAY_EPSILON_KM).min(t_max);
    let altitude = altitude_km(scene.planet, ray.at(probe_t));
    let layer = scene.majorant_grid.layer_for_altitude(altitude);
    let (lo, hi) = scene.majorant_grid.layer_bounds_km(layer);
    let mut next_t = t_max;

    for altitude_boundary in [lo, hi] {
        if altitude_boundary <= 0.0 || altitude_boundary >= scene.majorant_grid.top_altitude_km {
            continue;
        }
        let radius = scene.planet.ground_radius_km + altitude_boundary;
        if let Some((a, b)) = intersect_sphere(ray, radius) {
            for root in [a, b] {
                if root > t + RAY_EPSILON_KM && root < next_t {
                    next_t = root;
                }
            }
        }
    }

    next_t
}

fn segment_altitude_range(scene: &SceneData, ray: Ray, t0: f32, t1: f32) -> (f32, f32) {
    let mut min_radius = ray.at(t0).length();
    let mut max_radius = min_radius;
    for t in [t1, closest_approach_t(ray).clamp(t0, t1)] {
        let radius = ray.at(t).length();
        min_radius = min_radius.min(radius);
        max_radius = max_radius.max(radius);
    }
    (
        min_radius - scene.planet.ground_radius_km,
        max_radius - scene.planet.ground_radius_km,
    )
}

fn closest_approach_t(ray: Ray) -> f32 {
    -ray.origin.dot(ray.dir)
}

fn direct_sun_at_scatter(
    scene: &SceneData,
    config: &RenderConfig,
    pos: Vec3,
    view_dir: Vec3,
    band_index: usize,
    coeffs: MediumCoefficients,
    rng: &mut SamplerState,
) -> f32 {
    let (sun_dir, pdf) =
        sample_uniform_cone(scene.sun.direction, scene.sun.angular_radius_rad, rng);
    let Some(t_sun) = segment_to_sun_or_space(pos, sun_dir, scene.planet) else {
        return 0.0;
    };
    let trans = estimate_transmittance(
        scene,
        Ray::new(pos + sun_dir * RAY_EPSILON_KM, sun_dir),
        t_sun,
        band_index,
        rng,
        config.transmittance_estimator,
    );
    let mu = view_dir.dot(sun_dir).clamp(-1.0, 1.0);
    let phase = direct_scattering_phase(scene, coeffs, band_index, mu, pdf);
    scene.solar_radiance_w_m2_sr(band_index) * trans * phase / pdf
}

fn direct_scattering_phase(
    scene: &SceneData,
    coeffs: MediumCoefficients,
    band_index: usize,
    mu: f32,
    light_pdf: f32,
) -> f32 {
    let scattering = coeffs.scattering_total();
    if scattering <= 0.0 {
        return 0.0;
    }

    let mut phase = 0.0;
    let rayleigh_weight = coeffs.rayleigh_scattering_km_inv / scattering;
    if rayleigh_weight > 0.0 {
        let rayleigh = rayleigh_phase(mu);
        phase += rayleigh_weight * rayleigh * balance_heuristic(light_pdf, rayleigh);
    }

    for species_index in 0..SPECIES_COUNT {
        let species_weight = coeffs.aerosol_scattering_km_inv[species_index] / scattering;
        if species_weight <= 0.0 {
            continue;
        }
        let mode = ScatteringMode::Aerosol { species_index };
        let component_phase = phase_value(scene, mode, band_index, mu);
        let component_pdf = phase_sampling_pdf(scene, mode, band_index, mu);
        phase += species_weight * component_phase * balance_heuristic(light_pdf, component_pdf);
    }

    phase
}

fn ground_radiance(
    scene: &SceneData,
    config: &RenderConfig,
    pos: Vec3,
    band_index: usize,
    rng: &mut SamplerState,
) -> f32 {
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
    let trans = estimate_transmittance(
        scene,
        Ray::new(pos + sun_dir * RAY_EPSILON_KM, sun_dir),
        t_sun,
        band_index,
        rng,
        config.transmittance_estimator,
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

fn phase_sampling_pdf(scene: &SceneData, mode: ScatteringMode, band_index: usize, mu: f32) -> f32 {
    match mode {
        ScatteringMode::Rayleigh => rayleigh_phase(mu),
        ScatteringMode::Aerosol { species_index } => {
            scene
                .phase_table
                .sampling_pdf(species_index, band_index, mu)
        }
    }
}

fn sun_light_pdf(scene: &SceneData) -> f32 {
    1.0 / scene.sun.solid_angle_sr
}

fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let denom = pdf_a + pdf_b;
    if denom > 0.0 { pdf_a / denom } else { 0.0 }
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

    #[test]
    fn balance_heuristic_splits_equal_pdfs() {
        assert!((balance_heuristic(2.0, 2.0) - 0.5).abs() < 1.0e-6);
        assert_eq!(balance_heuristic(0.0, 2.0), 0.0);
    }

    #[test]
    fn weighted_collision_uses_albedo_as_throughput() {
        let mut rng = SamplerState::new(3);
        let mut throughput = 2.0;
        assert!(survive_collision(
            CollisionEstimator::Weighted,
            0.25,
            &mut throughput,
            &mut rng
        ));
        assert!((throughput - 0.5).abs() < 1.0e-6);

        assert!(!survive_collision(
            CollisionEstimator::Analog,
            0.0,
            &mut throughput,
            &mut rng
        ));
    }

    #[test]
    fn spectral_band_streams_can_share_random_numbers() {
        assert_eq!(
            band_stream(SpectralCorrelation::Common, 7, 0),
            band_stream(SpectralCorrelation::Common, 7, 3)
        );
        assert_ne!(
            band_stream(SpectralCorrelation::Independent, 7, 0),
            band_stream(SpectralCorrelation::Independent, 7, 3)
        );
    }

    #[test]
    fn depth_streams_are_stable_and_distinct() {
        assert_eq!(
            depth_stream(SCATTER_LIGHT_STREAM, 2),
            depth_stream(SCATTER_LIGHT_STREAM, 2)
        );
        assert_ne!(
            depth_stream(SCATTER_LIGHT_STREAM, 2),
            depth_stream(SCATTER_LIGHT_STREAM, 3)
        );
        assert_ne!(
            depth_stream(SCATTER_LIGHT_STREAM, 2),
            depth_stream(GROUND_LIGHT_STREAM, 2)
        );
    }

    #[test]
    fn direct_scattering_phase_sums_all_components() {
        let scene =
            crate::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0).expect("scene");
        let mut coeffs = MediumCoefficients {
            rayleigh_scattering_km_inv: 1.0,
            ..MediumCoefficients::default()
        };
        coeffs.aerosol_scattering_km_inv[0] = 3.0;

        let mu = 0.35;
        let light_pdf = 5.0;
        let rayleigh = rayleigh_phase(mu);
        let aerosol_mode = ScatteringMode::Aerosol { species_index: 0 };
        let aerosol = phase_value(&scene, aerosol_mode, 0, mu);
        let aerosol_pdf = phase_sampling_pdf(&scene, aerosol_mode, 0, mu);
        let expected = 0.25 * rayleigh * balance_heuristic(light_pdf, rayleigh)
            + 0.75 * aerosol * balance_heuristic(light_pdf, aerosol_pdf);

        let actual = direct_scattering_phase(&scene, coeffs, 0, mu, light_pdf);
        assert!((actual - expected).abs() < 1.0e-6);
    }
}
