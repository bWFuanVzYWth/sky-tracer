//! Integration studies on an explicitly small, fixed teacher grid, with
//! isolated parameter changes and an explicitly named combined candidate.
//! A denser numerical solve is a comparison, not a physical truth or an
//! estimate of the full teacher's interpolation error or bake duration.
use crate::{
    Result,
    asset::{BandLut, fingerprint},
    config::{BakeConfig, MAX_TEACHER_BYTES, RayStepMapping},
    mapping::{Geometry, scattering_cosine, unit},
    model::Model,
    reference_mapping::realtime_fit_heights,
    solver::{BakedBand, GpuBaker, OrderStats},
};
use glam::Vec3;
use serde::Serialize;
use std::{fs, path::Path, time::Instant};

#[derive(Clone, Debug, Serialize)]
pub struct StudyOptions {
    pub scattering: [usize; 4],
    /// Auxiliary optical-depth axes; independent of the scattering study grid.
    pub optical_depth: [usize; 2],
    /// Select the model band whose center is nearest each requested wavelength.
    pub bands_nm: Vec<f32>,
    /// Empty runs all variants. The dense comparison always runs first.
    pub only: Vec<String>,
    pub repeats: usize,
    /// One excluded solve of the first selected band primes each configuration.
    pub warmup: bool,
}
impl Default for StudyOptions {
    fn default() -> Self {
        Self {
            scattering: [8, 12, 25, 33],
            optical_depth: [64, 512],
            bands_nm: vec![450.0, 550.0, 650.0],
            only: Vec::new(),
            repeats: 1,
            warmup: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Query {
    pub altitude_km: f32,
    pub sun_elevation_deg: f32,
    pub view_elevation_deg: f32,
    pub relative_azimuth_deg: f32,
    pub region: String,
    /// These targeted probes have equal diagnostic weight, not solid-angle weights.
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Quantiles {
    pub count: usize,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorMetrics {
    pub relative: Quantiles,
    pub absolute: Quantiles,
    pub weighted_mean_absolute: f64,
    pub weighted_relative_l1: f64,
    pub solar_normalized_mean_absolute: f64,
    pub dark_count: usize,
    pub dark_absolute: Option<Quantiles>,
    pub relative_radiance_floor: f64,
    pub worst_relative_query_index: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct WallStages {
    pub ground: f64,
    pub density: f64,
    pub expand_density: f64,
    pub integrate: f64,
    pub accumulate: f64,
    pub diagnostics_and_readback: f64,
    /// Residual includes setup, coordinate cache, optical-depth work and final
    /// payload readback. It cannot isolate the optical-depth dispatch duration.
    pub setup_tau_and_final_readback: f64,
}
impl WallStages {
    fn from_orders(total: f64, orders: &[OrderStats]) -> Self {
        let mut stages = Self::default();
        for order in orders {
            if let Some(s) = &order.stage_wall_seconds {
                stages.ground += f64::from(s.ground);
                stages.density += f64::from(s.density);
                stages.expand_density += f64::from(s.expand_density);
                stages.integrate += f64::from(s.integrate);
                stages.accumulate += f64::from(s.accumulate);
                stages.diagnostics_and_readback += f64::from(s.diagnostics_and_readback);
            }
        }
        let order_wall: f64 = orders.iter().map(|s| f64::from(s.elapsed_seconds)).sum();
        stages.setup_tau_and_final_readback = (total - order_wall).max(0.0);
        stages
    }
    fn divided_by(&self, value: f64) -> Self {
        let value = value.max(f64::MIN_POSITIVE);
        Self {
            ground: self.ground / value,
            density: self.density / value,
            expand_density: self.expand_density / value,
            integrate: self.integrate / value,
            accumulate: self.accumulate / value,
            diagnostics_and_readback: self.diagnostics_and_readback / value,
            setup_tau_and_final_readback: self.setup_tau_and_final_readback / value,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BandRun {
    pub band_index: usize,
    pub center_nm: f32,
    pub solar_irradiance_w_m2: f32,
    pub repeat: usize,
    pub first_measured_call_for_configuration: bool,
    pub coordinate_cache_expected_primed: bool,
    pub total_bake_wall_seconds: f64,
    pub cpu_query_wall_seconds: f64,
    pub stages_wall_seconds: WallStages,
    pub stages_fraction_of_total_wall: WallStages,
    pub first_order: Option<usize>,
    pub last_order: Option<usize>,
    pub orders_completed: usize,
    pub stopped_by_tolerance: bool,
    pub orders: Vec<OrderStats>,
    pub radiance: Vec<f32>,
    pub difference_from_dense_control: ErrorMetrics,
    pub wall_ratio_to_baseline_same_grid_band_mean: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StudyCase {
    pub name: String,
    pub changed_parameter: String,
    pub config: BakeConfig,
    pub warmup_band_index: Option<usize>,
    pub warmup_wall_seconds: Option<f64>,
    pub runs: Vec<BandRun>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StudyReport {
    pub schema: String,
    pub complete: bool,
    pub adapter: String,
    pub model_fingerprint_fnv1a64: String,
    pub options: StudyOptions,
    pub selected_band_indices: Vec<usize>,
    pub query_count: usize,
    pub queries: Vec<Query>,
    pub planned_case_order: Vec<String>,
    pub gpu_timestamps_measured: bool,
    pub solver_initialization_wall_seconds: f64,
    pub full_teacher_scattering: [usize; 4],
    pub full_teacher_bytes_per_band_upper_bound: u64,
    pub limitations: Vec<String>,
    pub cases: Vec<StudyCase>,
}

/// Names accepted by --only, in their execution order after the dense solve.
pub fn variant_names() -> &'static [&'static str] {
    &[
        "dense_control",
        "baseline",
        "ray_128",
        "ray_384",
        "ray_uniform_768",
        "ray_logheight_128",
        "ray_logheight_192",
        "ray_logheight_256",
        "tau_1024",
        "angular_12x24",
        "angular_20x40",
        "balanced_logheight_192",
        "sun_2x16",
        "sun_4x8",
        "sun_4x32",
        "orders_16",
    ]
}

fn cases(g: Geometry, options: &StudyOptions, band_count: usize) -> Result<Vec<StudyCase>> {
    if !(1..=32).contains(&options.repeats) {
        return Err("sampling-study repeats must be in 1..=32".into());
    }
    for name in &options.only {
        if !variant_names().contains(&name.as_str()) {
            return Err(format!(
                "unknown sampling-study variant {name}; expected {}",
                variant_names().join(", ")
            )
            .into());
        }
    }
    let mut baseline = BakeConfig::current_reference();
    baseline.scattering = options.scattering;
    baseline.optical_depth = options.optical_depth;
    // Reject bad dimensions before height resampling can divide by zero.
    if baseline
        .scattering
        .iter()
        .chain(&baseline.optical_depth)
        .any(|&n| n < 2)
    {
        return Err("sampling-study scattering and optical-depth axes must be >= 2".into());
    }
    if baseline.asset_bytes(band_count)? > MAX_TEACHER_BYTES {
        return Err("sampling-study dimensions exceed the teacher asset budget".into());
    }
    baseline.scattering_altitudes_km = realtime_fit_heights(g, baseline.scattering[0]);
    baseline.ray_steps = 256;
    baseline.ray_step_mapping = RayStepMapping::UniformDistance;
    baseline.optical_depth_steps = 2048;
    baseline.angular_mu = 16;
    baseline.angular_phi = 32;
    baseline.sun_mu = 4;
    baseline.sun_phi = 16;
    baseline.max_orders = 24;
    baseline.min_orders = 8;
    baseline.relative_order_tolerance = 1e-4;
    baseline.validate(band_count)?;
    baseline.validate_top_height(g.top_height())?;
    let mut result = Vec::new();
    for &name in variant_names() {
        if name != "dense_control"
            && !options.only.is_empty()
            && !options.only.iter().any(|v| v == name)
        {
            continue;
        }
        let mut config = baseline.clone();
        let parameter = match name {
            "dense_control" => {
                config.ray_steps = 768;
                config.optical_depth_steps = 4096;
                config.angular_mu = 24;
                config.angular_phi = 48;
                config.sun_phi = 32;
                config.max_orders = 32;
                "several integration counts: numerical comparison only"
            }
            "baseline" => "none",
            "ray_128" => {
                config.ray_steps = 128;
                "ray_steps"
            }
            "ray_384" => {
                config.ray_steps = 384;
                "ray_steps"
            }
            "tau_1024" => {
                config.optical_depth_steps = 1024;
                "optical_depth_steps"
            }
            "ray_uniform_768" => {
                config.ray_steps = 768;
                "ray_steps"
            }
            "ray_logheight_128" => {
                config.ray_steps = 128;
                config.ray_step_mapping = RayStepMapping::LogHeightV1;
                "ray step count and log-height allocation"
            }
            "ray_logheight_192" => {
                config.ray_steps = 192;
                config.ray_step_mapping = RayStepMapping::LogHeightV1;
                "ray step count and log-height allocation"
            }
            "ray_logheight_256" => {
                config.ray_step_mapping = RayStepMapping::LogHeightV1;
                "ray step log-height allocation"
            }
            "angular_12x24" => {
                config.angular_mu = 12;
                config.angular_phi = 24;
                "angular_quadrature"
            }
            "angular_20x40" => {
                config.angular_mu = 20;
                config.angular_phi = 40;
                "angular_quadrature"
            }
            "balanced_logheight_192" => {
                config.ray_steps = 192;
                config.ray_step_mapping = RayStepMapping::LogHeightV1;
                config.angular_mu = 20;
                config.angular_phi = 40;
                "combined log-height ray allocation and angular quadrature"
            }
            "sun_2x16" => {
                config.sun_mu = 2;
                "sun_quadrature"
            }
            "sun_4x8" => {
                config.sun_phi = 8;
                "sun_quadrature"
            }
            "sun_4x32" => {
                config.sun_phi = 32;
                "sun_quadrature"
            }
            "orders_16" => {
                config.max_orders = 16;
                "max_orders"
            }
            _ => unreachable!(),
        };
        config.validate(band_count)?;
        result.push(StudyCase {
            name: name.into(),
            changed_parameter: parameter.into(),
            config,
            warmup_band_index: None,
            warmup_wall_seconds: None,
            runs: Vec::new(),
        });
    }
    Ok(result)
}

fn selected_bands(model: &Model, requested: &[f32]) -> Result<Vec<usize>> {
    if model.bands.is_empty() || requested.is_empty() {
        return Err("sampling-study needs a model and at least one requested wavelength".into());
    }
    let mut result = Vec::new();
    for &wavelength in requested {
        if !wavelength.is_finite() || wavelength <= 0.0 {
            return Err("sampling-study wavelengths must be finite and positive".into());
        }
        let index = model
            .bands
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.info.center_nm - wavelength)
                    .abs()
                    .total_cmp(&(b.info.center_nm - wavelength).abs())
            })
            .unwrap()
            .0;
        if !result.contains(&index) {
            result.push(index);
        }
    }
    Ok(result)
}

fn queries(g: Geometry) -> Vec<Query> {
    let mut result = Vec::new();
    for h in [0.00025, 0.2, 2.0, 11.0, 12.0, 35.0, 100.0, 120.0, 400.0] {
        let horizon = g.horizon(h).asin().to_degrees();
        for sun in [-12.0, -6.0, -1.0, 0.0, 30.0, 60.0, 89.7] {
            for (region, view) in [
                ("horizon_ground", horizon - 0.01),
                ("horizon_sky", horizon + 0.01),
                ("low_sky", horizon + 0.2),
                ("zenith", 90.0),
                ("near_sun", sun - 0.1),
                ("near_sun", sun + 0.1),
            ] {
                for az in [0.0, 90.0, 180.0] {
                    result.push(Query {
                        altitude_km: h,
                        sun_elevation_deg: sun,
                        view_elevation_deg: view.clamp(-90.0, 90.0),
                        relative_azimuth_deg: az,
                        region: region.into(),
                        weight: 1.0,
                    });
                }
            }
        }
    }
    result
}

fn direction(elevation: f32, azimuth: f32) -> Vec3 {
    let (s, c) = elevation.to_radians().sin_cos();
    let (sa, ca) = azimuth.to_radians().sin_cos();
    Vec3::new(c * ca, c * sa, s)
}

fn radiance_samples(
    model: &Model,
    config: &BakeConfig,
    index: usize,
    band: BakedBand,
    queries: &[Query],
) -> Result<Vec<f32>> {
    let lut = BandLut {
        geometry: model.geometry,
        config: config.clone(),
        info: model.bands[index].info.clone(),
        sun_radius: model.sun_radius,
        optical_depth: band.optical_depth,
        radiance: band.radiance,
        ground_irradiance: band.ground_irradiance,
        scattering_cosines: (0..config.scattering[3])
            .map(|i| scattering_cosine(unit(i, config.scattering[3])))
            .collect(),
    };
    queries
        .iter()
        .map(|q| {
            lut.sample(
                q.altitude_km,
                direction(q.view_elevation_deg, q.relative_azimuth_deg),
                direction(q.sun_elevation_deg, 0.0),
                false,
            )
        })
        .collect()
}

fn quantiles(mut values: Vec<f64>) -> Quantiles {
    values.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        let x = p * (values.len() - 1) as f64;
        let lo = x.floor() as usize;
        let hi = x.ceil() as usize;
        values[lo] * (1.0 - (x - lo as f64)) + values[hi] * (x - lo as f64)
    };
    Quantiles {
        count: values.len(),
        p50: percentile(0.5),
        p95: percentile(0.95),
        p99: percentile(0.99),
        max: *values.last().unwrap(),
    }
}

fn metrics(values: &[f32], control: &[f32], queries: &[Query], solar: f32) -> Result<ErrorMetrics> {
    if values.len() != control.len()
        || values.len() != queries.len()
        || values.is_empty()
        || values
            .iter()
            .chain(control)
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid sampling-study comparison arrays".into());
    }
    let solar = f64::from(solar).max(f64::MIN_POSITIVE);
    let floor = (solar * 1e-8).max(1e-30);
    let mut absolute = Vec::with_capacity(values.len());
    let mut relative = Vec::with_capacity(values.len());
    let mut dark = Vec::new();
    let (mut sum_abs, mut sum_reference, mut sum_weight) = (0.0, 0.0, 0.0);
    let (mut worst, mut worst_error) = (0, -1.0);
    for (i, ((&value, &reference), query)) in values.iter().zip(control).zip(queries).enumerate() {
        let reference = f64::from(reference);
        let error = (f64::from(value) - reference).abs();
        let rel = error / reference.max(floor);
        absolute.push(error);
        relative.push(rel);
        if reference < floor {
            dark.push(error);
        }
        sum_abs += query.weight * error;
        sum_reference += query.weight * reference.max(floor);
        sum_weight += query.weight;
        if rel > worst_error {
            worst = i;
            worst_error = rel;
        }
    }
    let dark_count = dark.len();
    Ok(ErrorMetrics {
        relative: quantiles(relative),
        absolute: quantiles(absolute),
        weighted_mean_absolute: sum_abs / sum_weight,
        weighted_relative_l1: sum_abs / sum_reference,
        solar_normalized_mean_absolute: sum_abs / sum_weight / solar,
        dark_count,
        dark_absolute: (!dark.is_empty()).then(|| quantiles(dark)),
        relative_radiance_floor: floor,
        worst_relative_query_index: worst,
    })
}

fn save(report: &StudyReport, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let part = output.with_extension("json.part");
    fs::write(&part, serde_json::to_vec_pretty(report)?)?;
    fs::rename(part, output)?;
    Ok(())
}

fn update_ratios(report: &mut StudyReport) {
    let baseline: Vec<(usize, f64)> = report
        .cases
        .iter()
        .find(|c| c.name == "baseline")
        .map_or_else(Vec::new, |case| {
            report
                .selected_band_indices
                .iter()
                .filter_map(|&band| {
                    let times: Vec<_> = case
                        .runs
                        .iter()
                        .filter(|r| r.band_index == band)
                        .map(|r| r.total_bake_wall_seconds)
                        .collect();
                    (!times.is_empty())
                        .then(|| (band, times.iter().sum::<f64>() / times.len() as f64))
                })
                .collect()
        });
    for case in &mut report.cases {
        for run in &mut case.runs {
            run.wall_ratio_to_baseline_same_grid_band_mean = baseline
                .iter()
                .find(|(band, _)| *band == run.band_index)
                .map(|(_, wall)| run.total_bake_wall_seconds / wall.max(f64::MIN_POSITIVE));
        }
    }
}

/// Execute with one persistent solver and checkpoint JSON after each completed
/// band/repeat. Existing output is refused to preserve prior measurements.
pub fn run(
    model: &Model,
    options: StudyOptions,
    output: &Path,
    mut progress: impl FnMut(&str),
) -> Result<StudyReport> {
    if output.exists() {
        return Err("sampling-study output exists; choose a new JSON path".into());
    }
    let band_indices = selected_bands(model, &options.bands_nm)?;
    let planned = cases(model.geometry, &options, model.bands.len())?;
    let queries = queries(model.geometry);
    let started = Instant::now();
    let gpu = GpuBaker::new()?;
    let initialization = started.elapsed().as_secs_f64();
    let teacher = BakeConfig::current_reference();
    let mut report = StudyReport {
        schema: "teacher-integration-sampling-study-v1".into(), complete: false,
        adapter: gpu.adapter_name.clone(), model_fingerprint_fnv1a64: fingerprint(model)?,
        options: options.clone(), selected_band_indices: band_indices.clone(),
        query_count: queries.len(), queries, planned_case_order: planned.iter().map(|c| c.name.clone()).collect(),
        gpu_timestamps_measured: false, solver_initialization_wall_seconds: initialization,
        full_teacher_scattering: teacher.scattering, full_teacher_bytes_per_band_upper_bound: teacher.band_bytes()?,
        limitations: vec![
            "Dense control changes integration counts on the same coarse grid; it is a numerical comparison, not physical truth or a full-grid interpolation convergence test.".into(),
            "All durations are CPU wall times including submission, waits and readback; the residual setup/tau/final-readback time cannot isolate the optical-depth pass or coordinate cache cost.".into(),
            "Small-grid working sets fit caches. Neither these durations nor linear multiplication by full-grid state count establishes full-teacher bake duration.".into(),
            "Relative errors use max(control radiance, band solar irradiance * 1e-8); dark probes also report raw absolute error. Probe weights are equal diagnostic weights, not a sky integral.".into(),
            "Queries exclude the direct solar disk contribution; near-sun probes still test transport illuminated by the finite solar disk. Early termination can change the completed order count.".into(),
            "Case order is fixed with dense control first. Optional warmup is excluded from measured runs; first-use and warm-cache effects remain visible in per-run metadata.".into(),
        ], cases: Vec::new(),
    };
    let mut control: Vec<Option<Vec<f32>>> = vec![None; model.bands.len()];
    save(&report, output)?;
    for mut case in planned {
        progress(&format!(
            "case {}: {} selected bands, {} repeats, {} probes/band",
            case.name,
            band_indices.len(),
            options.repeats,
            report.queries.len()
        ));
        if options.warmup {
            let index = band_indices[0];
            progress(&format!(
                "{}: excluded warmup, band {} ({:.0} nm)",
                case.name, index, model.bands[index].info.center_nm
            ));
            let start = Instant::now();
            gpu.bake_band(model, &case.config, index, |message| progress(message))?;
            case.warmup_band_index = Some(index);
            case.warmup_wall_seconds = Some(start.elapsed().as_secs_f64());
        }
        report.cases.push(case);
        let case_index = report.cases.len() - 1;
        for (selected_index, &index) in band_indices.iter().enumerate() {
            for repeat in 0..options.repeats {
                let case = &report.cases[case_index];
                progress(&format!(
                    "{}: band {} ({:.0} nm), repeat {}/{}",
                    case.name,
                    index,
                    model.bands[index].info.center_nm,
                    repeat + 1,
                    options.repeats
                ));
                let start = Instant::now();
                let baked =
                    gpu.bake_band(model, &case.config, index, |message| progress(message))?;
                let total_wall = start.elapsed().as_secs_f64();
                let stages = WallStages::from_orders(total_wall, &baked.orders);
                let orders = baked.orders.clone();
                let stopped_by_tolerance = baked.stopped_by_tolerance;
                let start = Instant::now();
                let samples = radiance_samples(model, &case.config, index, baked, &report.queries)?;
                let sample_wall = start.elapsed().as_secs_f64();
                if case.name == "dense_control" && repeat == 0 {
                    control[index] = Some(samples.clone());
                }
                let reference = control[index]
                    .as_ref()
                    .ok_or("dense comparison band missing")?;
                let difference = metrics(
                    &samples,
                    reference,
                    &report.queries,
                    model.bands[index].info.solar_irradiance_w_m2,
                )?;
                progress(&format!(
                    "{}: {:.3}s, {} orders, relative P95 {:.4}, P99 {:.4}, max {:.4}",
                    case.name,
                    total_wall,
                    orders.len(),
                    difference.relative.p95,
                    difference.relative.p99,
                    difference.relative.max
                ));
                report.cases[case_index].runs.push(BandRun {
                    band_index: index,
                    center_nm: model.bands[index].info.center_nm,
                    solar_irradiance_w_m2: model.bands[index].info.solar_irradiance_w_m2,
                    repeat,
                    first_measured_call_for_configuration: selected_index == 0 && repeat == 0,
                    coordinate_cache_expected_primed: options.warmup
                        || selected_index > 0
                        || repeat > 0,
                    total_bake_wall_seconds: total_wall,
                    cpu_query_wall_seconds: sample_wall,
                    stages_fraction_of_total_wall: stages.divided_by(total_wall),
                    stages_wall_seconds: stages,
                    first_order: orders.first().map(|s| s.order),
                    last_order: orders.last().map(|s| s.order),
                    orders_completed: orders.len(),
                    stopped_by_tolerance,
                    orders,
                    radiance: samples,
                    difference_from_dense_control: difference,
                    wall_ratio_to_baseline_same_grid_band_mean: None,
                });
                update_ratios(&mut report);
                save(&report, output)?;
            }
        }
    }
    report.complete = true;
    save(&report, output)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_keep_the_teacher_grid_fixed_and_filter_without_losing_control() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        let options = StudyOptions {
            only: vec!["ray_128".into(), "sun_4x8".into()],
            ..Default::default()
        };
        let configs = cases(g, &options, 41).unwrap();
        assert_eq!(
            configs.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["dense_control", "ray_128", "sun_4x8"]
        );
        for c in &configs {
            assert_eq!(c.config.scattering, options.scattering);
            assert_eq!(c.config.optical_depth, [64, 512]);
            assert_eq!(
                c.config.scattering_altitudes_km,
                configs[0].config.scattering_altitudes_km
            );
        }
        assert_eq!(configs[1].config.ray_steps, 128);
        assert_eq!(configs[1].config.sun_phi, 16);
        assert_eq!(configs[2].config.ray_steps, 256);
        assert_eq!(configs[2].config.sun_phi, 8);
        assert!(
            cases(
                g,
                &StudyOptions {
                    only: vec!["typo".into()],
                    ..Default::default()
                },
                41
            )
            .is_err()
        );
    }

    #[test]
    fn comparison_floor_separates_dark_absolute_error_and_relative_error() {
        let q = vec![
            Query {
                altitude_km: 0.2,
                sun_elevation_deg: 0.0,
                view_elevation_deg: 1.0,
                relative_azimuth_deg: 0.0,
                region: "test".into(),
                weight: 1.0
            };
            3
        ];
        let result = metrics(&[1.1, 2e-8, 0.0], &[1.0, 0.0, 0.0], &q, 1.0).unwrap();
        assert_eq!(result.dark_count, 2);
        assert_eq!(
            result.dark_absolute.as_ref().unwrap().max,
            f64::from(2e-8f32)
        );
        assert!((result.relative.max - 2.0).abs() < 1e-6);
        assert!(metrics(&[f32::NAN], &[0.0], &q[..1], 1.0).is_err());
    }

    #[test]
    fn targeted_queries_cover_both_horizon_sides_space_and_near_sun() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        let probes = queries(g);
        assert_eq!(probes.len(), 1134);
        assert!(
            probes.iter().all(|q| q.view_elevation_deg.is_finite()
                && (-90.0..=90.0).contains(&q.view_elevation_deg))
        );
        for height in [0.00025, 0.2, 2.0, 11.0, 12.0, 35.0, 100.0, 120.0, 400.0] {
            for sun in [-12.0, -6.0, -1.0, 0.0, 30.0, 60.0, 89.7] {
                for az in [0.0, 90.0, 180.0] {
                    assert!(probes.iter().any(|q| q.altitude_km == height
                        && q.sun_elevation_deg == sun
                        && q.relative_azimuth_deg == az
                        && q.region == "near_sun"));
                }
            }
        }
        let ground = probes
            .iter()
            .find(|q| q.altitude_km == 0.2 && q.region == "horizon_ground")
            .unwrap();
        let sky = probes
            .iter()
            .find(|q| q.altitude_km == 0.2 && q.region == "horizon_sky")
            .unwrap();
        assert!(ground.view_elevation_deg < g.horizon(0.2).asin().to_degrees());
        assert!(sky.view_elevation_deg > g.horizon(0.2).asin().to_degrees());
    }

    #[test]
    fn ray_allocation_variants_leave_other_integration_counts_unchanged() {
        let g = Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        let options = StudyOptions {
            only: vec![
                "baseline".into(),
                "ray_uniform_768".into(),
                "ray_logheight_192".into(),
            ],
            ..Default::default()
        };
        let configs = cases(g, &options, 41).unwrap();
        assert_eq!(configs[0].config.ray_steps, 768);
        let baseline = &configs[1].config;
        for c in &configs[2..] {
            assert_eq!(c.config.optical_depth_steps, baseline.optical_depth_steps);
            assert_eq!(c.config.angular_mu, baseline.angular_mu);
            assert_eq!(c.config.angular_phi, baseline.angular_phi);
            assert_eq!(c.config.sun_mu, baseline.sun_mu);
            assert_eq!(c.config.sun_phi, baseline.sun_phi);
            assert_eq!(c.config.max_orders, baseline.max_orders);
        }
        assert_eq!(
            configs[2].config.ray_step_mapping,
            RayStepMapping::UniformDistance
        );
        assert_eq!(
            configs[3].config.ray_step_mapping,
            RayStepMapping::LogHeightV1
        );
        assert_eq!(configs[3].config.ray_steps, 192);
    }
}
