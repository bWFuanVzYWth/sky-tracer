//! CPU-only comparison of exported cloud radiance and sample variance EXRs.
use clap::Parser;
use cloud_pt::Result;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(
    about = "Compare cloud reference EXRs using joint estimated standard errors; never creates a GPU"
)]
struct Cli {
    reference: PathBuf,
    candidate: Option<PathBuf>,
    /// One-image analytic check: albedo one, no sun/ground, constant sky and zero variance.
    #[arg(long)]
    constant_environment: bool,
    /// Require bit-identical decoded radiance and variance (e.g. single versus batch).
    #[arg(long)]
    exact_f32: bool,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 5.0)]
    sigma_limit: f64,
}

#[derive(Default)]
struct Pixels {
    width: usize,
    height: usize,
    values: Vec<[f32; 3]>,
}
struct Reference {
    metadata: Value,
    mean: Pixels,
    variance: Pixels,
    samples: u64,
}

fn read_pixels(path: &Path) -> Result<Pixels> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |size, _| Pixels {
            width: size.width(),
            height: size.height(),
            values: vec![[0.0; 3]; size.width() * size.height()],
        },
        |pixels, position, (r, g, b, _): (f32, f32, f32, f32)| {
            pixels.values[position.y() * pixels.width + position.x()] = [r, g, b];
        },
    )?;
    Ok(image.layer_data.channel_data.pixels)
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| format!("missing reference metadata field {name}").into())
}

fn load(directory: &Path) -> Result<Reference> {
    let metadata: Value = serde_json::from_slice(&fs::read(directory.join("asset.json"))?)?;
    if metadata["kind"] != "cloud_path_trace_reference_v1"
        || metadata["diagnostics_passed"] != true
        || metadata["failed_paths_discarded"] != false
        || metadata["sample_variance_valid"] != true
        || field(&metadata, "hard_scattering_depth_limit")? != &Value::Null
        || field(&metadata, "transmittance_threshold")? != &Value::Null
    {
        return Err(format!(
            "{} is not a validated complete-path reference with valid sample variance",
            directory.display()
        )
        .into());
    }
    let samples = metadata["samples_per_pixel"]
        .as_u64()
        .filter(|n| *n > 1 && *n <= u64::from(u32::MAX))
        .ok_or("reference needs at least two samples per pixel")?;
    let radiance = metadata["reference_file"]
        .as_str()
        .ok_or("missing radiance file")?;
    let variance_file = metadata["variance_file"]
        .as_str()
        .ok_or("missing sample variance file")?;
    if radiance != "radiance.exr" || variance_file != "sample_variance.exr" {
        return Err("unsupported reference EXR names".into());
    }
    let mean = read_pixels(&directory.join(radiance))?;
    let variance = read_pixels(&directory.join(variance_file))?;
    if mean.width == 0
        || mean.height == 0
        || mean.width != variance.width
        || mean.height != variance.height
        || metadata["dimensions"] != json!([mean.width, mean.height])
        || metadata["scene"]["render"]["width"] != json!(mean.width)
        || metadata["scene"]["render"]["height"] != json!(mean.height)
    {
        return Err("EXR and metadata dimensions do not match".into());
    }
    if mean
        .values
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
        || variance
            .values
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(
            "nonfinite or negative reference radiance/sample variance; no pixels may be dropped"
                .into(),
        );
    }
    Ok(Reference {
        metadata,
        mean,
        variance,
        samples,
    })
}

fn validate_same_model(a: &Reference, b: &Reference) -> Result<()> {
    if a.mean.width != b.mean.width || a.mean.height != b.mean.height {
        return Err("reference dimensions differ".into());
    }
    for name in ["color_space", "density_reconstruction"] {
        if field(&a.metadata, name)? != field(&b.metadata, name)? {
            return Err(format!("reference {name} differs").into());
        }
    }
    let scene_a = field(&a.metadata, "scene")?;
    let scene_b = field(&b.metadata, "scene")?;
    for name in ["source", "source_bytes", "grid", "volume_stats", "camera"] {
        if field(scene_a, name)? != field(scene_b, name)? {
            return Err(format!("reference scene {name} differs").into());
        }
    }
    let mut optics_a = field(scene_a, "transport")?
        .as_object()
        .ok_or("transport must be an object")?
        .clone();
    let mut optics_b = field(scene_b, "transport")?
        .as_object()
        .ok_or("transport must be an object")?
        .clone();
    for name in [
        "extinction_scale",
        "scattering_albedo",
        "phase_g",
        "sun_direction",
        "sun_irradiance",
        "sky_radiance",
        "ground",
    ] {
        if !optics_a.contains_key(name) || !optics_b.contains_key(name) {
            return Err(format!("missing physical transport field {name}").into());
        }
    }
    for name in [
        "spatial_majorants",
        "shadow_roulette",
        "roulette_start",
        "event_limit",
    ] {
        optics_a.remove(name);
        optics_b.remove(name);
    }
    if optics_a != optics_b {
        return Err("reference physical transport settings differ".into());
    }
    Ok(())
}

fn joint_se(variance_a: f64, samples_a: u64, variance_b: f64, samples_b: u64) -> f64 {
    (variance_a / samples_a as f64 + variance_b / samples_b as f64).sqrt()
}

fn standardized_difference(difference: f64, se: f64) -> Option<f64> {
    if se > 0.0 {
        Some(difference / se)
    } else if difference == 0.0 {
        Some(0.0)
    } else {
        None
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

fn compare(a: &Reference, b: &Reference, sigma_limit: f64) -> Value {
    let len = a.mean.values.len();
    let mut components = Vec::with_capacity(len * 3);
    let mut abs_z = Vec::with_capacity(len * 3);
    let mut relative = Vec::with_capacity(len);
    let mut outside = 0;
    let mut outside_three = 0;
    let mut zero_variance_mismatches = 0;
    let mut sum_a = [0.0; 3];
    let mut sum_b = [0.0; 3];
    let mut sum_variance_a = [0.0; 3];
    let mut sum_variance_b = [0.0; 3];
    for pixel in 0..len {
        let mut difference_norm_squared = 0.0;
        let mut reference_norm_squared = 0.0;
        for channel in 0..3 {
            let mean_a = f64::from(a.mean.values[pixel][channel]);
            let mean_b = f64::from(b.mean.values[pixel][channel]);
            let variance_a = f64::from(a.variance.values[pixel][channel]);
            let variance_b = f64::from(b.variance.values[pixel][channel]);
            let difference = mean_b - mean_a;
            let se = joint_se(variance_a, a.samples, variance_b, b.samples);
            let z = standardized_difference(difference, se);
            if z.is_none() {
                zero_variance_mismatches += 1;
            }
            if z.is_none_or(|value| value.abs() > sigma_limit) {
                outside += 1;
            }
            if z.is_none_or(|value| value.abs() > 3.0) {
                outside_three += 1;
            }
            if let Some(z) = z {
                abs_z.push(z.abs());
            }
            components.push(json!({"pixel":[pixel%a.mean.width,pixel/a.mean.width],"channel":(["R","G","B"][channel]),
                "reference_mean":mean_a,"candidate_mean":mean_b,"difference":difference,"joint_estimated_se":se,
                "standardized_difference":z,"zero_variance_mismatch":z.is_none()}));
            difference_norm_squared += difference * difference;
            reference_norm_squared += mean_a * mean_a;
            sum_a[channel] += mean_a;
            sum_b[channel] += mean_b;
            sum_variance_a[channel] += variance_a;
            sum_variance_b[channel] += variance_b;
        }
        relative
            .push(100.0 * difference_norm_squared.sqrt() / reference_norm_squared.sqrt().max(1e-9));
    }
    relative.sort_by(f64::total_cmp);
    abs_z.sort_by(f64::total_cmp);
    components.sort_by(|a, b| {
        let a = a["standardized_difference"]
            .as_f64()
            .map(f64::abs)
            .unwrap_or(f64::INFINITY);
        let b = b["standardized_difference"]
            .as_f64()
            .map(f64::abs)
            .unwrap_or(f64::INFINITY);
        b.total_cmp(&a)
    });
    let image_mean = (0..3).map(|channel| {
        let difference = (sum_b[channel]-sum_a[channel])/len as f64;
        let se = joint_se(sum_variance_a[channel], a.samples, sum_variance_b[channel], b.samples)/len as f64;
        json!({"channel":(["R","G","B"][channel]),"reference_mean":sum_a[channel]/len as f64,
            "candidate_mean":sum_b[channel]/len as f64,"difference":difference,"joint_estimated_se":se,
            "standardized_difference":standardized_difference(difference,se)})
    }).collect::<Vec<_>>();
    json!({"kind":"cloud_reference_statistical_comparison_v1","gpu_initialized":false,
        "dimensions":[a.mean.width,a.mean.height],"reference_samples":a.samples,"candidate_samples":b.samples,
        "reference_backend":a.metadata["scene"]["backend"],"candidate_backend":b.metadata["scene"]["backend"],
        "reference_seed":a.metadata["scene"]["render"]["seed"],"candidate_seed":b.metadata["scene"]["render"]["seed"],
        "reference_transport":a.metadata["scene"]["transport"],"candidate_transport":b.metadata["scene"]["transport"],
        "diagnostics_passed":true,"all_pixels_used":true,"dropped_paths_or_pixels":0,
        "component_count":len*3,"sigma_limit":sigma_limit,"components_outside_limit":outside,"components_outside_3sigma":outside_three,
        "zero_variance_mismatches":zero_variance_mismatches,"within_sigma_limit":outside==0,
        "max_abs_standardized_difference":if zero_variance_mismatches>0 {None}else{abs_z.last().copied()},
        "relative_rgb_difference_percent":{"p50":percentile(&relative,0.5),"p95":percentile(&relative,0.95),"p99":percentile(&relative,0.99),"max":relative.last()},
        "spatial_mean_rgb":image_mean,"largest_standardized_components":components.into_iter().take(12).collect::<Vec<_>>(),
        "statistical_interpretation":"Per-channel joint SE = sqrt(sample_variance_A/n_A + sample_variance_B/n_B), assuming independently sampled images. Spatial mean SE additionally assumes independent pixel streams. No RGB cross-channel covariance is available; no RGB chi-square claim is made. Heavy tails and multiple comparisons limit finite-sample z-score interpretation; passing is a consistency diagnostic, not proof of unbiasedness.",
        "provenance_limit":"Source path/byte count/grid/stats, physical parameters, camera and dimensions must agree. Original asset content hashes/transform are not in these manifests, so provenance equality alone does not certify byte-identical source assets. Exposure and unbiased proposal/roulette controls may differ."})
}

fn check_constant_environment(reference: &Reference) -> Result<Value> {
    let transport = &reference.metadata["scene"]["transport"];
    if transport["scattering_albedo"] != json!([1.0, 1.0, 1.0])
        || transport["sun_irradiance"] != json!([0.0, 0.0, 0.0])
        || field(transport, "ground")? != &Value::Null
    {
        return Err(
            "constant-environment limit requires albedo one, zero sun irradiance, and no ground"
                .into(),
        );
    }
    let sky = transport["sky_radiance"]
        .as_array()
        .filter(|array| array.len() == 3)
        .ok_or("invalid constant sky RGB")?;
    let mut expected = [0.0_f32; 3];
    for channel in 0..3 {
        let value = sky[channel]
            .as_f64()
            .ok_or("invalid constant sky component")?;
        expected[channel] = value as f32;
        if !expected[channel].is_finite() || expected[channel] < 0.0 {
            return Err("constant sky component must be finite and nonnegative".into());
        }
    }
    let mismatches = reference
        .mean
        .values
        .iter()
        .flatten()
        .zip(expected.into_iter().cycle())
        .filter(|(value, expected)| value.to_bits() != expected.to_bits())
        .count();
    let nonzero_variance = reference
        .variance
        .values
        .iter()
        .flatten()
        .filter(|value| **value != 0.0)
        .count();
    Ok(
        json!({"kind":"cloud_constant_environment_analytic_check_v1","gpu_initialized":false,
        "dimensions":[reference.mean.width,reference.mean.height],"samples_per_pixel":reference.samples,
        "backend":reference.metadata["scene"]["backend"],"expected_linear_rgb_f32":expected,
        "diagnostics_passed":true,"all_pixels_used":true,"dropped_paths_or_pixels":0,
        "nonmatching_mean_components":mismatches,"nonzero_variance_components":nonzero_variance,
        "analytic_limit_passed":mismatches==0 && nonzero_variance==0,
        "interpretation":"For a bounded nonemitting medium with scattering albedo one, no ground, zero delta-sun irradiance, and a constant environment, every completed path has exactly the same sky radiance; sample variance is exactly zero. This checks the declared limit and all exported pixel components."}),
    )
}

fn check_exact_f32(a: &Reference, b: &Reference) -> Result<Value> {
    if a.samples != b.samples
        || a.metadata["scene"]["backend"] != b.metadata["scene"]["backend"]
        || a.metadata["scene"]["render"]["seed"] != b.metadata["scene"]["render"]["seed"]
        || a.metadata["scene"]["transport"] != b.metadata["scene"]["transport"]
    {
        return Err("exact f32 comparison requires matching sample counts, backend, seed, and all algorithm controls".into());
    }
    let mismatches = |left: &[[f32; 3]], right: &[[f32; 3]]| {
        left.iter()
            .flatten()
            .zip(right.iter().flatten())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let mean_mismatches = mismatches(&a.mean.values, &b.mean.values);
    let variance_mismatches = mismatches(&a.variance.values, &b.variance.values);
    Ok(
        json!({"kind":"cloud_reference_exact_f32_comparison_v1","gpu_initialized":false,
        "dimensions":[a.mean.width,a.mean.height],"samples_per_pixel":a.samples,
        "reference_batch_size":a.metadata["scene"]["render"]["sample_batch_size"],
        "candidate_batch_size":b.metadata["scene"]["render"]["sample_batch_size"],
        "diagnostics_passed":true,"all_pixels_used":true,"dropped_paths_or_pixels":0,
        "radiance_component_mismatches":mean_mismatches,"variance_component_mismatches":variance_mismatches,
        "exact_f32_passed":mean_mismatches==0 && variance_mismatches==0,
        "interpretation":"Compares every decoded f32 radiance and sample-variance bit. EXR compression/header differences do not affect the result. This is a deterministic regression check, not an independence-based statistical test."}),
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.out.exists() || !cli.sigma_limit.is_finite() || cli.sigma_limit <= 0.0 {
        return Err("use a new output file and positive finite sigma limit".into());
    }
    let a = load(&cli.reference)?;
    if cli.constant_environment && cli.exact_f32 {
        return Err("choose one comparison mode".into());
    }
    let mut result = if cli.constant_environment {
        if cli.candidate.is_some() {
            return Err("constant-environment mode takes one reference directory".into());
        }
        check_constant_environment(&a)?
    } else {
        let candidate = cli
            .candidate
            .as_ref()
            .ok_or("comparison needs a candidate directory")?;
        let b = load(candidate)?;
        validate_same_model(&a, &b)?;
        if cli.exact_f32 {
            check_exact_f32(&a, &b)?
        } else {
            compare(&a, &b, cli.sigma_limit)
        }
    };
    result["reference"] = json!(cli.reference);
    result["candidate"] = json!(cli.candidate);
    if let Some(parent) = cli.out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out, serde_json::to_vec_pretty(&result)?)?;
    if cli.constant_environment {
        println!(
            "constant-environment mean mismatches {}, nonzero variances {}; report {}",
            result["nonmatching_mean_components"],
            result["nonzero_variance_components"],
            cli.out.display()
        );
        if result["analytic_limit_passed"] != true {
            return Err(
                "cloud constant-environment analytic limit failed; inspect the report".into(),
            );
        }
    } else if cli.exact_f32 {
        println!(
            "exact f32 radiance mismatches {}, variance mismatches {}; report {}",
            result["radiance_component_mismatches"],
            result["variance_component_mismatches"],
            cli.out.display()
        );
        if result["exact_f32_passed"] != true {
            return Err("cloud exact f32 comparison failed; inspect the report".into());
        }
    } else {
        println!(
            "{} / {} components outside {} estimated SE; report {}",
            result["components_outside_limit"],
            result["component_count"],
            cli.sigma_limit,
            cli.out.display()
        );
    }
    if !cli.constant_environment && !cli.exact_f32 && result["within_sigma_limit"] != true {
        return Err("cloud image means exceed the requested estimated-SE diagnostic limit; inspect the report".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_mean_variances_add_without_reusing_sample_variance() {
        assert_eq!(joint_se(4.0, 16, 9.0, 36), 0.5_f64.sqrt());
        assert_eq!(standardized_difference(0.0, 0.0), Some(0.0));
        assert_eq!(standardized_difference(1.0, 0.0), None);
    }
}
