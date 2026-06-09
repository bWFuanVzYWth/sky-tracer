use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;
use exr::prelude::read_first_rgba_layer_from_file;
use sky_core::asset::{SPECTRAL_SKY_VIEW_LUT_KIND, SpectralAssetManifest};
use sky_core::spectrum::{BAND_COUNT, cie_1931_2deg};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Search physically weighted three-wavelength sky-view LUT primaries"
)]
struct Cli {
    #[arg(long = "asset-dir", required = true)]
    asset_dirs: Vec<PathBuf>,
    #[arg(long, default_value_t = 65_536)]
    screen_samples: usize,
    #[arg(long, default_value_t = 512)]
    refine_top: usize,
    #[arg(long, default_value_t = 11)]
    top: usize,
    #[arg(long, default_value = "target/primary_search")]
    out_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, Default)]
struct Xyz {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Oklab {
    l: f64,
    a: f64,
    b: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Combo {
    a: usize,
    b: usize,
    c: usize,
}

#[derive(Clone, Debug)]
struct ComboResult {
    combo: Combo,
    wavelengths_nm: [f32; 3],
    weights_nm: [f64; 3],
    sample_count: usize,
    mean: f64,
    rmse: f64,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    max: f64,
    score: f64,
}

#[derive(Debug)]
struct Dataset {
    asset_summaries: Vec<AssetSummary>,
    wavelengths_nm: Vec<f32>,
    band_width_nm: Vec<f64>,
    bands: Vec<Vec<f32>>,
    reference_oklab: Vec<Oklab>,
    reference_y: Vec<f64>,
    reference_scale: f64,
}

#[derive(Debug)]
struct AssetSummary {
    path: PathBuf,
    dimensions: [usize; 2],
    spp: usize,
    sun_elevation_deg: f32,
    sample_start: usize,
    sample_count: usize,
}

#[derive(Debug)]
struct LinearImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out_dir)?;

    let dataset = Dataset::load(&cli.asset_dirs)?;
    let sample_indices = screening_indices(dataset.sample_count(), cli.screen_samples);
    let combos = all_combos(dataset.wavelengths_nm.len());
    println!(
        "loaded {} assets, {} samples, {} bands, {} wavelength triples",
        dataset.asset_summaries.len(),
        dataset.sample_count(),
        dataset.wavelengths_nm.len(),
        combos.len()
    );
    println!(
        "screening with {} deterministic samples, refining top {} on all samples",
        sample_indices.len(),
        cli.refine_top
    );

    let mut screened = combos
        .iter()
        .copied()
        .map(|combo| evaluate_combo(&dataset, combo, Some(&sample_indices), false))
        .collect::<Vec<_>>();
    screened.sort_by(|a, b| a.score.total_cmp(&b.score));

    let refine_count = cli.refine_top.min(screened.len());
    let mut refined = screened
        .iter()
        .take(refine_count)
        .map(|result| evaluate_combo(&dataset, result.combo, None, true))
        .collect::<Vec<_>>();
    refined.sort_by(|a, b| a.score.total_cmp(&b.score));

    let top_count = cli.top.min(refined.len());
    write_csv(&cli.out_dir.join("ranking.csv"), &refined[..top_count])?;
    write_per_elevation_csv(
        &cli.out_dir.join("ranking_by_elevation.csv"),
        &dataset,
        &refined[..top_count],
    )?;
    write_report(
        &cli.out_dir.join("report.md"),
        &dataset,
        &sample_indices,
        refine_count,
        &refined[..top_count],
    )?;

    println!("top {}:", top_count);
    for (rank, result) in refined.iter().take(top_count).enumerate() {
        println!(
            "#{:<2} {:>5.0}/{:>5.0}/{:>5.0}nm score={:.6} rmse={:.6} p95={:.6} max={:.6}",
            rank + 1,
            result.wavelengths_nm[0],
            result.wavelengths_nm[1],
            result.wavelengths_nm[2],
            result.score,
            result.rmse,
            result.p95,
            result.max
        );
    }
    println!("wrote {}", cli.out_dir.display());
    Ok(())
}

impl Dataset {
    fn load(asset_dirs: &[PathBuf]) -> Result<Self, Box<dyn Error>> {
        if asset_dirs.is_empty() {
            return Err("at least one --asset-dir is required".into());
        }

        let mut asset_summaries = Vec::new();
        let mut wavelengths_nm = Vec::new();
        let mut bands = Vec::<Vec<f32>>::new();
        let mut reference_xyz = Vec::<Xyz>::new();
        let mut reference_y = Vec::<f64>::new();

        for asset_dir in asset_dirs {
            let manifest_path = asset_dir.join("asset.json");
            let manifest: SpectralAssetManifest =
                serde_json::from_reader(File::open(&manifest_path)?)?;
            validate_manifest(&manifest, &manifest_path)?;

            if wavelengths_nm.is_empty() {
                wavelengths_nm = manifest.band_centers_nm.clone();
                bands = vec![Vec::new(); wavelengths_nm.len()];
            } else if wavelengths_nm != manifest.band_centers_nm {
                return Err(format!(
                    "{}: wavelength grid differs from first asset",
                    manifest_path.display()
                )
                .into());
            }

            let width = manifest.dimensions[0];
            let height = manifest.dimensions[1];
            let asset_sample_count = width * height;
            let mut asset_bands = Vec::with_capacity(wavelengths_nm.len());
            for file in &manifest.files.band_exrs {
                let image = read_band_exr(&asset_dir.join(file))?;
                if image.width != width || image.height != height {
                    return Err(format!(
                        "{}: dimensions {}x{} differ from manifest {}x{}",
                        asset_dir.join(file).display(),
                        image.width,
                        image.height,
                        width,
                        height
                    )
                    .into());
                }
                asset_bands.push(image.pixels);
            }

            let cie_weights = wavelengths_nm
                .iter()
                .map(|lambda| {
                    let xyz = cie_1931_2deg(*lambda);
                    Xyz {
                        x: xyz.x as f64,
                        y: xyz.y as f64,
                        z: xyz.z as f64,
                    }
                })
                .collect::<Vec<_>>();
            for pixel in 0..asset_sample_count {
                let mut xyz = Xyz::default();
                for (band, weight) in asset_bands.iter().zip(cie_weights.iter()) {
                    let value = band[pixel] as f64;
                    xyz.x += value * weight.x;
                    xyz.y += value * weight.y;
                    xyz.z += value * weight.z;
                }
                reference_y.push(xyz.y);
                reference_xyz.push(xyz);
            }
            for (dst, src) in bands.iter_mut().zip(asset_bands.into_iter()) {
                dst.extend(src);
            }
            let sample_start = reference_xyz.len() - asset_sample_count;
            asset_summaries.push(AssetSummary {
                path: asset_dir.clone(),
                dimensions: manifest.dimensions,
                spp: manifest.spp,
                sun_elevation_deg: manifest.sun_elevation_deg,
                sample_start,
                sample_count: asset_sample_count,
            });
        }

        let reference_scale = percentile(reference_y.clone(), 0.99).max(1.0e-12);
        let reference_oklab = reference_xyz
            .iter()
            .map(|xyz| xyz_to_oklab(*xyz / reference_scale))
            .collect::<Vec<_>>();
        Ok(Self {
            asset_summaries,
            band_width_nm: band_widths_from_centers(&wavelengths_nm),
            wavelengths_nm,
            bands,
            reference_oklab,
            reference_y,
            reference_scale,
        })
    }

    fn sample_count(&self) -> usize {
        self.reference_oklab.len()
    }
}

fn validate_manifest(
    manifest: &SpectralAssetManifest,
    manifest_path: &Path,
) -> Result<(), Box<dyn Error>> {
    if manifest.version != 0 {
        return Err(format!("{}: expected version 0", manifest_path.display()).into());
    }
    if manifest.kind != SPECTRAL_SKY_VIEW_LUT_KIND {
        return Err(format!(
            "{}: expected kind {}, found {}",
            manifest_path.display(),
            SPECTRAL_SKY_VIEW_LUT_KIND,
            manifest.kind
        )
        .into());
    }
    if manifest.band_centers_nm.len() != BAND_COUNT {
        return Err(format!(
            "{}: expected {} bands, found {}",
            manifest_path.display(),
            BAND_COUNT,
            manifest.band_centers_nm.len()
        )
        .into());
    }
    if manifest.files.band_exrs.len() != manifest.band_centers_nm.len() {
        return Err(format!("{}: band file count mismatch", manifest_path.display()).into());
    }
    Ok(())
}

fn read_band_exr(path: &Path) -> Result<LinearImage, Box<dyn Error>> {
    let image = read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            let width = resolution.width();
            let height = resolution.height();
            LinearImage {
                width,
                height,
                pixels: vec![0.0; width * height],
            }
        },
        |image, position, (r, _g, _b, _a): (f32, f32, f32, f32)| {
            image.pixels[position.y() * image.width + position.x()] = r;
        },
    )?;
    Ok(image.layer_data.channel_data.pixels)
}

fn evaluate_combo(
    dataset: &Dataset,
    combo: Combo,
    indices: Option<&[usize]>,
    exact_percentiles: bool,
) -> ComboResult {
    let weights_nm = quadrature_weights(&dataset.wavelengths_nm, &dataset.band_width_nm, combo);
    let wavelengths_nm = [
        dataset.wavelengths_nm[combo.a],
        dataset.wavelengths_nm[combo.b],
        dataset.wavelengths_nm[combo.c],
    ];
    let cie = [
        cie_1931_2deg(wavelengths_nm[0]),
        cie_1931_2deg(wavelengths_nm[1]),
        cie_1931_2deg(wavelengths_nm[2]),
    ];
    let selected = [
        &dataset.bands[combo.a],
        &dataset.bands[combo.b],
        &dataset.bands[combo.c],
    ];
    let band_widths = [
        dataset.band_width_nm[combo.a],
        dataset.band_width_nm[combo.b],
        dataset.band_width_nm[combo.c],
    ];
    let sample_count = indices.map_or_else(|| dataset.sample_count(), <[usize]>::len);
    let mut errors = exact_percentiles.then(|| Vec::with_capacity(sample_count));
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for sample in sample_iterator(dataset.sample_count(), indices) {
        let mut xyz = Xyz::default();
        for channel in 0..3 {
            // Baked band EXRs are already integrated over their source band.
            // Scale by physical Voronoi width to approximate the larger sparse bin.
            let spectral_value =
                selected[channel][sample] as f64 * weights_nm[channel] / band_widths[channel];
            xyz.x += spectral_value * cie[channel].x as f64;
            xyz.y += spectral_value * cie[channel].y as f64;
            xyz.z += spectral_value * cie[channel].z as f64;
        }
        let candidate = xyz_to_oklab(xyz / dataset.reference_scale);
        let error = oklab_distance(candidate, dataset.reference_oklab[sample]);
        sum += error;
        sum_squares += error * error;
        if let Some(errors) = &mut errors {
            errors.push(error);
        }
    }
    let mean = sum / sample_count as f64;
    let rmse = (sum_squares / sample_count as f64).sqrt();
    let (p50, p90, p95, p99, max, score) = if exact_percentiles {
        let errors = errors.expect("exact percentile errors");
        let max = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sorted = sorted_finite(errors);
        let p50 = percentile_sorted(&sorted, 0.50);
        let p90 = percentile_sorted(&sorted, 0.90);
        let p95 = percentile_sorted(&sorted, 0.95);
        let p99 = percentile_sorted(&sorted, 0.99);
        (p50, p90, p95, p99, max, rmse + 0.25 * p95)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, rmse)
    };
    ComboResult {
        combo,
        wavelengths_nm,
        weights_nm,
        sample_count,
        mean,
        rmse,
        p50,
        p90,
        p95,
        p99,
        max,
        score,
    }
}

fn sample_iterator<'a>(
    sample_count: usize,
    indices: Option<&'a [usize]>,
) -> Box<dyn Iterator<Item = usize> + 'a> {
    match indices {
        Some(indices) => Box::new(indices.iter().copied()),
        None => Box::new(0..sample_count),
    }
}

fn all_combos(len: usize) -> Vec<Combo> {
    let mut combos = Vec::new();
    for a in 0..len {
        for b in a + 1..len {
            for c in b + 1..len {
                combos.push(Combo { a, b, c });
            }
        }
    }
    combos
}

fn band_widths_from_centers(wavelengths_nm: &[f32]) -> Vec<f64> {
    let mut widths = Vec::with_capacity(wavelengths_nm.len());
    for i in 0..wavelengths_nm.len() {
        let lo = if i == 0 {
            wavelengths_nm[i] as f64 - 0.5 * (wavelengths_nm[i + 1] - wavelengths_nm[i]) as f64
        } else {
            0.5 * (wavelengths_nm[i - 1] + wavelengths_nm[i]) as f64
        };
        let hi = if i + 1 == wavelengths_nm.len() {
            wavelengths_nm[i] as f64 + 0.5 * (wavelengths_nm[i] - wavelengths_nm[i - 1]) as f64
        } else {
            0.5 * (wavelengths_nm[i] + wavelengths_nm[i + 1]) as f64
        };
        widths.push(hi - lo);
    }
    widths
}

fn quadrature_weights(wavelengths_nm: &[f32], band_widths_nm: &[f64], combo: Combo) -> [f64; 3] {
    let indices = [combo.a, combo.b, combo.c];
    let first_center = wavelengths_nm[0] as f64;
    let last_center = wavelengths_nm[wavelengths_nm.len() - 1] as f64;
    let lo_bound = first_center - 0.5 * band_widths_nm[0];
    let hi_bound = last_center + 0.5 * band_widths_nm[band_widths_nm.len() - 1];
    let centers = [
        wavelengths_nm[indices[0]] as f64,
        wavelengths_nm[indices[1]] as f64,
        wavelengths_nm[indices[2]] as f64,
    ];
    let split01 = 0.5 * (centers[0] + centers[1]);
    let split12 = 0.5 * (centers[1] + centers[2]);
    [split01 - lo_bound, split12 - split01, hi_bound - split12]
}

fn screening_indices(sample_count: usize, requested: usize) -> Vec<usize> {
    if requested == 0 || requested >= sample_count {
        return (0..sample_count).collect();
    }
    let stride = sample_count as f64 / requested as f64;
    (0..requested)
        .map(|i| ((i as f64 + 0.5) * stride).floor() as usize)
        .map(|index| index.min(sample_count - 1))
        .collect()
}

fn xyz_to_oklab(xyz: Xyz) -> Oklab {
    let l = (0.818_933_010_1 * xyz.x + 0.361_866_742_4 * xyz.y - 0.128_859_713_7 * xyz.z).cbrt();
    let m = (0.032_984_543_6 * xyz.x + 0.929_311_871_5 * xyz.y + 0.036_145_638_7 * xyz.z).cbrt();
    let s = (0.048_200_301_8 * xyz.x + 0.264_366_269_1 * xyz.y + 0.633_851_707 * xyz.z).cbrt();
    Oklab {
        l: 0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s,
        a: 1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s,
        b: 0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s,
    }
}

fn oklab_distance(a: Oklab, b: Oklab) -> f64 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

fn percentile(values: Vec<f64>, q: f64) -> f64 {
    let values = sorted_finite(values);
    percentile_sorted(&values, q)
}

fn sorted_finite(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    values
}

fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let index = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn write_csv(path: &Path, results: &[ComboResult]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "rank,lambda0_nm,lambda1_nm,lambda2_nm,weight0_nm,weight1_nm,weight2_nm,score,mean,rmse,p50,p90,p95,p99,max,sample_count"
    )?;
    for (rank, result) in results.iter().enumerate() {
        writeln!(
            file,
            "{},{:.0},{:.0},{:.0},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
            rank + 1,
            result.wavelengths_nm[0],
            result.wavelengths_nm[1],
            result.wavelengths_nm[2],
            result.weights_nm[0],
            result.weights_nm[1],
            result.weights_nm[2],
            result.score,
            result.mean,
            result.rmse,
            result.p50,
            result.p90,
            result.p95,
            result.p99,
            result.max,
            result.sample_count
        )?;
    }
    Ok(())
}

fn write_per_elevation_csv(
    path: &Path,
    dataset: &Dataset,
    results: &[ComboResult],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "rank,elevation_deg,lambda0_nm,lambda1_nm,lambda2_nm,score,mean,rmse,p50,p90,p95,p99,max,sample_count"
    )?;
    for (rank, result) in results.iter().enumerate() {
        for asset in &dataset.asset_summaries {
            let indices = asset_sample_indices(asset);
            let per_asset = evaluate_combo(dataset, result.combo, Some(&indices), true);
            writeln!(
                file,
                "{},{:.6},{:.0},{:.0},{:.0},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
                rank + 1,
                asset.sun_elevation_deg,
                result.wavelengths_nm[0],
                result.wavelengths_nm[1],
                result.wavelengths_nm[2],
                per_asset.score,
                per_asset.mean,
                per_asset.rmse,
                per_asset.p50,
                per_asset.p90,
                per_asset.p95,
                per_asset.p99,
                per_asset.max,
                per_asset.sample_count
            )?;
        }
    }
    Ok(())
}

fn asset_sample_indices(asset: &AssetSummary) -> Vec<usize> {
    (asset.sample_start..asset.sample_start + asset.sample_count).collect()
}

fn write_report(
    path: &Path,
    dataset: &Dataset,
    screen_indices: &[usize],
    refine_count: usize,
    results: &[ComboResult],
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "# Sky-View Three-Wavelength Search")?;
    writeln!(file)?;
    writeln!(file, "## Dataset")?;
    writeln!(file)?;
    writeln!(file, "- Assets: {}", dataset.asset_summaries.len())?;
    writeln!(file, "- Total LUT samples: {}", dataset.sample_count())?;
    writeln!(
        file,
        "- Wavelength grid: {} bands, {:.0}..{:.0} nm",
        dataset.wavelengths_nm.len(),
        dataset.wavelengths_nm[0],
        dataset.wavelengths_nm[dataset.wavelengths_nm.len() - 1]
    )?;
    writeln!(
        file,
        "- Reference Y p99 scale: {:.9e}",
        percentile(dataset.reference_y.clone(), 0.99)
    )?;
    writeln!(file, "- Screening samples: {}", screen_indices.len())?;
    writeln!(
        file,
        "- Full-resolution refined candidates: {}",
        refine_count
    )?;
    writeln!(file)?;
    writeln!(file, "| Elevation | SPP | Dimensions | Samples | Asset |")?;
    writeln!(file, "| ---: | ---: | ---: | ---: | --- |")?;
    for asset in &dataset.asset_summaries {
        writeln!(
            file,
            "| {:.2} | {} | {}x{} | {} | `{}` |",
            asset.sun_elevation_deg,
            asset.spp,
            asset.dimensions[0],
            asset.dimensions[1],
            asset.sample_count,
            asset.path.display()
        )?;
    }
    writeln!(file)?;
    writeln!(file, "## Ranking")?;
    writeln!(file)?;
    writeln!(
        file,
        "Screening ranks all triples by deterministic-subsample RMSE. Final score is `RMSE_OK + 0.25 * P95_OK`, computed on all loaded LUT samples after direct physical XYZ quadrature and Oklab conversion."
    )?;
    writeln!(file)?;
    writeln!(
        file,
        "| Rank | Wavelengths nm | Weights nm | Score | RMSE | Mean | P95 | P99 | Max |"
    )?;
    writeln!(
        file,
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for (rank, result) in results.iter().enumerate() {
        writeln!(
            file,
            "| {} | {:.0}, {:.0}, {:.0} | {:.1}, {:.1}, {:.1} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
            rank + 1,
            result.wavelengths_nm[0],
            result.wavelengths_nm[1],
            result.wavelengths_nm[2],
            result.weights_nm[0],
            result.weights_nm[1],
            result.weights_nm[2],
            result.score,
            result.rmse,
            result.mean,
            result.p95,
            result.p99,
            result.max
        )?;
    }
    if let Some(best) = results.first() {
        writeln!(file)?;
        writeln!(file, "## Best Combo By Elevation")?;
        writeln!(file)?;
        writeln!(
            file,
            "Best overall combo: `{:.0}, {:.0}, {:.0} nm`.",
            best.wavelengths_nm[0], best.wavelengths_nm[1], best.wavelengths_nm[2]
        )?;
        writeln!(file)?;
        writeln!(
            file,
            "| Elevation | Score | RMSE | Mean | P95 | P99 | Max |"
        )?;
        writeln!(file, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")?;
        for asset in &dataset.asset_summaries {
            let indices = asset_sample_indices(asset);
            let result = evaluate_combo(dataset, best.combo, Some(&indices), true);
            writeln!(
                file,
                "| {:.2} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
                asset.sun_elevation_deg,
                result.score,
                result.rmse,
                result.mean,
                result.p95,
                result.p99,
                result.max
            )?;
        }
        writeln!(file)?;
        writeln!(
            file,
            "Full per-elevation metrics for every ranked combo are in `ranking_by_elevation.csv`. Rank 2 through rank 11 are the top 10 runners-up when 11 results are requested."
        )?;
    }
    Ok(())
}

impl std::ops::Div<f64> for Xyz {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}
