use clap::{Parser, Subcommand, ValueEnum};
use glam::Vec3;
use sky_atmosphere_lut::{
    Result,
    asset::{Manifest, fingerprint},
    config::{AngularIntegration, BakeConfig},
    model::Model,
    solver::GpuBaker,
};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Offline phase-inclusive spectral atmosphere LUT baker (f32 / wgpu)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Reference,
    Development,
    Smoke,
}
#[derive(clap::Args)]
struct Input {
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long, value_enum, default_value = "reference")]
    preset: Preset,
    /// Complete JSON BakeConfig; overrides preset.
    #[arg(long)]
    config: Option<PathBuf>,
}
#[derive(Subcommand)]
enum Command {
    /// Validate inputs, print byte/work budgets and the resolved config; no GPU work.
    Plan {
        #[command(flatten)]
        input: Input,
        /// CPU enumeration of exact duplicate states and compact readback costs.
        #[arg(long)]
        analyze_work: bool,
    },
    /// Create an asset, writing each completed band with a checksum.
    Bake {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value = "out/atmosphere_lut")]
        out: PathBuf,
        /// Continue the same configuration and input model, verifying finished bands.
        #[arg(long)]
        resume: bool,
        /// Bake just this zero-based band for expensive convergence studies.
        #[arg(long)]
        band_index: Option<usize>,
    },
    Inspect {
        asset: PathBuf,
        #[arg(long)]
        verify: bool,
    },
    /// CPU-only conversion of a completed spectral bake into linear Rec.2020.
    ExportRgb {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// CPU-only packing of an RGB export, keeping every grid point and its f32 exponent.
    CompressRgb {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        block_texels: usize,
    },
    /// CPU samples of the spectral reference for fitting a separate production LUT.
    Synthesize {
        source: PathBuf,
        /// JSON array of altitude, solar elevation, view elevation and relative azimuth.
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the complete band-integrated radiance spectrum as CSV.
    Sample {
        asset: PathBuf,
        #[arg(long, default_value_t = 0.2)]
        altitude_km: f32,
        #[arg(long, default_value_t = 30.0, allow_hyphen_values = true)]
        sun_elevation_deg: f32,
        #[arg(long, default_value_t = 45.0, allow_hyphen_values = true)]
        view_elevation_deg: f32,
        #[arg(long, default_value_t = 90.0)]
        relative_azimuth_deg: f32,
        #[arg(long)]
        sun_disk: bool,
    },
    #[cfg(feature = "reference")]
    /// Compare against the existing GPU spectral path tracer with identical input data.
    Compare {
        asset: PathBuf,
        /// Diagnostic PT prefix; omitted means the normal untruncated estimator.
        #[arg(long)]
        pt_max_orders: Option<u32>,
        /// Compare one completed band; permits a partial diagnostic bake.
        #[arg(long)]
        band_index: Option<usize>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        width: usize,
        #[arg(long, default_value_t = 16)]
        height: usize,
        #[arg(long, default_value_t = 1024)]
        spp: usize,
        #[arg(long)]
        seed: Option<u64>,
        /// Samples per pixel axis for deterministic LUT pixel integration.
        #[arg(long, default_value_t = 4)]
        pixel_samples: usize,
        #[arg(long, default_value_t = 30.0, allow_hyphen_values = true)]
        sun_elevation_deg: f32,
        #[arg(long, default_value_t = 0.2)]
        altitude_km: f32,
        #[arg(long, default_value = "out/lut_comparison.json")]
        out: PathBuf,
    },
}

impl Input {
    fn load(&self) -> Result<(Model, BakeConfig)> {
        let scene =
            sky_core::data::load_scene_data(&self.data_dir, 0.0, 0.0).map_err(|e| e.to_string())?;
        let model = Model::from_scene(&scene)?;
        let config = if let Some(path) = &self.config {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            match self.preset {
                Preset::Reference => BakeConfig::reference(),
                Preset::Development => BakeConfig::development(),
                Preset::Smoke => BakeConfig::smoke(),
            }
        };
        config.validate(model.bands.len())?;
        config.validate_top_height(model.geometry.top_height())?;
        Ok((model, config))
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Plan {
            input,
            analyze_work,
        } => {
            let (model, config) = input.load()?;
            let n = config.scattering_len();
            let work = analyze_work.then(|| {
                let nodes =
                    sky_atmosphere_lut::bake_schedule::angular_nodes(model.geometry, &config);
                sky_atmosphere_lut::bake_schedule::WorkEstimate::from_nodes(&config, &nodes)
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "bands":model.bands.len(),"model_fingerprint_fnv1a64":fingerprint(&model)?,"asset_bytes_including_metadata_reserve":config.asset_bytes(model.bands.len())?,
                    "decimal_gb":config.asset_bytes(model.bands.len())? as f32/1e9,
                    "scattering_texels_per_band":n,"angular_lookups_per_later_order_per_band":n as u64*config.angular_mu as u64*config.angular_phi as u64*if config.angular_integration==AngularIntegration::SunAndView {2} else {1},
                    "working_transport_storage_bytes":4*(4*n+config.optical_depth_len()+4*config.ground_sun_samples+n.div_ceil(64)*5+n.div_ceil(64).div_ceil(256)*5),
                    "work_reuse":work,"gpu_timing_measured":false,
                    "config":config
                }))?
            );
        }
        Command::Bake {
            input,
            out,
            resume,
            band_index,
        } => {
            let (model, config) = input.load()?;
            if band_index.is_some_and(|i| i >= model.bands.len()) {
                return Err("band index out of range".into());
            }
            let mut manifest = if resume {
                let m = Manifest::open(&out)?;
                if m.rgb.is_some() {
                    return Err("cannot resume baking an exported RGB asset".into());
                }
                if m.solver != sky_atmosphere_lut::asset::SOLVER {
                    return Err("cannot resume a LUT baked with an older solver; use a new output directory".into());
                }
                if m.config != config || m.model_fingerprint_fnv1a64 != fingerprint(&model)? {
                    return Err("resume input model or configuration differs from asset".into());
                }
                for (i, record) in m.records.iter().enumerate() {
                    if record.is_some() {
                        m.read_band(&out, i)?;
                    }
                }
                if m.complete() {
                    println!("Asset is already complete; all checksums verified.");
                    return Ok(());
                }
                m
            } else {
                if out.exists() {
                    return Err(
                        "output already exists; choose a new directory or use --resume".into(),
                    );
                }
                Manifest::new(&model, config.clone(), String::new())?
            };
            let gpu = GpuBaker::new()?;
            eprintln!(
                "GPU: {}; asset budget {} bytes",
                gpu.adapter_name,
                config.asset_bytes(model.bands.len())?
            );
            manifest.adapter = gpu.adapter_name.clone();
            fs::create_dir_all(&out)?;
            manifest.save(&out)?;
            for i in 0..model.bands.len() {
                if band_index.is_some_and(|selected| selected != i) {
                    continue;
                }
                if manifest.records[i].is_some() {
                    continue;
                }
                let band = gpu.bake_band(&model, &config, i, |s| eprintln!("{s}"))?;
                manifest.write_band(&out, i, band)?;
                eprintln!(
                    "committed band {}/{} ({:.0} nm)",
                    i + 1,
                    model.bands.len(),
                    model.bands[i].info.center_nm
                );
            }
            println!(
                "{}: {}",
                if manifest.complete() {
                    "Complete"
                } else {
                    "Selected band complete; asset remains partial"
                },
                out.join("asset.json").display()
            );
        }
        Command::Inspect { asset, verify } => {
            let m = Manifest::open(&asset)?;
            if verify {
                if m.rgb.is_some() {
                    sky_atmosphere_lut::rgb::verify(&m, &asset)?;
                }
                for (i, r) in m.records.iter().enumerate() {
                    if r.is_some() {
                        m.read_band(&asset, i)?;
                    }
                }
            }
            println!("{}", serde_json::to_string_pretty(&m)?);
        }
        Command::ExportRgb { source, out } => {
            let m = sky_atmosphere_lut::rgb::export(&source, &out)?;
            sky_atmosphere_lut::rgb::verify(&m, &out)?;
            println!("Verified RGB asset: {}", out.join("asset.json").display());
        }
        Command::CompressRgb {
            source,
            out,
            block_texels,
        } => {
            let m = sky_atmosphere_lut::packed::compress(&source, &out, block_texels)?;
            sky_atmosphere_lut::rgb::verify(&m, &out)?;
            println!(
                "Verified packed RGB asset: {}",
                out.join("asset.json").display()
            );
        }
        Command::Synthesize {
            source,
            queries,
            out,
        } => {
            let points: Vec<sky_atmosphere_lut::synthesis::SamplePoint> =
                serde_json::from_slice(&fs::read(queries)?)?;
            sky_atmosphere_lut::synthesis::export(&source, &points, &out)?;
            println!("Reference dataset: {}", out.join("dataset.json").display());
        }
        Command::Sample {
            asset,
            altitude_km,
            sun_elevation_deg,
            view_elevation_deg,
            relative_azimuth_deg,
            sun_disk,
        } => {
            let m = Manifest::open(&asset)?;
            if m.rgb.is_some() {
                return Err("spectral sampling requires the source spectral asset".into());
            }
            if !m.complete() {
                return Err("asset is incomplete".into());
            }
            println!("center_nm,lower_nm,upper_nm,radiance_w_m2_sr");
            for i in 0..m.bands.len() {
                let band = m.read_band(&asset, i)?;
                let value = band.sample(
                    altitude_km,
                    direction(view_elevation_deg, relative_azimuth_deg),
                    direction(sun_elevation_deg, 0.0),
                    sun_disk,
                )?;
                println!(
                    "{},{},{},{}",
                    band.info.center_nm, band.info.lower_nm, band.info.upper_nm, value
                );
            }
        }
        #[cfg(feature = "reference")]
        Command::Compare {
            asset,
            pt_max_orders,
            band_index,
            data_dir,
            width,
            height,
            spp,
            seed,
            pixel_samples,
            sun_elevation_deg,
            altitude_km,
            out,
        } => {
            compare(
                &asset,
                pt_max_orders,
                band_index,
                &data_dir,
                width,
                height,
                spp,
                seed,
                pixel_samples,
                sun_elevation_deg,
                altitude_km,
                &out,
            )?;
        }
    }
    Ok(())
}

fn direction(elevation: f32, azimuth: f32) -> Vec3 {
    let e = elevation.to_radians();
    let a = azimuth.to_radians();
    Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
}

#[cfg(feature = "reference")]
#[allow(clippy::too_many_arguments)]
fn compare(
    asset: &std::path::Path,
    pt_max_orders: Option<u32>,
    band_index: Option<usize>,
    data: &std::path::Path,
    width: usize,
    height: usize,
    spp: usize,
    seed: Option<u64>,
    pixel_samples: usize,
    elevation: f32,
    altitude: f32,
    out: &std::path::Path,
) -> Result<()> {
    use sky_spectral_path_tracer::{RenderConfig, render};
    let m = Manifest::open(asset)?;
    if m.rgb.is_some() {
        return Err("spectral comparison requires the source spectral asset; use demo snapshots for RGB comparison".into());
    }
    let scene = sky_core::data::load_scene_data(data, elevation, 0.0).map_err(|e| e.to_string())?;
    if band_index.is_some_and(|i| i >= m.bands.len() || m.records[i].is_none()) {
        return Err("selected band is not available".into());
    }
    if band_index.is_none() && !m.complete() {
        return Err("comparison needs a complete asset".into());
    }
    let input_fingerprint = fingerprint(&Model::from_scene(&scene)?)?;
    if m.model_fingerprint_fnv1a64 != input_fingerprint {
        return Err(format!(
            "input model fingerprint {input_fingerprint} differs from asset {}",
            m.model_fingerprint_fnv1a64
        )
        .into());
    }
    if (m.config.ground_albedo - sky_core::atmosphere::GROUND_ALBEDO).abs() > 1e-6 {
        return Err(
            "the existing path tracer uses fixed ground albedo 0.18; the LUT albedo differs".into(),
        );
    }
    if pixel_samples == 0 || pixel_samples > 64 {
        return Err("pixel_samples must be in 1..=64".into());
    }
    if !altitude.is_finite()
        || altitude < 0.0
        || !elevation.is_finite()
        || !(-90.0..=90.0).contains(&elevation)
    {
        return Err("invalid comparison height or solar elevation".into());
    }
    eprintln!("Rendering GPU path-traced reference, {width}x{height}, {spp} spp...");
    let render_config = RenderConfig {
        width,
        height,
        spp,
        seed: seed.unwrap_or_else(|| RenderConfig::default().seed),
        sun_elevation_deg: elevation,
        observer_altitude_km: altitude,
        ..Default::default()
    };
    let film = if let Some(band) = band_index {
        sky_spectral_path_tracer::render_band(&scene, &render_config, band, pt_max_orders)?
    } else {
        match pt_max_orders {
            Some(n) => sky_spectral_path_tracer::render_orders(&scene, &render_config, n)?,
            None => render(&scene, &render_config)?,
        }
    };
    let sun = direction(elevation, 0.0);
    let mut bands = Vec::new();
    for i in 0..m.bands.len() {
        if band_index.is_some_and(|selected| selected != i) {
            continue;
        }
        let lut = m.read_band(asset, i)?;
        let mut statistics = [Statistics::default(); 6];
        let mut profiles = Vec::new();
        for y in 0..height {
            for x in 0..width {
                // 4x4 pixel integration matches the path tracer's uniformly jittered panorama.
                // Pixels touching the direct solar disk are excluded (the disk is not baked).
                if pixel_overlaps_sun(x, y, width, height, elevation, m.sun_radius_rad) {
                    continue;
                }
                let mut value = 0.0;
                for sy in 0..pixel_samples {
                    for sx in 0..pixel_samples {
                        let u =
                            (x as f32 + (sx as f32 + 0.5) / pixel_samples as f32) / width as f32;
                        let v =
                            (y as f32 + (sy as f32 + 0.5) / pixel_samples as f32) / height as f32;
                        let dir = direction(90.0 - v * 180.0, u * 360.0 - 180.0);
                        value += lut.sample(altitude, dir, sun, false)?
                            / (pixel_samples * pixel_samples) as f32;
                    }
                }
                let reference = film.pixel_spectrum(y * width + x)[i];
                let view_elevation = 90.0 - (y as f32 + 0.5) * 180.0 / height as f32;
                let azimuth = (x as f32 + 0.5) * 360.0 / width as f32 - 180.0;
                let horizon = m.geometry.horizon(altitude).asin().to_degrees();
                let above_horizon = view_elevation - horizon;
                let sun_angle = direction(view_elevation, azimuth)
                    .dot(sun)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                let sky = above_horizon > 0.0;
                if x == 0 || x == width / 2 {
                    profiles.push(serde_json::json!({"elevation_deg":view_elevation,"azimuth_deg":azimuth,"lut":value,"path_traced":reference}));
                }
                let masks = [
                    true,
                    sky,
                    sky && sun_angle < 10.0,
                    sky && (10.0..30.0).contains(&sun_angle),
                    above_horizon.abs() < 3.0,
                    sky && above_horizon < 12.0 && azimuth.abs() > 150.0,
                ];
                for (stats, included) in statistics.iter_mut().zip(masks) {
                    if included {
                        stats.add(value, reference);
                    }
                }
            }
        }
        if statistics[0].count == 0 {
            return Err("all comparison pixels excluded; increase image dimensions".into());
        }
        let mut entry = statistics[0].json();
        entry["center_nm"] = m.bands[i].center_nm.into();
        entry["vertical_profiles"] = profiles.into();
        entry["regions"] = serde_json::json!({
            "sky":statistics[1].json(),"near_sun":statistics[2].json(),"solar_aureole":statistics[3].json(),
            "horizon":statistics[4].json(),"earth_shadow":statistics[5].json()});
        bands.push(entry);
    }
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        out,
        serde_json::to_vec_pretty(
            &serde_json::json!({"width":width,"height":height,"spp":spp,"sun_elevation_deg":elevation,
        "observer_altitude_km":altitude,"pt_seed":render_config.seed,"pt_transport_version":sky_core::asset::LAYERED_TRANSPORT_VERSION,"pt_max_orders":pt_max_orders,"lut_pixel_quadrature":format!("{pixel_samples}x{pixel_samples}"),"excluded":"pixels intersecting solar disk bounding box",
        "asset":asset,"model_fingerprint_fnv1a64":m.model_fingerprint_fnv1a64,"coordinate_mapping":m.coordinate_mapping,"config":m.config,
        "region_definitions":{"sky":"above geometric horizon","near_sun":"sky, scattering angle < 10 degrees, solar disk pixels excluded",
        "solar_aureole":"sky, scattering angle 10..30 degrees","horizon":"within 3 degrees of geometric horizon on either side",
        "earth_shadow":"anti-solar azimuth +/-30 degrees, 0..12 degrees above geometric horizon; useful at negative solar elevations"},
        "note":"diagnostic, not a pass/fail criterion; path tracer noise, finite order and grid bias remain","bands":bands}),
        )?,
    )?;
    println!("Comparison: {}", out.display());
    Ok(())
}

#[cfg(feature = "reference")]
#[derive(Clone, Copy, Default)]
struct Statistics {
    squared_error: f32,
    squared_ref: f32,
    sum_ref: f32,
    sum_lut: f32,
    count: usize,
}
#[cfg(feature = "reference")]
impl Statistics {
    fn add(&mut self, value: f32, reference: f32) {
        let error = value - reference;
        self.squared_error += error * error;
        self.squared_ref += reference * reference;
        self.sum_ref += reference;
        self.sum_lut += value;
        self.count += 1;
    }
    fn json(&self) -> serde_json::Value {
        if self.count == 0 {
            return serde_json::json!({"pixels":0,"rmse":null,"relative_l2":null,"mean_lut":null,"mean_path_traced":null});
        }
        serde_json::json!({"pixels":self.count,"rmse":(self.squared_error/self.count as f32).sqrt(),
            "relative_l2":(self.squared_error/self.squared_ref.max(1e-30)).sqrt(),
            "mean_lut":self.sum_lut/self.count as f32,"mean_path_traced":self.sum_ref/self.count as f32})
    }
}

#[cfg(feature = "reference")]
fn pixel_overlaps_sun(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    elevation: f32,
    radius: f32,
) -> bool {
    let alpha = radius.to_degrees();
    let low_e = 90.0 - (y + 1) as f32 * 180.0 / height as f32;
    let high_e = 90.0 - y as f32 * 180.0 / height as f32;
    if high_e < elevation - alpha || low_e > elevation + alpha {
        return false;
    }
    if elevation.abs() + alpha >= 90.0 {
        return true;
    }
    let longitude_extent = (radius.sin() / elevation.to_radians().cos())
        .clamp(0.0, 1.0)
        .asin()
        .to_degrees();
    let low_a = x as f32 * 360.0 / width as f32 - 180.0;
    let high_a = (x + 1) as f32 * 360.0 / width as f32 - 180.0;
    high_a >= -longitude_extent && low_a <= longitude_extent
}
