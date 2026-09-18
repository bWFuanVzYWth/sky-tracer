//! Per-order local diagnostics. GPU computes all transport; CPU only samples
//! the cumulative results at identical world-space directions after each order.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    asset::{Manifest, fingerprint},
    config::BakeConfig,
    mapping::{State, sample_radiance_state, scattering_cosine, unit},
    model::Model,
    solver::{BakedBand, GpuBaker},
};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 17)]
    band: usize,
    #[arg(long)]
    asset: Option<PathBuf>,
    /// Save exact 1/2/4-order prefixes for paired truncated-PT comparisons.
    #[arg(long)]
    prefixes: Option<PathBuf>,
}
fn direction(e: f32, a: f32) -> Vec3 {
    let e = e.to_radians();
    let a = a.to_radians();
    Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
}
fn main() -> sky_atmosphere_lut::Result<()> {
    let args = Args::parse();
    let config: BakeConfig = serde_json::from_slice(&fs::read(&args.config)?)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let g = model.geometry;
    let mut poses = Vec::new();
    for solar in [-6.0_f32, -4.0, 0.0, 20.0, 90.0] {
        for (elevation, azimuth) in [(2.0, 0.0), (10.0, 180.0), (30.0, 180.0), (90.0, 0.0)] {
            poses.push((solar, elevation, azimuth));
        }
    }
    for delta in [
        -10.0_f32, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0,
    ] {
        poses.push((20.0, 20.0 + delta, 0.0));
    }
    let states: Vec<_> = poses
        .iter()
        .map(|&(solar, elevation, azimuth)| {
            let ray = direction(elevation, azimuth);
            let sun = direction(solar, 0.0);
            State {
                altitude_km: 0.2,
                mu: ray.z,
                mu_s: sun.z,
                nu: ray.dot(sun),
                ground: g.hits_ground(0.2, ray.z),
            }
        })
        .collect();
    let nodes: Vec<_> = (0..config.scattering[3])
        .map(|i| scattering_cosine(unit(i, config.scattering[3])))
        .collect();
    let gpu = GpuBaker::new()?;
    let mut orders = Vec::new();
    let mut prefixes = Vec::new();
    let band = gpu.bake_band_observed(
        &model,
        &config,
        args.band,
        |s| eprintln!("{s}"),
        |order, values| {
            let radiance: Vec<_> = states
                .iter()
                .map(|&s| sample_radiance_state(values, g, &config, s, &nodes))
                .collect();
            orders.push(serde_json::json!({"order":order,"radiance":radiance}));
            if args.prefixes.is_some() && [1, 2, 4].contains(&order) {
                prefixes.push((order, values.to_vec()));
            }
        },
    )?;
    if let Some(root) = &args.prefixes {
        for (order, mut radiance) in prefixes {
            let path = root.join(format!("order_{order}"));
            if path.exists() {
                return Err("prefix asset already exists".into());
            }
            fs::create_dir_all(&path)?;
            let ground_irradiance = radiance.split_off(config.scattering_len());
            let mut c = config.clone();
            c.max_orders = order;
            c.min_orders = c.min_orders.min(order);
            let mut manifest = Manifest::new(&model, c, gpu.adapter_name.clone())?;
            manifest.write_band(
                &path,
                args.band,
                BakedBand {
                    optical_depth: band.optical_depth.clone(),
                    radiance,
                    ground_irradiance,
                    orders: band.orders[..order].to_vec(),
                    stopped_by_tolerance: false,
                },
            )?;
        }
    }
    if let Some(path) = &args.asset {
        if path.exists() {
            return Err("diagnostic asset already exists".into());
        }
        fs::create_dir_all(path)?;
        let mut manifest = Manifest::new(&model, config.clone(), gpu.adapter_name.clone())?;
        manifest.write_band(path, args.band, band)?;
    }
    let report = serde_json::json!({"config":config,"band":args.band,"model_fingerprint_fnv1a64":fingerprint(&model)?,"poses_sun_view_azimuth_deg":poses,"orders":orders});
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(args.out, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
