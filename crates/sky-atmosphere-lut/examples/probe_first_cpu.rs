//! Compare first-order assets and recomputed interpolation nodes, without a GPU.
#[path = "support/first_scattering.rs"]
mod first_scattering;
use clap::Parser;
use sky_atmosphere_lut::{
    Result, asset::Manifest, mapping::State, model::Model, reference_mapping::ReferenceStencil,
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 17)]
    band: usize,
    #[arg(long)]
    cubic_view: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let manifest = Manifest::open(&args.source)?;
    if manifest.config.max_orders != 1 {
        return Err("this diagnostic needs a first-order asset".into());
    }
    let mut lut = manifest.read_band(&args.source, args.band)?;
    if args.cubic_view {
        lut.config.view_interpolation =
            sky_atmosphere_lut::config::ViewInterpolation::MonotoneCubic;
        lut.config.validate(manifest.bands.len())?;
    }
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if manifest.model_fingerprint_fnv1a64 != sky_atmosphere_lut::asset::fingerprint(&model)? {
        return Err("source model differs from probe model".into());
    }
    let mut poses: Vec<[f32; 4]> =
        serde_json::from_slice(&fs::read("scripts/reference_quality_suspect_poses.json")?)?;
    for sun in [0.0, 20.0, 60.0, 85.0, 89.0] {
        for above in [0.05, 1.0, 3.0, 10.0, 20.9544] {
            poses.push([0.2, sun, 0.0, above]);
        }
    }
    let mut cache = HashMap::new();
    let mut rows = Vec::new();
    for [h, sun, az, above] in poses {
        let e = lut.geometry.horizon(h).asin() + above.to_radians();
        let se = sun.to_radians();
        let s = State {
            altitude_km: h,
            mu: e.sin(),
            mu_s: se.sin(),
            nu: e.sin() * se.sin() + e.cos() * se.cos() * az.to_radians().cos(),
            ground: false,
        };
        let Some((s, _)) = lut.geometry.atmosphere_entry(s) else {
            continue;
        };
        let direct = first_scattering::first(&lut, &model.bands[args.band], s);
        let stencil = ReferenceStencil::new(lut.geometry, &lut.config, s, None);
        let stored = stencil.sample_with(|i| lut.radiance[i]);
        let ideal_nodes = stencil.sample_with(|i| {
            *cache.entry(i).or_insert_with(|| {
                first_scattering::first(
                    &lut,
                    &model.bands[args.band],
                    lut.geometry.state_config(i, &lut.config),
                )
            })
        });
        rows.push(serde_json::json!({"h":h,"sun":sun,"az":az,"above":above,"direct":direct,"stored":stored,"ideal_nodes":ideal_nodes,
            "relative_error":if direct>1e-20 {Some(stored/direct-1.0)} else {None},"interpolation_error":if direct>1e-20 {Some(ideal_nodes/direct-1.0)} else {None}}));
    }
    let value = serde_json::json!({"source":args.source,"lookup_override":args.cubic_view,"solver":manifest.solver,"mapping":manifest.coordinate_mapping,"config":lut.config,
        "band_nm":lut.info.center_nm,"oracle_steps":512,"oracle_sun":[8,32],"oracle_nodes":cache.len(),"rows":rows});
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(args.out, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}
