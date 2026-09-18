//! Same-device ABBA measurement of duplicate reuse, with every texel compared.
use sky_atmosphere_lut::{Result, config::BakeConfig, model::Model, solver::GpuBaker};
use std::{fs, path::Path};
fn main() -> Result<()> {
    let scene =
        sky_core::data::load_scene_data(Path::new("data"), 0.0, 0.0).map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let c = BakeConfig {
        max_orders: 2,
        min_orders: 2,
        ..BakeConfig::reference()
    };
    let dense = GpuBaker::new_with_work_reuse(false)?;
    let reuse = GpuBaker::new_with_work_reuse(true)?;
    let mut reference: Option<Vec<f32>> = None;
    let mut runs = Vec::new();
    for enabled in [false, true, true, false] {
        let baker = if enabled { &reuse } else { &dense };
        let band = baker.bake_band(&model, &c, 17, |s| eprintln!("reuse={enabled}: {s}"))?;
        let mut different = 0usize;
        let mut max_abs = 0.0_f32;
        if let Some(values) = &reference {
            for (&a, &b) in band.radiance.iter().zip(values) {
                if a != b {
                    different += 1;
                    max_abs = max_abs.max((a - b).abs());
                }
            }
        } else {
            reference = Some(band.radiance);
        }
        if different != 0 {
            return Err(
                format!("reuse changed {different} texels, max absolute error {max_abs}").into(),
            );
        }
        runs.push(serde_json::json!({"reuse":enabled,"orders":band.orders,"different_texels":different,"max_abs":max_abs}));
    }
    fs::write(
        "out/lut_v6_design/reuse_benchmark.json",
        serde_json::to_vec_pretty(
            &serde_json::json!({"adapter":reuse.adapter_name,"config":c,"runs":runs}),
        )?,
    )?;
    Ok(())
}
