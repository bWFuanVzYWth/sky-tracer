use clap::Parser;
use sky_atmosphere_lut::{Result, four_wave::CpuSource, mapping::State, model::Model};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    resource: PathBuf,
    #[arg(long)]
    audit: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let source = CpuSource::open(&a.resource)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    let models = [7, 13, 20, 27].map(|i| model.bands[i].clone());
    let input: serde_json::Value = serde_json::from_slice(&fs::read(&a.audit)?)?;
    let mut rows = Vec::new();
    for row in input["rows"].as_array().ok_or("missing audit rows")? {
        let h = row["height_km"].as_f64().unwrap() as f32;
        let solar = (row["sun_elevation_deg"].as_f64().unwrap() as f32).to_radians();
        let nm = row["wavelength_nm"].as_f64().unwrap() as f32;
        let k = source
            .resource
            .wavelengths_nm
            .iter()
            .position(|&v| v == nm)
            .ok_or("wavelength mismatch")?;
        for direction in row["directions"].as_array().unwrap() {
            let v: [f32; 3] = serde_json::from_value(direction["view"].clone())?;
            let point = State {
                altitude_km: h,
                mu: v[2],
                mu_s: solar.sin(),
                nu: v[0] * solar.cos() + v[2] * solar.sin(),
                ground: source.resource.geometry.hits_ground(h, v[2]),
            };
            let predicted = source.source(point, &models)[k];
            let expected = direction["source"].as_f64().unwrap() as f32 / 10.0;
            rows.push(serde_json::json!({"height_km":h,"sun_elevation_deg":solar.to_degrees(),"wavelength_nm":nm,"expected":expected,"predicted":predicted,"relative_percent":100.0*(predicted/expected.max(1e-30)-1.0)}));
        }
    }
    fs::write(
        a.out,
        serde_json::to_vec_pretty(
            &serde_json::json!({"kind":"packed_source_vs_direct_angular_cpu","rows":rows}),
        )?,
    )?;
    Ok(())
}
