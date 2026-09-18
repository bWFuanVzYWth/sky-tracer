//! Quantify redundant work and readback volume without creating a GPU device.
use sky_atmosphere_lut::{Result, bake_schedule, config::BakeConfig, mapping::Geometry};
use std::{fs, time::Instant};
fn main() -> Result<()> {
    let c = BakeConfig::reference();
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    let start = Instant::now();
    let nodes = bake_schedule::angular_nodes(g, &c);
    let seconds = start.elapsed().as_secs_f32();
    let work = bake_schedule::WorkEstimate::from_nodes(&c, &nodes);
    let value = serde_json::json!({
        "config":c,"dense_states":work.dense_states,"phase_unique_states":work.phase_unique_states,
        "cpu_unique_states":work.cpu_unique_states,"cpu_removed_fraction":1.0-work.cpu_unique_states as f64/c.scattering_len() as f64,
        "guaranteed_phase_removed_fraction":1.0-work.phase_unique_states as f64/c.scattering_len() as f64,
        "coordinate_cache_build_seconds":seconds,"coordinate_cache_bytes":nodes.len()*4,
        "active_states_by_radius":work.active_states_by_radius,"old_statistics_bytes_per_order":work.old_statistics_bytes_per_order,
        "compact_statistics_bytes_per_order":work.compact_statistics_bytes_per_order,"note":"CPU view collapse also needs shader equality; GPU time and adapter-specific sharing are unmeasured."
    });
    fs::create_dir_all("out/lut_v6_design")?;
    fs::write(
        "out/lut_v6_design/work.json",
        serde_json::to_vec_pretty(&value)?,
    )?;
    println!("{value}");
    Ok(())
}
