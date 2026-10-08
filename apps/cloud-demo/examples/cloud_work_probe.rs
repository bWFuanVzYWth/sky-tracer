// CPU-only workload witness, using the declared physical model and an
// independent CPU random stream. This is not a bit-exact GPU path replay.
use cloud_pt::{
    config::{Camera, RenderConfig},
    sampling::Pcg32,
    transport::{TransportSettings, trace_sample},
    vdb,
};
use std::{path::Path, time::Instant};

fn main() -> cloud_pt::Result<()> {
    let volume = vdb::load_vdb(
        Path::new("assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb"),
        "density",
    )?;
    let camera = Camera::default();
    let config = RenderConfig::default();
    let mut settings = TransportSettings::default();
    settings.spatial_majorants = false;
    settings.shadow_roulette = false;
    settings.event_limit = 20_000_000; // CPU only; never used to launch a GPU kernel.
    let mut maximum = 0;
    let start = Instant::now();
    println!(
        "sample,collisions,null_collisions,shadow_candidates,surface_events,total_event_lower_bound,seconds"
    );
    for sample in 0..128 {
        let mut rng = Pcg32::for_sample(config.seed, 2081, sample);
        let ray = camera.ray(64, 36, 33.0 + rng.open01(), 32.0 + rng.open01())?;
        let time = Instant::now();
        let result = trace_sample(&volume, ray, &settings, &mut rng)?;
        let count = result.collisions
            + result.null_collisions
            + result.shadow_events
            + result.surface_events;
        maximum = maximum.max(count);
        println!(
            "{sample},{},{},{},{},{count},{:.6}",
            result.collisions,
            result.null_collisions,
            result.shadow_events,
            result.surface_events,
            time.elapsed().as_secs_f64()
        );
        if count > 1_000_000 {
            println!("witness_ray={ray:?}");
            println!("witness_radiance={:?}", result.radiance);
            println!(
                "cpu_completed_path_exceeds_gpu_budget=true; total_seconds={:.6}",
                start.elapsed().as_secs_f64()
            );
            return Ok(());
        }
    }
    println!(
        "no_witness_in_128_independent_cpu_paths; maximum={maximum}; seconds={:.6}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
