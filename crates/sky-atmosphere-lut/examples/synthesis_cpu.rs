//! Benchmark the complete CPU dataset path, including band I/O and checksums.
use clap::Parser;
use sky_atmosphere_lut::{
    Result,
    synthesis::{SamplePoint, export},
};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 65536)]
    samples: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples == 0 || args.samples > 10_000_000 {
        return Err("samples must be in 1..=10000000".into());
    }
    // Reproducible coverage over four physical variables, with extra low Sun
    // and near-ground samples. This is a throughput set, not an error metric.
    let mut rng = 0x7195_a621_u32;
    let mut random = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng >> 8) as f32 / 16_777_216.0
    };
    let points: Vec<_> = (0..args.samples)
        .map(|i| {
            let h = random();
            let sun = random();
            SamplePoint {
                altitude_km: if i % 8 == 0 {
                    120.0 + 880.0 * h
                } else {
                    120.0 * h.powi(4)
                },
                sun_elevation_deg: if i % 2 == 0 {
                    -18.0 + 30.0 * sun
                } else {
                    -90.0 + 180.0 * sun
                },
                view_elevation_deg: (2.0 * random() - 1.0).asin().to_degrees(),
                relative_azimuth_deg: 180.0 * random(),
            }
        })
        .collect();
    let start = Instant::now();
    export(&args.source, &points, &args.out)?;
    let seconds = start.elapsed().as_secs_f32();
    let result = serde_json::json!({
        "source": args.source,
        "samples": args.samples,
        "elapsed_seconds": seconds,
        "rgb_samples_per_second": args.samples as f32 / seconds,
        "available_cpu_threads": std::thread::available_parallelism().map_or(1, usize::from),
        "includes": "geometry compilation, all spectral files and checksums, spectral interpolation, Rec.2020 integration, output files",
        "note": "CPU only; throughput depends on filesystem cache, query ordering, LUT mapping and concurrent work"
    });
    fs::write(
        args.out.join("timing.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
