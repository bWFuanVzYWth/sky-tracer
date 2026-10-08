//! CPU-only fixed-28 versus f32-saturated inverse experiment; no wgpu objects.
use clap::Parser;
use serde_json::{Value, json};
use sky_realtime::{
    geometry::Geometry,
    mapping::{self, SkyChart},
};
use std::{cell::Cell, f32::consts::PI, fs, hint::black_box, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 256)]
    frames: usize,
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    #[arg(long)]
    bitwise_endpoint: bool,
}
fn inverse_28(f: impl Fn(f32) -> f32, u: f32, mut lo: f32, mut hi: f32) -> f32 {
    if u <= 0.0 {
        return lo;
    }
    if u >= 1.0 {
        return hi;
    }
    for _ in 0..28 {
        let mid = (lo + hi) * 0.5;
        if f(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}
fn inverse_saturated(f: impl Fn(f32) -> f32, u: f32, mut lo: f32, mut hi: f32) -> f32 {
    if u <= 0.0 {
        return lo;
    }
    if u >= 1.0 {
        return hi;
    }
    for _ in 0..28 {
        let mid = (lo + hi) * 0.5;
        let stalled = mid == lo || mid == hi;
        if f(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
        // Apply the original branch before returning: signed zero can otherwise change.
        if stalled {
            break;
        }
    }
    (lo + hi) * 0.5
}
fn inverse_candidate(
    f: impl Fn(f32) -> f32,
    u: f32,
    mut lo: f32,
    mut hi: f32,
    bitwise: bool,
) -> f32 {
    if !bitwise {
        return inverse_saturated(f, u, lo, hi);
    }
    if u <= 0.0 {
        return lo;
    }
    if u >= 1.0 {
        return hi;
    }
    for _ in 0..28 {
        let mid = (lo + hi) * 0.5;
        if mid.to_bits() == lo.to_bits() || mid.to_bits() == hi.to_bits() {
            return mid;
        }
        if f(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}
fn cache(
    g: Geometry,
    h: f32,
    sun: f32,
    size: u32,
    fitted: bool,
    saturated: bool,
    bitwise: bool,
) -> (Vec<[f32; 4]>, [u32; 4]) {
    let sky_rows = size * 3 / 4;
    let lower = if fitted && h >= g.top_height() {
        sky_rows
    } else {
        sky_rows / 2
    };
    let counts = [lower, sky_rows - lower, size - sky_rows];
    let mut coefficients = Vec::new();
    let mut rows = Vec::new();
    for chart in 0..3 {
        let c = SkyChart::new(g, h, sun, chart, fitted);
        coefficients.extend(c.coefficients());
        for y in 0..counts[chart] {
            let u = y as f32 / (counts[chart] - 1) as f32;
            let e = if saturated {
                inverse_candidate(|e| c.forward(e), u, c.bounds[0], c.bounds[1], bitwise)
            } else {
                inverse_28(|e| c.forward(e), u, c.bounds[0], c.bounds[1])
            };
            rows.push([e.sin(), e, 0.0, 0.0]);
        }
    }
    coefficients.extend(rows);
    (coefficients, [size, size, lower, sky_rows])
}
fn stats(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let p = |x: f64| s[((s.len() - 1) as f64 * x).round() as usize];
    json!({"mean_us":v.iter().sum::<f64>()/v.len() as f64,"median_us":p(0.5),"p95_us":p(0.95),"p99_us":p(0.99),"calls":v.len()})
}
fn mismatch(a: &[[f32; 4]], b: &[[f32; 4]]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_zero_requires_more_than_numeric_endpoint_equality() {
        let lo = -f32::from_bits(1);
        let mid = (lo + 0.0) * 0.5;
        assert_eq!(mid.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(mid, 0.0); // A direct numeric early return would produce -0.
        let reference = inverse_28(|_| 0.0, 0.5, lo, 0.0);
        assert_eq!(reference.to_bits(), 0.0_f32.to_bits());
        for bitwise in [false, true] {
            assert_eq!(
                inverse_candidate(|_| 0.0, 0.5, lo, 0.0, bitwise).to_bits(),
                reference.to_bits()
            );
        }
    }
    #[test]
    fn finite_angle_stall_reduces_cdf_calls_without_changing_bits() {
        let lo = 1.0_f32;
        let hi = f32::from_bits(lo.to_bits() + 1);
        let old_calls = Cell::new(0);
        let new_calls = Cell::new(0);
        let old = inverse_28(
            |_| {
                old_calls.set(old_calls.get() + 1);
                0.0
            },
            0.5,
            lo,
            hi,
        );
        let new = inverse_candidate(
            |_| {
                new_calls.set(new_calls.get() + 1);
                0.0
            },
            0.5,
            lo,
            hi,
            true,
        );
        assert_eq!(old.to_bits(), new.to_bits());
        assert_eq!(old_calls.get(), 28);
        assert_eq!(new_calls.get(), 0);
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Options::parse();
    if args.out.exists() || !(16..=4096).contains(&args.frames) || !(1..=32).contains(&args.rounds)
    {
        return Err("new output path and valid frame/round counts required".into());
    }
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    black_box(mapping::sky_cache(g, 0.2, 0.0, 256, true));
    let mut results = Vec::new();
    let mut packed_count = 0;
    let mut mismatch_count = 0;
    let mut production_baseline_mismatches = 0;
    let mut row_inverse_count = 0;
    let mut old_evaluations = 0u64;
    let mut new_evaluations = 0u64;
    let mut saved_histogram = [0u64; 29];
    for (name, h, sun, size) in [
        ("daylight_height_0.2", 0.2, 85.0, 256),
        ("sunset_height_0.2", 0.2, 0.0, 256),
        ("twilight_height_0.2", 0.2, -6.0, 256),
        ("twilight_height_35", 35.0, -6.0, 256),
        ("twilight_height_108", 108.0, -6.0, 256),
        ("space_height_400", 400.0, -6.0, 256),
        ("sunset_sky_128", 0.2, 0.0, 128),
        ("sunset_sky_512", 0.2, 0.0, 512),
    ] {
        let angles: Vec<_> = (0..args.frames)
            .map(|i| (sun - 1.0 + 2.0 * i as f32 / (args.frames - 1) as f32).to_radians())
            .collect();
        let mut stage_old_evals = 0u64;
        let mut stage_new_evals = 0u64;
        for fitted in [false, true] {
            for &angle in &angles {
                let old = cache(g, h, angle, size, fitted, false, args.bitwise_endpoint);
                let new = cache(g, h, angle, size, fitted, true, args.bitwise_endpoint);
                let prod = mapping::sky_cache(g, h, angle, size, fitted);
                assert_eq!(old.1, new.1);
                assert_eq!(old.1, prod.1);
                mismatch_count += mismatch(&old.0, &new.0);
                production_baseline_mismatches += mismatch(&old.0, &prod.0);
                packed_count += old.0.len() * 4;
                let sky_rows = size * 3 / 4;
                let lower = if fitted && h >= g.top_height() {
                    sky_rows
                } else {
                    sky_rows / 2
                };
                let counts = [lower, sky_rows - lower, size - sky_rows];
                for chart in 0..3 {
                    let c = SkyChart::new(g, h, angle, chart, fitted);
                    for y in 0..counts[chart] {
                        let u = y as f32 / (counts[chart] - 1) as f32;
                        let count_old = Cell::new(0u64);
                        let count_new = Cell::new(0u64);
                        let x = inverse_28(
                            |e| {
                                count_old.set(count_old.get() + 1);
                                c.forward(e)
                            },
                            u,
                            c.bounds[0],
                            c.bounds[1],
                        );
                        let z = inverse_candidate(
                            |e| {
                                count_new.set(count_new.get() + 1);
                                c.forward(e)
                            },
                            u,
                            c.bounds[0],
                            c.bounds[1],
                            args.bitwise_endpoint,
                        );
                        assert_eq!(x.to_bits(), z.to_bits());
                        let saved = count_old.get() - count_new.get();
                        saved_histogram[saved as usize] += 1;
                        row_inverse_count += 1;
                        old_evaluations += count_old.get();
                        new_evaluations += count_new.get();
                        if fitted {
                            stage_old_evals += count_old.get();
                            stage_new_evals += count_new.get();
                        }
                    }
                }
            }
        }
        for i in 0..64 {
            black_box(cache(
                black_box(g),
                black_box(h),
                black_box(angles[i % args.frames]),
                black_box(size),
                true,
                false,
                args.bitwise_endpoint,
            ));
            black_box(cache(
                black_box(g),
                black_box(h),
                black_box(angles[i % args.frames]),
                black_box(size),
                true,
                true,
                args.bitwise_endpoint,
            ));
        }
        let mut old_us = Vec::new();
        let mut new_us = Vec::new();
        let mut paired_saved = Vec::new();
        for round in 0..args.rounds {
            for i in 0..args.frames {
                let mut times = [0.0; 2];
                for order in 0..2 {
                    let kind = (order + i + round) % 2;
                    let start = Instant::now();
                    black_box(cache(
                        black_box(g),
                        black_box(h),
                        black_box(angles[i]),
                        black_box(size),
                        true,
                        kind == 1,
                        args.bitwise_endpoint,
                    ));
                    times[kind] = start.elapsed().as_secs_f64() * 1e6;
                }
                old_us.push(times[0]);
                new_us.push(times[1]);
                paired_saved.push(times[0] - times[1]);
            }
        }
        let old_stats = stats(&old_us);
        let new_stats = stats(&new_us);
        let saved =
            old_stats["median_us"].as_f64().unwrap() - new_stats["median_us"].as_f64().unwrap();
        eprintln!(
            "{name}: old {:.3} us, saturated {:.3}, saved {saved:.3}",
            old_stats["median_us"].as_f64().unwrap(),
            new_stats["median_us"].as_f64().unwrap()
        );
        results.push(json!({"trajectory":name,"height_km":h,"sky_size":size,
            "sun_elevation_deg_range":[sun-1.0,sun+1.0],"fixed_28":old_stats,"saturated":new_stats,
            "paired_saved":stats(&paired_saved),"median_difference_us":saved,
            "cdf_eval_count_fixed_28":stage_old_evals,"cdf_eval_count_saturated":stage_new_evals}));
    }
    // Generic edge behavior: endpoints, NaN u/outputs, infinite/overflow ranges, signed/subnormal zero.
    let tiny = f32::from_bits(1);
    let nan = f32::from_bits(0x7fc12345);
    let intervals = [
        (-PI / 2.0, PI / 2.0),
        (0.0, PI / 2.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (-tiny, 0.0),
        (0.0, tiny),
        (-tiny, -0.0),
        (1.0, f32::from_bits(1.0_f32.to_bits() + 1)),
        (f32::from_bits(1.0_f32.to_bits() - 1), 1.0),
        (1e38, 3e38),
        (f32::MAX, f32::MAX),
        (f32::NEG_INFINITY, 0.0),
        (0.0, f32::INFINITY),
        (f32::NEG_INFINITY, f32::INFINITY),
        (nan, 1.0),
        (0.0, nan),
    ];
    let us = [
        f32::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        tiny,
        0.01,
        0.5,
        f32::from_bits(1.0_f32.to_bits() - 1),
        1.0,
        f32::INFINITY,
        nan,
    ];
    let mut edge_count = 0;
    let mut edge_mismatches = 0;
    for (lo, hi) in intervals {
        for u in us {
            for kind in 0..6 {
                let f = |x: f32| match kind {
                    0 => x,
                    1 => 0.0,
                    2 => 1.0,
                    3 => nan,
                    4 => (x + PI / 2.0) / PI,
                    _ => {
                        if x.is_sign_negative() {
                            0.0
                        } else {
                            1.0
                        }
                    }
                };
                let a = inverse_28(f, u, lo, hi);
                let b = inverse_candidate(f, u, lo, hi, args.bitwise_endpoint);
                if a.to_bits() != b.to_bits() {
                    edge_mismatches += 1;
                }
                edge_count += 1;
            }
        }
    }
    let report = json!({"kind":"sky_cpu_inverse_saturation_experiment_v1","gpu_initialized":false,
        "production_mapping_modified":false,"frames_per_trajectory":args.frames,"rounds":args.rounds,
        "equivalence":{"packed_f32_components":packed_count,"packed_bit_mismatches":mismatch_count,
            "production_vs_independent_28loop_bit_mismatches":production_baseline_mismatches,
            "edge_cases":edge_count,"edge_bit_mismatches":edge_mismatches},
        "cdf_work":{"row_inverses":row_inverse_count,"old_evaluations":old_evaluations,
            "saturated_evaluations":new_evaluations,"saved_evaluations_histogram":saved_histogram},
        "results":results,
        "scope":"CPU release, paired old/new per frame with alternating order, black_box inputs/outputs. Timing uses identical fitted SkyChart construction and cache packing; eight fixed-height Sun trajectories. Additional legacy and extreme IEEE754 checks are outside timing. No GPU/Renderer call.",
        "candidate":if args.bitwise_endpoint {"Return mid before CDF evaluation only when its bits equal one endpoint. This distinguishes -0 from +0; matching NaN payloads also stop."}else{"Remember mid==lo||mid==hi before original comparison/update, then break after that update and retain original final midpoint. This preserves signed zero, unlike returning mid before the branch."},
        "bitwise_endpoint_guard":args.bitwise_endpoint,
        "limitations":"Single unpinned process during other machine work. Per-frame timer tails can vary. Finite exhaustive probes are regression evidence, not a proof for every possible Fn or f32 input. CDF evaluation side-effect counts intentionally differ."});
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(args.out, serde_json::to_vec_pretty(&report)?)?;
    if mismatch_count + production_baseline_mismatches + edge_mismatches != 0 {
        return Err("bitwise reference mismatch".into());
    }
    Ok(())
}
