//! CPU-only observer-coordinate profiling. This executable never creates wgpu objects.
use clap::Parser;
use serde_json::{Value, json};
use sky_realtime::{
    View,
    geometry::Geometry,
    mapping::{self, SkyChart},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    f32::consts::PI,
    fs,
    hint::black_box,
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

struct CountedAllocator;
static COUNT_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT_ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNT_ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

#[derive(Parser)]
struct Options {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 256)]
    frames: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

#[derive(Clone, Copy)]
struct Shape {
    weights: [f32; 4],
    widths: [f32; 3],
}
struct Calibration {
    low: Shape,
    upper: Shape,
    ground: Shape,
    space: Shape,
    near: Shape,
}
fn array<const N: usize>(value: &Value) -> [f32; N] {
    std::array::from_fn(|i| value[i].as_f64().unwrap() as f32)
}
fn calibration() -> &'static Calibration {
    static FIT: OnceLock<Calibration> = OnceLock::new();
    FIT.get_or_init(|| {
        // Use the exact production JSON f64 -> f32 conversion, not a new literal fit.
        let root: Value = serde_json::from_str(mapping::CALIBRATION_JSON).unwrap();
        let shape = |key: &str| Shape {
            weights: array(&root["sky"][key]["weights"]),
            widths: array(&root["sky"][key]["widths"]),
        };
        Calibration {
            low: shape("low"),
            upper: shape("upper"),
            ground: shape("ground"),
            space: shape("space"),
            near: shape("ground_near"),
        }
    })
}
fn typed_chart(g: Geometry, h: f32, sun: f32, chart: usize) -> SkyChart {
    let fit = calibration();
    let hor = g.horizon(h).asin();
    let space = h >= g.top_height();
    let outer = if space {
        -((g.bottom + g.top_height()) / (g.bottom + h))
            .clamp(0.0, 1.0)
            .acos()
    } else {
        hor
    };
    let bounds = match chart {
        0 => [hor, if space { outer } else { 0.0 }],
        1 => [0.0, PI * 0.5],
        _ => [-PI * 0.5, hor],
    };
    let Shape {
        mut weights,
        mut widths,
    } = if chart == 2 {
        fit.ground
    } else if space {
        fit.space
    } else if chart == 1 {
        fit.upper
    } else {
        fit.low
    };
    if chart == 2 && !space {
        let smooth = |x: f32| {
            let x = x.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let t = 1.0 - smooth((h - 1.0) / 3.0);
        for j in 0..4 {
            weights[j] = weights[j] * (1.0 - t) + fit.near.weights[j] * t;
        }
        for j in 0..3 {
            widths[j] = widths[j] * (1.0 - t) + fit.near.widths[j] * t;
        }
    }
    SkyChart {
        bounds,
        centers: [hor, sun, outer],
        weights,
        widths,
        fitted: true,
    }
}

/// Experimental only: invariant terms keep the production operation order.
struct PreparedChart<'a> {
    chart: &'a SkyChart,
    span: f32,
    outside: [bool; 3],
    begin: [f32; 3],
    denominator: [f32; 3],
    outside_numerator: [f32; 3],
}
impl<'a> PreparedChart<'a> {
    fn new(chart: &'a SkyChart) -> Self {
        let [lo, hi] = chart.bounds;
        let mut result = Self {
            chart,
            span: (hi - lo).max(1e-10),
            outside: [false; 3],
            begin: [0.0; 3],
            denominator: [0.0; 3],
            outside_numerator: [0.0; 3],
        };
        for j in 0..3 {
            result.outside[j] = chart.fitted && (chart.centers[j] < lo || chart.centers[j] > hi);
            let f = |d: f32| {
                if chart.fitted {
                    d / (d.abs() + chart.widths[j])
                } else {
                    (d / chart.widths[j]).atan()
                }
            };
            let a = f(lo - chart.centers[j]);
            let b = f(hi - chart.centers[j]);
            result.begin[j] = a;
            result.denominator[j] = (b - a).max(1e-10);
            result.outside_numerator[j] = (hi - chart.centers[j]).abs() + chart.widths[j];
        }
        result
    }
    fn forward(&self, e: f32) -> f32 {
        let c = self.chart;
        let [lo, hi] = c.bounds;
        let e = e.clamp(lo, hi);
        let mut y = c.weights[0] * (e - lo) / self.span;
        for j in 0..3 {
            if self.outside[j] {
                y += c.weights[j + 1] * (e - lo) / self.span * self.outside_numerator[j]
                    / ((e - c.centers[j]).abs() + c.widths[j]);
                continue;
            }
            let d = e - c.centers[j];
            let f = if c.fitted {
                d / (d.abs() + c.widths[j])
            } else {
                (d / c.widths[j]).atan()
            };
            y += c.weights[j + 1] * (f - self.begin[j]) / self.denominator[j];
        }
        y.clamp(0.0, 1.0)
    }
}
fn row_counts(g: Geometry, h: f32, size: u32) -> ([u32; 3], [u32; 4]) {
    let sky = size * 3 / 4;
    let lower = if h >= g.top_height() { sky } else { sky / 2 };
    ([lower, sky - lower, size - sky], [size, size, lower, sky])
}
fn charts(g: Geometry, h: f32, sun: f32, typed: bool) -> [SkyChart; 3] {
    std::array::from_fn(|i| {
        if typed {
            typed_chart(g, h, sun, i)
        } else {
            SkyChart::new(g, h, sun, i, true)
        }
    })
}
fn candidate_cache(
    g: Geometry,
    h: f32,
    sun: f32,
    size: u32,
    typed: bool,
    prepared: bool,
) -> (Vec<[f32; 4]>, [u32; 4]) {
    let (counts, layout) = row_counts(g, h, size);
    let cs = charts(g, h, sun, typed);
    let mut coefficients = Vec::new();
    let mut rows = Vec::new();
    for chart in 0..3 {
        let c = &cs[chart];
        let fast = PreparedChart::new(c);
        coefficients.extend(c.coefficients());
        for y in 0..counts[chart] {
            let u = y as f32 / (counts[chart] - 1) as f32;
            let e = if prepared {
                mapping::inverse(|e| fast.forward(e), u, c.bounds[0], c.bounds[1])
            } else {
                mapping::inverse(|e| c.forward(e), u, c.bounds[0], c.bounds[1])
            };
            rows.push([e.sin(), e, 0.0, 0.0]);
        }
    }
    coefficients.extend(rows);
    (coefficients, layout)
}
fn inverse_rows(
    g: Geometry,
    h: f32,
    size: u32,
    cs: &[SkyChart; 3],
    prepared: bool,
) -> Vec<[f32; 4]> {
    let (counts, _) = row_counts(g, h, size);
    let mut rows = Vec::new();
    for chart in 0..3 {
        let c = &cs[chart];
        let fast = PreparedChart::new(c);
        for y in 0..counts[chart] {
            let u = y as f32 / (counts[chart] - 1) as f32;
            let e = if prepared {
                mapping::inverse(|e| fast.forward(e), u, c.bounds[0], c.bounds[1])
            } else {
                mapping::inverse(|e| c.forward(e), u, c.bounds[0], c.bounds[1])
            };
            rows.push([e.sin(), e, 0.0, 0.0]);
        }
    }
    rows
}
fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    json!({"mean_us":values.iter().sum::<f64>() / values.len() as f64,
        "median_us":percentile(0.5),"p95_us":percentile(0.95),"p99_us":percentile(0.99),
        "minimum_us":sorted[0],"maximum_us":sorted[sorted.len()-1],"calls":values.len()})
}
fn allocation_count<T>(f: impl FnOnce() -> T) -> Value {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    COUNT_ENABLED.store(true, Ordering::Relaxed);
    black_box(f());
    COUNT_ENABLED.store(false, Ordering::Relaxed);
    json!({"allocator_calls_per_frame":ALLOCATIONS.load(Ordering::Relaxed),
        "requested_allocation_bytes_per_frame":ALLOCATED_BYTES.load(Ordering::Relaxed)})
}
fn measure<T>(frames: usize, rounds: usize, mut f: impl FnMut(usize) -> T) -> Value {
    for i in 0..64 {
        black_box(f(i % frames));
    }
    let allocations = allocation_count(|| f(0));
    let mut us = Vec::with_capacity(frames * rounds);
    for _ in 0..rounds {
        for i in 0..frames {
            let start = Instant::now();
            black_box(f(i));
            us.push(start.elapsed().as_secs_f64() * 1e6);
        }
    }
    json!({"time":stats(&us),"allocations":allocations})
}
fn view_and_parameters(h: f32, sun: f32, size: u32) -> ([f32; 4], [f32; 4], [u32; 4]) {
    // Arithmetic portion of Renderer::render params closure only. No GPU upload/encoding.
    let view = View {
        yaw_deg: 12.0,
        pitch_deg: 4.0,
        fov_y_deg: 60.0,
        sun_azimuth_deg: 31.0,
        sun_elevation_deg: sun.to_degrees(),
        altitude_km: h,
    };
    let sky = size * 3 / 4;
    let lower = if h >= 120.0 { sky } else { sky / 2 };
    (
        [
            view.yaw_deg.to_radians(),
            view.pitch_deg.to_radians(),
            view.fov_y_deg.to_radians(),
            h.max(0.0),
        ],
        [
            view.sun_azimuth_deg.to_radians(),
            view.sun_elevation_deg.to_radians(),
            0.0,
            6360.0,
        ],
        [size, size, lower, sky],
    )
}
fn compare_rows(a: &[[f32; 4]], b: &[[f32; 4]]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Options::parse();
    if args.out.exists() || !(16..=4096).contains(&args.frames) || !(1..=32).contains(&args.rounds)
    {
        return Err("requires a new output file, frames in [16,4096], rounds in [1,32]".into());
    }
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    let cold = Instant::now();
    black_box(mapping::sky_cache(g, 0.2, 0.0, 256, true));
    let cold_us = cold.elapsed().as_secs_f64() * 1e6;
    black_box(calibration());
    let mut results = Vec::new();
    let mut typed_mismatches = 0;
    let mut prepared_mismatches = 0;
    let mut checked_components = 0;
    for (name, h, sun_center_deg, size) in [
        ("daylight_height_0.2", 0.2, 85.0, 256),
        ("sunset_height_0.2", 0.2, 0.0, 256),
        ("twilight_height_0.2", 0.2, -6.0, 256),
        ("twilight_height_35", 35.0, -6.0, 256),
        ("twilight_height_108", 108.0, -6.0, 256),
        ("space_height_400", 400.0, -6.0, 256),
        ("sunset_sky_128", 0.2, 0.0, 128),
        ("sunset_sky_512", 0.2, 0.0, 512),
    ] {
        let suns: Vec<_> = (0..args.frames)
            .map(|i| {
                (sun_center_deg - 1.0 + 2.0 * i as f32 / (args.frames - 1) as f32).to_radians()
            })
            .collect();
        for &sun in &suns {
            let original = mapping::sky_cache(g, h, sun, size, true);
            let typed = candidate_cache(g, h, sun, size, true, false);
            let fast = candidate_cache(g, h, sun, size, false, true);
            assert_eq!(original.1, typed.1);
            assert_eq!(original.1, fast.1);
            typed_mismatches += compare_rows(&original.0, &typed.0);
            prepared_mismatches += compare_rows(&original.0, &fast.0);
            checked_components += original.0.len() * 4;
        }
        let chart_samples: Vec<_> = suns.iter().map(|&s| charts(g, h, s, false)).collect();
        let production = measure(args.frames, args.rounds, |i| {
            mapping::sky_cache(g, h, suns[i], size, true)
        });
        let stage_charts = measure(args.frames, args.rounds, |i| {
            let c = charts(g, h, suns[i], false);
            let packed: [[[f32; 4]; 4]; 3] = std::array::from_fn(|j| c[j].coefficients());
            (c, packed)
        });
        let typed_charts = measure(args.frames, args.rounds, |i| charts(g, h, suns[i], true));
        let stage_inverse = measure(args.frames, args.rounds, |i| {
            inverse_rows(g, h, size, &chart_samples[i], false)
        });
        let typed = measure(args.frames, args.rounds, |i| {
            candidate_cache(g, h, suns[i], size, true, false)
        });
        let prepared = measure(args.frames, args.rounds, |i| {
            candidate_cache(g, h, suns[i], size, false, true)
        });
        let frame_params = measure(args.frames, args.rounds, |i| {
            view_and_parameters(h, suns[i], size)
        });
        eprintln!(
            "{name}: production {:.3} us, typed {:.3}, invariant-CDF {:.3}",
            production["time"]["median_us"].as_f64().unwrap(),
            typed["time"]["median_us"].as_f64().unwrap(),
            prepared["time"]["median_us"].as_f64().unwrap()
        );
        results.push(json!({"trajectory":name,"height_km":h,"sun_elevation_deg_range":[sun_center_deg-1.0,sun_center_deg+1.0],
            "sky_size":size,"production_sky_cache":production,"chart_json_coefficients":stage_charts,
            "typed_chart_construction":typed_charts,"inverse_rows_only":stage_inverse,
            "experimental_typed_cache":typed,"experimental_invariant_cdf_cache":prepared,
            "frame_parameter_arithmetic_only":frame_params}));
    }
    // Additional altitude/angle extremes and legacy chart branches, outside timed paths.
    let mut forward_mismatches = 0;
    for fitted in [false, true] {
        for h in [
            0.0, 0.2, 1.0, 2.0, 4.0, 35.0, 108.0, 119.99, 120.0, 400.0, 36000.0,
        ] {
            for sun_deg in [-90.0_f32, -18.0, -6.0, 0.0, 6.0, 85.0, 90.0] {
                for chart in 0..3 {
                    let c = SkyChart::new(g, h, sun_deg.to_radians(), chart, fitted);
                    let p = PreparedChart::new(&c);
                    for i in 0..257 {
                        let e = c.bounds[0] + (c.bounds[1] - c.bounds[0]) * i as f32 / 256.0;
                        if c.forward(e).to_bits() != p.forward(e).to_bits() {
                            forward_mismatches += 1;
                        }
                    }
                }
            }
        }
    }
    let report = json!({"kind":"sky_cpu_observer_mapping_profile_v1","gpu_initialized":false,
        "production_mapping_modified":false,"geometry":g,"cold_first_sky_cache_us":cold_us,
        "frames_per_trajectory":args.frames,"rounds":args.rounds,"results":results,
        "equivalence":{"checked_packed_components":checked_components,
            "typed_calibration_cache_bit_mismatches":typed_mismatches,
            "invariant_cdf_cache_bit_mismatches":prepared_mismatches,
            "additional_fitted_and_legacy_forward_bit_mismatches":forward_mismatches},
        "scope":"Release CPU-only microbenchmark of public mapping calls. Per-call wall timers include allocation/deallocation and black_box; production cache lookup cold start is reported separately. Fixed-height Sun trajectories repeat each round; there is no GPU initialization/upload/encoding/readback, no Renderer::render call, and no screen-size dependency. Experimental implementations live only in this example.",
        "limitations":"Single-process non-pinned CPU timing; concurrent machine work can affect tails. Allocator counting is disabled for timed calls; its enabled check remains in the allocator wrapper. Arithmetic preparation mirror excludes uniform struct copying and wgpu operations, whose actual full-path costs belong to the GPU profiling command."});
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    if typed_mismatches + prepared_mismatches + forward_mismatches != 0 {
        return Err("experimental cache changes were not bitwise equivalent".into());
    }
    Ok(())
}
