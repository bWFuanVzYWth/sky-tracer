//! Sparse CPU experiment: march direct sunlight, interpolate only the remainder.
//! Evaluates real grid stencils; it does not allocate/bake an entire new LUT.
use clap::Parser;
use half::bf16;
use rayon::prelude::*;
use serde::Serialize;
use sky_atmosphere_lut::{
    Result,
    asset::{Manifest, fingerprint},
    config::BakeConfig,
    mapping::State,
    model::Model,
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb::rec2020_weights,
    synthesis::SamplePoint,
};
use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};
#[path = "support/frozen_direct.rs"]
mod frozen_direct;
use frozen_direct::direct;

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16)]
    samples_per_region: usize,
    /// Single-band pilots are not RGB quality measurements.
    #[arg(long, value_delimiter = ',', default_value = "17")]
    bands: Vec<usize>,
    #[arg(long)]
    all_bands: bool,
    #[arg(long, default_value_t = 256)]
    steps: usize,
    #[arg(long, default_value_t = 16)]
    threads: usize,
}

#[derive(Clone, Copy, Serialize)]
struct RuntimeSpec {
    steps: usize,
    point_sun: bool,
    log_height: bool,
}

#[derive(Serialize)]
struct Query {
    region: String,
    point: SamplePoint,
}

fn queries(count: usize) -> Result<Vec<Query>> {
    let mut result = Vec::new();
    let old: Vec<SamplePoint> =
        serde_json::from_slice(&fs::read("out/lut_v6_packed_cpu.queries.json")?)?;
    let names = [
        "noon",
        "afternoon",
        "sunset",
        "blue_hour",
        "aircraft",
        "shadow_30km",
        "top_120km",
        "orbit_400km",
        "random",
    ];
    for (j, region) in names.into_iter().enumerate() {
        let begin = j * 8309;
        let end = if j == 8 { old.len() } else { begin + 8309 };
        for k in 0..count {
            result.push(Query {
                region: region.into(),
                point: old
                    [begin + ((2 * k + 1) * (end - begin) / (2 * count)).min(end - begin - 1)],
            });
        }
    }
    let fresh: Vec<SamplePoint> =
        serde_json::from_slice(&fs::read("out/lut_joint16_fresh_queries/queries.json")?)?;
    for (region, begin, end) in [
        ("noon_halo", 0, 4096),
        ("moving_shadow", 4096, 14491),
        ("upper_space", 14491, 18091),
    ] {
        for k in 0..count {
            result.push(Query {
                region: region.into(),
                point: fresh
                    [begin + ((2 * k + 1) * (end - begin) / (2 * count)).min(end - begin - 1)],
            });
        }
    }
    Ok(result)
}

struct Grid {
    name: &'static str,
    config: BakeConfig,
    heights: Vec<usize>,
    stencils: Vec<Option<ReferenceStencil>>,
}
impl Grid {
    fn source_index(&self, mut i: usize, source: [usize; 4]) -> usize {
        let [_, nv, ns, np] = self.config.scattering;
        let p = i % np;
        i /= np;
        let s = i % ns;
        i /= ns;
        let v = i % nv;
        let h = i / nv;
        (((self.heights[h] * source[1] + v) * source[2] + s * (source[2] - 1) / (ns - 1))
            * source[3])
            + p * (source[3] - 1) / (np - 1)
    }
}
fn height_indices(heights: &[f32], count: usize) -> Vec<usize> {
    let mut selected = BTreeSet::from([0, heights.len() - 1]);
    for h in [0.2_f32, 1.0, 2.0, 11.0, 12.0, 35.0] {
        selected.insert(
            (0..heights.len())
                .min_by(|&a, &b| (heights[a] - h).abs().total_cmp(&(heights[b] - h).abs()))
                .unwrap(),
        );
    }
    while selected.len() < count {
        let mut best = (0, 0);
        for i in 0..heights.len() {
            let distance = selected.iter().map(|&s| s.abs_diff(i)).min().unwrap();
            if distance > best.1 {
                best = (i, distance);
            }
        }
        selected.insert(best.0);
    }
    selected.into_iter().collect()
}
fn add(sum: &mut [f32; 3], value: f32, weight: [f32; 3]) {
    for c in 0..3 {
        sum[c] += value * weight[c];
    }
}
fn norm(v: [f32; 3]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples_per_region == 0 || args.steps == 0 || args.threads == 0 {
        return Err("counts must be positive".into());
    }
    if args.out.exists() {
        return Err("output exists; use a new probe directory".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let started = Instant::now();
    let m = Manifest::open(&args.source)?;
    if m.config.scattering != [80, 32, 193, 257] || m.rgb.is_some() {
        return Err("this experiment expects the frozen spectral v6 grid".into());
    }
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != m.model_fingerprint_fnv1a64 {
        return Err("source/model fingerprint mismatch".into());
    }
    let bands: Vec<usize> = if args.all_bands {
        (0..m.bands.len()).collect()
    } else {
        args.bands.clone()
    };
    if bands.is_empty() || bands.iter().any(|&b| b >= m.bands.len()) {
        return Err("invalid bands".into());
    }
    let q = queries(args.samples_per_region)?;
    let g = m.geometry;
    let states: Vec<_> = q
        .iter()
        .map(|q| {
            let [h, mu, mu_s, nu] = q.point.packed()?;
            Ok(g.atmosphere_entry(State {
                altitude_km: h,
                mu,
                mu_s,
                nu,
                ground: g.hits_ground(h, mu),
            })
            .map(|(s, _)| s))
        })
        .collect::<Result<_>>()?;
    let heights = radius_nodes(g, &m.config);
    let mut grids = Vec::new();
    for (name, nh, ns, np) in [
        ("full", 80, 193, 257),
        ("phase65", 80, 193, 65),
        ("phase33", 80, 193, 33),
        ("budget_rgb16", 32, 65, 33),
    ] {
        let hi = height_indices(&heights, nh);
        let mut c = m.config.clone();
        c.scattering = [nh, 32, ns, np];
        c.scattering_altitudes_km = hi.iter().map(|&i| heights[i]).collect();
        let stencils = states
            .iter()
            .map(|s| s.map(|s| ReferenceStencil::new(g, &c, s, Some(&c.scattering_altitudes_km))))
            .collect();
        grids.push(Grid {
            name,
            config: c,
            heights: hi,
            stencils,
        });
    }
    let mut required = BTreeSet::new();
    for grid in &grids {
        for stencil in grid.stencils.iter().flatten() {
            stencil.sample_with(|i| {
                required.insert(grid.source_index(i, m.config.scattering));
                0.0
            });
        }
    }
    let ids: Vec<usize> = required.into_iter().collect();
    let positions: std::collections::HashMap<usize, usize> =
        ids.iter().enumerate().map(|(a, &b)| (b, a)).collect();
    let node_states: Vec<_> = ids.iter().map(|&i| g.state_config(i, &m.config)).collect();
    eprintln!(
        "{} queries, {} unique grid nodes, {} bands",
        q.len(),
        ids.len(),
        bands.len()
    );
    let mut total_nodes = vec![[0.0; 3]; ids.len()];
    let mut remainder_nodes = total_nodes.clone();
    let mut teacher = vec![[0.0; 3]; q.len()];
    let mut single = teacher.clone();
    let mut boundary = teacher.clone();
    let runtime_specs: Vec<_> = [
        (256, false),
        (128, false),
        (64, false),
        (32, false),
        (16, false),
        (64, true),
        (32, true),
        (1024, false),
    ]
    .map(|(steps, point_sun)| RuntimeSpec {
        steps,
        point_sun,
        log_height: false,
    })
    .into_iter()
    .chain(
        [
            (64, false),
            (32, false),
            (16, false),
            (64, true),
            (32, true),
        ]
        .map(|(steps, point_sun)| RuntimeSpec {
            steps,
            point_sun,
            log_height: true,
        }),
    )
    .collect();
    let mut runtime = vec![teacher.clone(); runtime_specs.len()];
    let weights = rec2020_weights(&m);
    let mut node_stats = Vec::new();
    fs::create_dir_all(&args.out)?;
    for &bi in &bands {
        let band_started = Instant::now();
        let lut = m.read_band(&args.source, bi)?;
        let bm = &model.bands[bi];
        let w = if args.all_bands {
            weights[bi]
        } else {
            [1.0, 0.0, 0.0]
        };
        let parts: Vec<_> = node_states
            .par_iter()
            .map(|&s| direct(&lut, bm, s, args.steps, false, false))
            .collect();
        let mut negative = 0;
        let mut worst_negative = 0.0_f32;
        for (j, &i) in ids.iter().enumerate() {
            let rem = lut.radiance[i] - parts[j][0] - parts[j][1];
            if rem < 0.0 {
                negative += 1;
                worst_negative = worst_negative
                    .max(-rem / lut.radiance[i].max(lut.info.solar_irradiance_w_m2 * 1e-12));
            }
            add(&mut total_nodes[j], lut.radiance[i], w);
            // Retain negative subtraction results; do not silently erase the mismatch.
            add(&mut remainder_nodes[j], rem, w);
        }
        let query_parts: Vec<_> = states
            .par_iter()
            .map(|&s| s.map_or([0.0; 2], |s| direct(&lut, bm, s, args.steps, false, false)))
            .collect();
        for j in 0..q.len() {
            let target = grids[0].stencils[j]
                .as_ref()
                .map_or(0.0, |st| st.sample_with(|i| lut.radiance[i]));
            add(&mut teacher[j], target, w);
            add(&mut single[j], query_parts[j][0], w);
            add(&mut boundary[j], query_parts[j][1], w);
        }
        for (k, spec) in runtime_specs.iter().enumerate() {
            let p: Vec<_> = states
                .par_iter()
                .map(|&s| {
                    s.map_or([0.0; 2], |s| {
                        direct(&lut, bm, s, spec.steps, spec.point_sun, spec.log_height)
                    })
                })
                .collect();
            for j in 0..q.len() {
                add(&mut runtime[k][j], p[j][0] + p[j][1], w);
            }
        }
        node_stats.push(serde_json::json!({"band":bi,"nm":bm.info.center_nm,"negative_remainders":negative,
            "worst_negative_fraction_of_total":worst_negative,"seconds":band_started.elapsed().as_secs_f32()}));
        eprintln!(
            "band {} ({:.0} nm): {:.1}s, negative {}/{}",
            bi,
            bm.info.center_nm,
            band_started.elapsed().as_secs_f32(),
            negative,
            ids.len()
        );
        fs::write(
            args.out.join("progress.json"),
            serde_json::to_vec_pretty(&node_stats)?,
        )?;
    }
    let direct_nodes: Vec<[f32; 3]> = total_nodes
        .iter()
        .zip(&remainder_nodes)
        .map(|(l, m)| std::array::from_fn(|c| l[c] - m[c]))
        .collect();
    // bfloat16 is a simple 16-bit scalar encoding with the f32 exponent range.
    // Packed three-channel buffers need 6 bytes/node, with shader bit decoding;
    // this is not a claim about native RGB16 texture formats/filtering.
    let total_bf16: Vec<_> = total_nodes
        .iter()
        .map(|v| v.map(|x| bf16::from_f32(x).to_f32()))
        .collect();
    let remainder_bf16: Vec<_> = remainder_nodes
        .iter()
        .map(|v| v.map(|x| bf16::from_f32(x).to_f32()))
        .collect();
    let mut rows = Vec::new();
    for (j, query) in q.iter().enumerate() {
        let mut candidates = serde_json::Map::new();
        for grid in &grids {
            let interpolate = |values: &[[f32; 3]], clamp: bool| -> [f32; 3] {
                std::array::from_fn(|c| {
                    grid.stencils[j].as_ref().map_or(0.0, |st| {
                        st.sample_with(|i| {
                            let value =
                                values[positions[&grid.source_index(i, m.config.scattering)]][c];
                            if clamp { value.max(0.0) } else { value }
                        })
                    })
                })
            };
            let raw = interpolate(&total_nodes, false);
            let interpolated_direct = interpolate(&direct_nodes, false);
            let raw_bf16 = interpolate(&total_bf16, false);
            let indirect_bf16 = interpolate(&remainder_bf16, false);
            let remainder = interpolate(&remainder_nodes, false);
            let nonnegative = interpolate(&remainder_nodes, true);
            let hybrid: [f32; 3] =
                std::array::from_fn(|c| single[j][c] + boundary[j][c] + remainder[c]);
            let clamped: [f32; 3] =
                std::array::from_fn(|c| single[j][c] + boundary[j][c] + nonnegative[c]);
            let hybrid_bf16: [f32; 3] =
                std::array::from_fn(|c| single[j][c] + boundary[j][c] + indirect_bf16[c]);
            candidates.insert(grid.name.into(),serde_json::json!({"total_lut":raw,"hybrid_signed":hybrid,"hybrid_clamped":clamped,"interpolated_direct":interpolated_direct,
                "total_bf16":raw_bf16,"hybrid_bf16":hybrid_bf16}));
        }
        rows.push(serde_json::json!({"region":query.region,"point":query.point,"teacher":teacher[j],"single":single[j],"boundary":boundary[j],
            "direct_fraction":norm(single[j])/norm(teacher[j]).max(1e-30),"candidates":candidates,
            "runtime_direct":runtime.iter().map(|v|v[j]).collect::<Vec<_>>()}));
    }
    let metadata = serde_json::json!({"kind":"cpu_single_plus_multiple_feasibility_v1","source":args.source,"model":m.model_fingerprint_fnv1a64,
        "source_checksums":m.records.iter().map(|r|r.as_ref().map(|r|&r.checksum_fnv1a64)).collect::<Vec<_>>(),
        "bands":bands,"full_spectral_rec2020":args.all_bands,"steps":args.steps,"sun_samples":m.config.sun_mu*m.config.sun_phi,
        "queries":q.len(),"unique_nodes":ids.len(),"elapsed_seconds":started.elapsed().as_secs_f32(),"node_stats":node_stats,
        "runtime_specs":runtime_specs,"grids":grids.iter().map(|x|serde_json::json!({"name":x.name,"config":x.config,
            "rgb16_radiance_bytes":x.config.scattering_len()*6})).collect::<Vec<_>>(),"rows":rows,
        "limitations":["Sparse feasibility study, not a complete production resource or GPU implementation",
            "bf16 methods include 16-bit node quantization; others are f32. No spectral reduction or optical-depth compression is included",
            "Subtracting independently integrated direct light from the interpolated teacher is a numerical remainder, not separately saved scattering orders",
            "Reference errors remain; this compares against the frozen v6 teacher, not unbiased transport",
            "Finite-distance fog and volume shadows are not evaluated"]});
    fs::write(
        args.out.join("probe.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    // Retain sparse nodes for follow-up CPU quantization without re-integration.
    fs::write(
        args.out.join("node_indices.json"),
        serde_json::to_vec(&ids)?,
    )?;
    fs::write(
        args.out.join("total_nodes.f32"),
        bytemuck::cast_slice(&total_nodes),
    )?;
    fs::write(
        args.out.join("remainder_nodes.f32"),
        bytemuck::cast_slice(&remainder_nodes),
    )?;
    Ok(())
}
