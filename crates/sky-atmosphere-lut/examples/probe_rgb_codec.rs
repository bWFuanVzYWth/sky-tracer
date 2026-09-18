//! CPU block-layout/codec experiment; never creates a GPU adapter.
use half::f16;
use rayon::prelude::*;
use sky_atmosphere_lut::{Result, asset::Manifest, rgb};
use std::{path::PathBuf, time::Instant};

fn stats(mut v: Vec<f32>) -> serde_json::Value {
    v.sort_unstable_by(f32::total_cmp);
    serde_json::json!({"p50":v[v.len()/2],"p95":v[v.len()*95/100],"p99":v[v.len()*99/100],"max":v[v.len()-1]})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).ok_or("source directory required")?);
    let output = PathBuf::from(args.get(2).ok_or("output JSON required")?);
    let m = Manifest::open(&dir)?;
    let mut channels = Vec::new();
    for i in 0..3 {
        channels.push(rgb::read_channel(&m, &dir, i)?.1);
        eprintln!("read channel {i}");
    }
    let [nr, nm, ns, nn] = m.config.scattering;
    let range: Vec<_> = channels
        .iter()
        .map(|v| {
            serde_json::json!({
                "min":v.iter().copied().fold(f32::INFINITY,f32::min),
                "max":v.iter().copied().fold(f32::NEG_INFINITY,f32::max),
                "negative":v.iter().filter(|&&x|x<0.0).count(),
                "zero":v.iter().filter(|&&x|x==0.0).count(),
            })
        })
        .collect();
    let mut results = Vec::new();
    // Same stratified random states for all trials, including all altitudes.
    for (label, phase, view, sun) in [
        ("phase16", 16, 1, 1),
        ("phase4_view4", 4, 4, 1),
        ("phase4_sun4", 4, 1, 4),
    ] {
        let blocks: Vec<[f32; 48]> = (0..32768_u64)
            .map(|i| {
                let mut seed = i.wrapping_mul(6364136223846793005).wrapping_add(19);
                let mut next = |n: usize| {
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    (seed.wrapping_mul(2685821657736338717) % (n as u64)) as usize
                };
                let r = next(nr);
                let v = next(nm.div_ceil(view)) * view;
                let s = next(ns.div_ceil(sun)) * sun;
                let n = next(nn.div_ceil(phase)) * phase;
                std::array::from_fn(|k| {
                    let p = k / 3;
                    let index = ((r * nm + (v + p / phase % view).min(nm - 1)) * ns
                        + (s + p / (phase * view)).min(ns - 1))
                        * nn
                        + (n + p % phase).min(nn - 1);
                    channels[k % 3][index]
                })
            })
            .collect();
        for (codec, settings) in [
            ("bc6h_fast", Some(intel_tex_2::bc6h::fast_settings())),
            ("bc6h_slow", Some(intel_tex_2::bc6h::slow_settings())),
            ("scaled_f16", None),
        ] {
            let start = Instant::now();
            let errors: Vec<(f32, f32)> = blocks
                .par_iter()
                .map(|src| {
                    let max = src.iter().copied().fold(0.0, f32::max);
                    let exponent = if max > 0.0 {
                        max.log2().floor() as i32 - 10
                    } else {
                        0
                    };
                    let scale = 2.0_f32.powi(exponent.clamp(-100, 100));
                    let mut dst = [0.0; 48];
                    if let Some(settings) = &settings {
                        let mut rgba = [0u16; 64];
                        for k in 0..48 {
                            rgba[k / 3 * 4 + k % 3] =
                                f16::from_f32((src[k] / scale).max(0.0)).to_bits();
                        }
                        let surface = intel_tex_2::RgbaSurface {
                            width: 4,
                            height: 4,
                            stride: 32,
                            data: bytemuck::cast_slice(&rgba),
                        };
                        let compressed = intel_tex_2::bc6h::compress_blocks(settings, &surface);
                        bcdec_rs::bc6h_float(&compressed, &mut dst, 12, false);
                        for x in &mut dst {
                            *x *= scale;
                        }
                    } else {
                        for k in 0..48 {
                            dst[k] = f16::from_f32(src[k] / scale).to_f32() * scale;
                        }
                    }
                    let mut peak = 0.0_f32;
                    let mut channel = 0.0_f32;
                    for p in 0..16 {
                        let a = &src[p * 3..p * 3 + 3];
                        let b = &dst[p * 3..p * 3 + 3];
                        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-20);
                        let diff = a
                            .iter()
                            .zip(b)
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum::<f32>()
                            .sqrt();
                        peak = peak.max(diff / norm);
                        for c in 0..3 {
                            channel = channel
                                .max((a[c] - b[c]).abs() / a[c].abs().max(norm * 0.01).max(1e-20));
                        }
                    }
                    (peak, channel)
                })
                .collect();
            let count = nr * nm.div_ceil(view) * ns.div_ceil(sun) * nn.div_ceil(phase);
            let escapes: Vec<_> = [0.005,0.01,0.02,0.05].iter().map(|&t| {
                let bad=errors.iter().filter(|&&(_,c)|c>t).count();
                serde_json::json!({"threshold":t,"fraction":bad as f32/errors.len() as f32,"estimated_bc_plus_f32_bytes":count*20+(count as f32*bad as f32/errors.len() as f32*192.0) as usize})
            }).collect();
            let result = serde_json::json!({"layout":label,"codec":codec,"blocks":errors.len(),"seconds":start.elapsed().as_secs_f32(),"block_worst_rgb_relative":stats(errors.iter().map(|x|x.0).collect()),"block_worst_channel_relative_floor_1percent_rgb_norm":stats(errors.iter().map(|x|x.1).collect()),"escapes":escapes});
            eprintln!("{result}");
            results.push(result);
        }
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"source":dir,"range":range,"experiments":results}),
        )?,
    )?;
    Ok(())
}
