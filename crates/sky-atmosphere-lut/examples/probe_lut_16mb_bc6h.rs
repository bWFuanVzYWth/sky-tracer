//! CPU BC6H encoding of a nested reference grid, plus independently decoded
//! f32 files for the existing off-grid CPU validator. No GPU adapter is used.
use half::f16;
use rayon::prelude::*;
use sky_atmosphere_lut::Result;
use std::{fs, path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let source = Path::new(args.get(1).ok_or("source grid")?);
    let output = Path::new(args.get(2).ok_or("output")?);
    let mode = args.get(3).ok_or("linear/sqrt/log mode")?;
    if output.exists() {
        return Err("output exists".into());
    }
    let mut meta: serde_json::Value =
        serde_json::from_slice(&fs::read(source.join("candidate.json"))?)?;
    let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
    let nn = shape[3];
    let rows = shape[..3].iter().product::<usize>();
    let bpr = nn.div_ceil(16);
    let blocks = rows * bpr;
    let budget = blocks * 17 + meta["sun_bytes"].as_u64().unwrap() as usize + 131072;
    if budget > 16_000_000 {
        return Err("BC6H layout exceeds 16 MB".into());
    }
    let mut raw = Vec::new();
    for c in 0..3 {
        raw.push(
            fs::read(source.join(format!("decoded_{c}.f32")))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
        );
    }
    let start = Instant::now();
    let settings = intel_tex_2::bc6h::basic_settings();
    let result: Vec<_> = (0..blocks)
        .into_par_iter()
        .map(|b| {
            let row = b / bpr;
            let begin = b % bpr * 16;
            let values: [f32; 48] = std::array::from_fn(|i| {
                raw[i % 3][row * nn + (begin + i / 3).min(nn - 1)].max(0.0)
            });
            let transformed = values.map(|v| match mode.as_str() {
                "sqrt" => v.sqrt(),
                "log" => v.max(1e-38).log2(),
                _ => v,
            });
            let tag = if mode == "log" {
                transformed
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min)
                    .floor()
                    .clamp(-126.0, 126.0) as i8
            } else {
                (transformed
                    .iter()
                    .copied()
                    .fold(0.0, f32::max)
                    .max(1e-30)
                    .log2()
                    .floor()
                    - 10.0)
                    .clamp(-100.0, 100.0) as i8
            };
            let scale = 2.0_f32.powi(tag as i32);
            let mut rgba = [0u16; 64];
            for i in 0..48 {
                let v = if mode == "log" {
                    transformed[i] - tag as f32 + 1.0
                } else {
                    transformed[i] / scale
                };
                rgba[i / 3 * 4 + i % 3] = f16::from_f32(v).to_bits();
            }
            let surface = intel_tex_2::RgbaSurface {
                width: 4,
                height: 4,
                stride: 32,
                data: bytemuck::cast_slice(&rgba),
            };
            let compressed = intel_tex_2::bc6h::compress_blocks(&settings, &surface);
            let mut decoded = [0.0; 48];
            bcdec_rs::bc6h_float(&compressed, &mut decoded, 12, false);
            for v in &mut decoded {
                *v = if mode == "log" {
                    (*v + tag as f32 - 1.0).exp2()
                } else if mode == "sqrt" {
                    (*v * scale).powi(2)
                } else {
                    *v * scale
                };
            }
            (compressed, tag, decoded)
        })
        .collect();
    fs::create_dir_all(output)?;
    let mut compressed = Vec::with_capacity(blocks * 16);
    let mut tags = Vec::with_capacity(blocks);
    for (bytes, tag, _) in &result {
        compressed.extend(bytes);
        tags.push(*tag as u8);
    }
    fs::write(output.join("radiance.bc6h"), compressed)?;
    fs::write(output.join("block_tag.i8"), tags)?;
    for c in 0..3 {
        let mut decoded = vec![0.0_f32; rows * nn];
        for (b, (_, _, v)) in result.iter().enumerate() {
            let row = b / bpr;
            let begin = b % bpr * 16;
            for i in 0..16 {
                if begin + i < nn {
                    decoded[row * nn + begin + i] = v[i * 3 + c];
                }
            }
        }
        fs::write(
            output.join(format!("decoded_{c}.f32")),
            bytemuck::cast_slice(&decoded),
        )?;
    }
    fs::copy(source.join("sun.f16"), output.join("sun.f16"))?;
    meta["kind"] = "cpu_bc6h_16mb_probe_v1".into();
    meta["gpu_payload_budget_bytes"] = budget.into();
    meta["storage_transform"] = mode.as_str().into();
    meta["encoder"] = "intel_tex_2 basic / CPU".into();
    meta["cpu_seconds"] = start.elapsed().as_secs_f32().into();
    meta["bc6h_bytes"] = (blocks * 16).into();
    meta["block_tag_bytes"] = blocks.into();
    fs::write(
        output.join("candidate.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    println!(
        "{} {} bytes, {:.1} CPU seconds",
        output.display(),
        budget,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
