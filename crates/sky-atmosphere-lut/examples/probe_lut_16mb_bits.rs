//! CPU precision/size sweep of block deltas, on a resampled reference grid.
//! Keeps all exponent bits. Files decoded_*.f32 are audit intermediates only.
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result,
    packed::{decode_block, encode_block},
};
use std::{collections::HashMap, fs, path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let source = Path::new(args.get(1).ok_or("source")?);
    let output = Path::new(args.get(2).ok_or("output")?);
    let mantissa: u32 = args.get(3).ok_or("mantissa bits")?.parse()?;
    if !(4..=11).contains(&mantissa) || output.exists() {
        return Err("invalid precision or existing output".into());
    }
    let shift = 23 - mantissa;
    let mut meta: serde_json::Value =
        serde_json::from_slice(&fs::read(source.join("candidate.json"))?)?;
    let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
    let count = shape.iter().product::<usize>();
    let mut raw = Vec::new();
    for c in 0..3 {
        raw.push(
            fs::read(source.join(format!("decoded_{c}.f32")))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect::<Vec<_>>(),
        );
    }
    let start = Instant::now();
    // Reuse the proven 20-bit integer block codec. Place the shorter integer
    // code in its mantissa-11 carrier; all carrier bit patterns are finite and
    // exact, including codes that originally represented negative radiance.
    let blocks: Vec<_> = (0..count.div_ceil(16))
        .into_par_iter()
        .map(|b| {
            let carrier: Vec<[f32; 3]> = (0..16)
                .map(|j| {
                    std::array::from_fn(|c| {
                        let bits = raw[c][(b * 16 + j).min(count - 1)].to_bits();
                        let q = bits.wrapping_add(1 << (shift - 1)) >> shift;
                        let q = if f32::from_bits(q << shift).is_finite() {
                            q
                        } else {
                            bits >> shift
                        };
                        f32::from_bits(q << 12)
                    })
                })
                .collect();
            encode_block(&carrier)
        })
        .collect();
    let mut map = Vec::new();
    let mut data = Vec::new();
    let mut dictionary = HashMap::new();
    for block in blocks {
        let offset = if let Some(&o) = dictionary.get(&block) {
            o
        } else {
            let o = data.len() as u32;
            data.extend_from_slice(&block);
            dictionary.insert(block, o);
            o
        };
        map.push(offset);
    }
    fs::create_dir_all(output)?;
    fs::write(output.join("blocks.u32"), bytemuck::cast_slice(&map))?;
    fs::write(output.join("radiance.u32"), bytemuck::cast_slice(&data))?;
    for (c, channel) in raw.iter().enumerate() {
        let decoded: Vec<f32> = (0..count)
            .map(|i| {
                let q = decode_block(&data[map[i / 16] as usize..], 16, i % 16, c).to_bits() >> 12;
                let value = f32::from_bits(q << shift);
                assert!(value.is_finite());
                let original = channel[i].to_bits();
                let expected = original.wrapping_add(1 << (shift - 1)) >> shift;
                let expected = if f32::from_bits(expected << shift).is_finite() {
                    expected
                } else {
                    original >> shift
                };
                assert_eq!(
                    q, expected,
                    "integer codec must preserve the quantized code exactly"
                );
                value
            })
            .collect();
        fs::write(
            output.join(format!("decoded_{c}.f32")),
            bytemuck::cast_slice(&decoded),
        )?;
    }
    fs::copy(source.join("sun.f16"), output.join("sun.f16"))?;
    let budget =
        (map.len() + data.len()) * 4 + meta["sun_bytes"].as_u64().unwrap() as usize + 131072;
    meta["kind"] = "cpu_nested_grid_block_mantissa_probe_v1".into();
    meta["gpu_payload_budget_bytes"] = budget.into();
    meta["mantissa_bits"] = mantissa.into();
    meta["map_bytes"] = (map.len() * 4).into();
    meta["data_bytes"] = (data.len() * 4).into();
    meta["within_16mb_budget"] = (budget <= 16_000_000).into();
    meta["block_texels"] = 16.into();
    meta["cpu_seconds"] = start.elapsed().as_secs_f32().into();
    fs::write(
        output.join("candidate.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    println!(
        "{} mantissa {}: {} bytes, {:.2} s",
        output.display(),
        mantissa,
        budget,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
