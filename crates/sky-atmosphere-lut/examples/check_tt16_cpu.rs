//! Independent scalar-f32 contraction of stored TT factors versus CPU expansion.
use half::f16;
use sky_atmosphere_lut::Result;
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
fn f32s(p: &Path) -> Result<Vec<f32>> {
    Ok(fs::read(p)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}
fn f16s(p: &Path) -> Result<Vec<f32>> {
    Ok(fs::read(p)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f16::from_le_bytes(b).to_f32())
        .collect())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let dir = Path::new(args.get(1).ok_or("TT directory")?);
    let m: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("candidate.json"))?)?;
    let [nh, nv, ns, nn]: [usize; 4] = serde_json::from_value(m["shape"].clone())?;
    let r = m["rank"].as_u64().ok_or("rank")? as usize;
    let t = m["tail_rank"].as_u64().ok_or("tail rank")? as usize;
    let a = f16s(&dir.join("core_a.f16"))?;
    let b = f16s(&dir.join("core_b.f16"))?;
    let c = f16s(&dir.join("core_c.f16"))?;
    let mean = f32s(&dir.join("mean.f32"))?;
    let scale = f32s(&dir.join("scale.f32"))?;
    let floor = m["transform_floor"].as_f64().ok_or("floor")? as f32;
    let hsv = m["scale_domain"] == "hsv";
    let mut expanded = Vec::new();
    for channel in 0..3 {
        expanded.push(fs::File::open(dir.join(format!("decoded_{channel}.f32")))?);
    }
    let mut errors = Vec::new();
    let mut state = 692384721_u64;
    for _ in 0..2048 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let index = state as usize % (nh * nv * ns * nn);
        let h = index / (nv * ns * nn);
        let v = index / (ns * nn) % nv;
        let s = index / nn % ns;
        let p = index % nn;
        let row = h * ns + s;
        let peak = scale[if hsv { row * nv + v } else { row }];
        let mut middle = vec![0.0_f32; t];
        for j in 0..r {
            for (k, value) in middle.iter_mut().enumerate() {
                *value += a[row * r + j] * b[(j * nv + v) * t + k];
            }
        }
        for (channel, file) in expanded.iter_mut().enumerate() {
            let feature = p * 3 + channel;
            let mut z = 0.0_f32;
            for (k, &value) in middle.iter().enumerate() {
                z += value * c[k * nn * 3 + feature];
            }
            z += mean[v * nn * 3 + feature];
            let value = if m["transform"] == "log1p" {
                z.clamp(0.0, 20.0).exp_m1() * floor * peak
            } else {
                z.min(0.0).exp() * peak
            };
            file.seek(SeekFrom::Start((index * 4) as u64))?;
            let mut bytes = [0; 4];
            file.read_exact(&mut bytes)?;
            let reference = f32::from_le_bytes(bytes);
            let error = (value - reference).abs() / reference.abs().max(peak * 1e-6).max(1e-30);
            if !value.is_finite() || error > 1e-3 {
                return Err(format!("TT expansion mismatch at {index}/{channel}: {value} vs {reference}, normalized error {error}").into());
            }
            errors.push(error);
        }
    }
    errors.sort_unstable_by(f32::total_cmp);
    let report = serde_json::json!({"cpu_only":true,"sampled_nodes":2048,"sampled_channels":errors.len(),
        "independent_scalar_float32_contraction":true,"error_denominator":"max(abs(expanded),peak*1e-6,1e-30)",
        "normalized_difference_p99":errors[errors.len()*99/100],"normalized_difference_max":errors.last(),
        "not_a_gpu_test":true});
    fs::write(
        dir.join("decoder_check.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
