//! CPU packing of an existing RGB grid. Keep the f32 exponent and 11 mantissa
//! bits, then bit-pack block-local integer differences and share identical blocks.
//! No change to grid coordinates, interpolation, or transport. Decode corners
//! before interpolation; this is not hardware filtering of encoded radiance.
use crate::{
    Result,
    asset::{Hash, Manifest},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path, time::Instant};

pub const FORMAT: &str = "rgb-f32exp-m11-block-delta-v1";
const SHIFT: u32 = 12;
const BASE_BITS: u32 = 20;
const BASE_MASK: u32 = (1 << BASE_BITS) - 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedStorage {
    pub format: String,
    pub block_texels: usize,
    pub map_words: usize,
    pub data_words: usize,
    pub unique_blocks: usize,
    pub map_checksum: String,
    pub data_checksum: String,
    pub sun_checksum: String,
    /// Provenance only: the original spectral tau remains in the source asset.
    pub source_rgb: String,
}
impl PackedStorage {
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        if self.format != FORMAT
            || ![16, 32, 64].contains(&self.block_texels)
            || self.map_words != m.config.scattering_len().div_ceil(self.block_texels)
            || self.unique_blocks == 0
            || self.unique_blocks > self.map_words
            || self.data_words < self.unique_blocks * 3
            || self.data_words > self.map_words * (3 + (3 * self.block_texels * 20).div_ceil(32))
            || self.data_words > u32::MAX as usize
            || [&self.map_checksum, &self.data_checksum, &self.sun_checksum]
                .iter()
                .any(|s| s.len() != 16 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err("invalid packed RGB metadata".into());
        }
        Ok(())
    }
    pub fn gpu_bytes(&self, m: &Manifest) -> u64 {
        ((self.map_words + self.data_words + 3 * m.config.optical_depth_len()) * 4) as u64
    }
}
pub struct PackedLut {
    pub map: Vec<u32>,
    pub data: Vec<u32>,
    pub sun: [Vec<f32>; 3],
    pub block_texels: usize,
}

/// Round magnitude, preserving signs, exact zero and the f32 exponent range.
/// Except saturation next to f32::MAX, normal values have at most 1/4096
/// relative rounding error. Subnormals have at most 2048 f32 subnormal units
/// of absolute error (~2.87e-42). Saturation stays within 1/2048 relative error.
pub fn quantize(value: f32) -> u32 {
    let bits = value.to_bits();
    let q = bits.wrapping_add(1 << (SHIFT - 1)) >> SHIFT;
    if f32::from_bits(q << SHIFT).is_finite() {
        q
    } else {
        bits >> SHIFT
    }
}

pub fn encode_block(values: &[[f32; 3]]) -> Vec<u32> {
    let n = values.len();
    let q: Vec<[u32; 3]> = values.iter().map(|v| v.map(quantize)).collect();
    let base: [u32; 3] = std::array::from_fn(|c| q.iter().map(|v| v[c]).min().unwrap());
    let widths: [u32; 3] = std::array::from_fn(|c| {
        32 - (q.iter().map(|v| v[c]).max().unwrap() - base[c]).leading_zeros()
    });
    let mut words = vec![0u32; 3 + (n * widths.iter().sum::<u32>() as usize).div_ceil(32)];
    let mut bit = 0;
    for c in 0..3 {
        words[c] = base[c] | (widths[c] << BASE_BITS);
        for value in &q {
            if widths[c] > 0 {
                let d = value[c] - base[c];
                let w = 3 + bit / 32;
                let shift = bit % 32;
                words[w] |= d << shift;
                if shift + widths[c] as usize > 32 {
                    words[w + 1] |= d >> (32 - shift);
                }
            }
            bit += widths[c] as usize;
        }
    }
    words
}

pub fn decode_block(words: &[u32], n: usize, index: usize, channel: usize) -> f32 {
    let header = words[channel];
    let width = header >> BASE_BITS;
    let base = header & BASE_MASK;
    if width == 0 {
        return f32::from_bits(base << SHIFT);
    }
    let bit = n * words[..channel]
        .iter()
        .map(|h| (h >> BASE_BITS) as usize)
        .sum::<usize>()
        + index * width as usize;
    let w = 3 + bit / 32;
    let shift = bit % 32;
    let mut delta = words[w] >> shift;
    if shift + width as usize > 32 {
        delta |= words[w + 1] << (32 - shift);
    }
    delta &= (1u32 << width) - 1;
    f32::from_bits((base + delta) << SHIFT)
}
impl PackedLut {
    pub fn fetch(&self, index: usize, channel: usize) -> f32 {
        let offset = self.map[index / self.block_texels] as usize;
        decode_block(
            &self.data[offset..],
            self.block_texels,
            index % self.block_texels,
            channel,
        )
    }
    pub fn read(m: &Manifest, dir: &Path) -> Result<Self> {
        let p = m
            .rgb
            .as_ref()
            .and_then(|r| r.packed.as_ref())
            .ok_or("asset is not packed RGB")?;
        p.validate(m)?;
        let map = read_words(dir, "blocks.bin", p.map_words, &p.map_checksum)?;
        let data = read_words(dir, "radiance.bin", p.data_words, &p.data_checksum)?;
        let sun_words = read_words(
            dir,
            "sun.bin",
            3 * m.config.optical_depth_len(),
            &p.sun_checksum,
        )?;
        let sun: [Vec<f32>; 3] = std::array::from_fn(|c| {
            sun_words[c * m.config.optical_depth_len()..(c + 1) * m.config.optical_depth_len()]
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect()
        });
        if sun.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite packed Sun table".into());
        }
        // Validate block boundaries and every reconstructed code before upload.
        let mut starts = vec![false; data.len()];
        let mut offset = 0;
        let mut unique = 0;
        while offset < data.len() {
            if offset + 3 > data.len() {
                return Err("truncated packed block header".into());
            }
            let widths = [
                data[offset] >> 20,
                data[offset + 1] >> 20,
                data[offset + 2] >> 20,
            ];
            if widths.iter().any(|&w| w > 20) {
                return Err("invalid packed bit width".into());
            }
            let len = 3 + (p.block_texels * widths.iter().sum::<u32>() as usize).div_ceil(32);
            if offset + len > data.len() {
                return Err("truncated packed block payload".into());
            }
            for i in 0..p.block_texels {
                for c in 0..3 {
                    let value = decode_block(&data[offset..offset + len], p.block_texels, i, c);
                    if !value.is_finite() {
                        return Err("nonfinite packed radiance".into());
                    }
                }
            }
            starts[offset] = true;
            offset += len;
            unique += 1;
        }
        if unique != p.unique_blocks
            || map
                .iter()
                .any(|&o| !starts.get(o as usize).copied().unwrap_or(false))
        {
            return Err("invalid packed block index".into());
        }
        Ok(Self {
            map,
            data,
            sun,
            block_texels: p.block_texels,
        })
    }
}
fn read_words(dir: &Path, name: &str, len: usize, checksum: &str) -> Result<Vec<u32>> {
    let path = dir.join(name);
    if fs::metadata(&path)?.len() != len as u64 * 4 {
        return Err(format!("packed {name} size mismatch").into());
    }
    let bytes = fs::read(path)?;
    let mut hash = Hash::new();
    hash.update(&bytes);
    if hash.finish() != checksum {
        return Err(format!("packed {name} checksum mismatch").into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect())
}
fn write_words(dir: &Path, name: &str, data: &[u32]) -> Result<String> {
    use std::io::Write;
    let mut hash = Hash::new();
    let mut writer = std::io::BufWriter::new(fs::File::create(dir.join(format!("{name}.part")))?);
    for &word in data {
        let bytes = word.to_le_bytes();
        hash.update(&bytes);
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(dir.join(format!("{name}.part")), dir.join(name))?;
    Ok(hash.finish())
}

pub fn compress(source: &Path, output: &Path, block_texels: usize) -> Result<Manifest> {
    if ![16, 32, 64].contains(&block_texels) {
        return Err("block size must be 16, 32 or 64".into());
    }
    let mut m = Manifest::open(source)?;
    if m.rgb.as_ref().is_none_or(|r| r.packed.is_some()) {
        return Err("packing needs an uncompressed RGB export".into());
    }
    if output.exists() {
        return Err("packed output already exists; choose a new directory".into());
    }
    let start = Instant::now();
    let mut radiance = Vec::new();
    let mut solar = Vec::new();
    for c in 0..3 {
        let (s, r) = crate::rgb::read_channel(&m, source, c)?;
        radiance.push(r);
        solar.extend(s.iter().map(|v| v.to_bits()));
        eprintln!("CPU packing: read and verified channel {c}");
    }
    let n = m.config.scattering_len();
    let count = n.div_ceil(block_texels);
    let mut map = Vec::with_capacity(count);
    let mut data = Vec::new();
    let mut dictionary: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut code_words_before_sharing = 0;
    let mut max_relative = 0.0_f32;
    let mut max_absolute = 0.0_f32;
    let mut width_counts = [0u64; 21];
    // Bounded parallel batches; peak working memory does not include every
    // temporary encoded block. The output is deterministic across CPU counts.
    for begin in (0..count).step_by(65536) {
        let blocks: Vec<_> = (begin..(begin + 65536).min(count))
            .into_par_iter()
            .map(|b| {
                let v: Vec<[f32; 3]> = (0..block_texels)
                    .map(|i| {
                        std::array::from_fn(|c| radiance[c][(b * block_texels + i).min(n - 1)])
                    })
                    .collect();
                encode_block(&v)
            })
            .collect();
        for block in blocks {
            code_words_before_sharing += block.len();
            for &header in &block[..3] {
                width_counts[(header >> 20) as usize] += 1;
            }
            if let Some(&offset) = dictionary.get(&block) {
                map.push(offset);
            } else {
                let offset =
                    u32::try_from(data.len()).map_err(|_| "packed data exceeds u32 addressing")?;
                data.extend_from_slice(&block);
                dictionary.insert(block, offset);
                map.push(offset);
            }
        }
        if begin % 524288 == 0 {
            eprintln!(
                "CPU packing: {}/{} blocks, {} MiB so far",
                map.len(),
                count,
                (map.len() + data.len()) * 4 / 1048576
            );
        }
    }
    // Exhaustive node audit, with actual integer decoder, not just the quantizer.
    for i in 0..n {
        let offset = map[i / block_texels] as usize;
        for (c, channel) in radiance.iter().enumerate() {
            let a = channel[i];
            let b = decode_block(&data[offset..], block_texels, i % block_texels, c);
            let expected = f32::from_bits(quantize(a) << SHIFT);
            if b.to_bits() != expected.to_bits() {
                return Err("packed decoder mismatch".into());
            }
            max_absolute = max_absolute.max((a - b).abs());
            max_relative = max_relative.max((a - b).abs() / a.abs().max(1e-30));
        }
    }
    fs::create_dir_all(output)?;
    let packed = PackedStorage {
        format: FORMAT.into(),
        block_texels,
        map_words: map.len(),
        data_words: data.len(),
        unique_blocks: dictionary.len(),
        map_checksum: write_words(output, "blocks.bin", &map)?,
        data_checksum: write_words(output, "radiance.bin", &data)?,
        sun_checksum: write_words(output, "sun.bin", &solar)?,
        source_rgb: source.to_string_lossy().into(),
    };
    let report = serde_json::json!({"source":source,"format":FORMAT,"block_texels":block_texels,"scattering_texels":n,
        "source_rgb_gpu_bytes":12*(n+m.config.optical_depth_len()),"packed_gpu_bytes":packed.gpu_bytes(&m),
        "blocks":count,"unique_blocks":packed.unique_blocks,"unshared_radiance_bytes":(count+code_words_before_sharing)*4,
        "bits_per_channel_histogram":width_counts,"all_nodes_verified":true,"max_channel_relative_floor_1e_30":max_relative,
        "max_channel_absolute":max_absolute,"cpu_seconds":start.elapsed().as_secs_f32(),
        "grid_unchanged":true,"spectral_tau_embedded":false,"finite_segment_rendering_supported":false,
        "reference_quality_accepted":false,"comparison_baseline":"v6 RGB node export; spectral interpolation error is separate"});
    m.scalar_format = FORMAT.into();
    m.rgb.as_mut().unwrap().packed = Some(packed);
    m.save(output)?;
    fs::write(
        output.join("compression.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    eprintln!("{report}");
    Ok(m)
}
