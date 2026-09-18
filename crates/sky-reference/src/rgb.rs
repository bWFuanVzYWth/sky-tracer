//! CPU export after spectral transport; runtime radiance is linear Rec.2020.
use crate::physics::spectrum::{SpectralBand, SpectralRgbConverter};
use crate::{
    Result,
    asset::{Hash, Manifest, RGB_KIND},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"SKYRGB01";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RgbStorage {
    pub color_space: String,
    pub weights: Vec<[f32; 3]>,
    pub solar_irradiance: [f32; 3],
    /// Each channel file: 8-byte magic, attenuated solar irradiance, radiance.
    pub channel_checksums: [String; 3],
    /// Exact original spectral optical depths, retained for future segment work.
    pub optical_depth_checksum: String,
    /// Packed payload replaces channel files. Checksums above identify the
    /// source RGB/tau payloads; packed files have their own checksums.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed: Option<crate::packed::PackedStorage>,
}

impl RgbStorage {
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        if self.color_space != "linear Rec.2020 / solar D65"
            || self.weights.len() != m.bands.len()
            || self
                .weights
                .iter()
                .flatten()
                .chain(self.solar_irradiance.iter())
                .any(|v| !v.is_finite())
            || self
                .channel_checksums
                .iter()
                .chain(std::iter::once(&self.optical_depth_checksum))
                .any(|s| s.len() != 16 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
            || m.records.iter().any(Option::is_some)
        {
            return Err("invalid RGB LUT metadata".into());
        }
        if let Some(packed) = &self.packed {
            packed.validate(m)?;
        }
        Ok(())
    }
}

pub fn rec2020_weights(m: &Manifest) -> Vec<[f32; 3]> {
    let bands: Vec<_> = m
        .bands
        .iter()
        .enumerate()
        .map(|(index, b)| SpectralBand {
            index,
            center_nm: b.center_nm,
            lower_nm: b.lower_nm,
            upper_nm: b.upper_nm,
            solar_irradiance_w_m2: b.solar_irradiance_w_m2,
            ozone_cross_section_cm2: 0.0,
        })
        .collect();
    let converter = SpectralRgbConverter::new_solar_d65(&bands);
    (0..bands.len())
        .map(|i| {
            let mut basis = vec![0.0; bands.len()];
            basis[i] = 1.0;
            let c = converter.to_linear_srgb(&basis);
            [
                0.627_404 * c.r + 0.329_282 * c.g + 0.0433136 * c.b,
                0.0690970 * c.r + 0.919_54 * c.g + 0.0113612 * c.b,
                0.0163916 * c.r + 0.0880132 * c.g + 0.8955952 * c.b,
            ]
        })
        .collect()
}

fn write_values(writer: &mut impl Write, hash: &mut Hash, values: &[f32]) -> Result<()> {
    for &value in values {
        if !value.is_finite() {
            return Err("nonfinite RGB export".into());
        }
        let bytes = value.to_le_bytes();
        writer.write_all(&bytes)?;
        hash.update(&bytes);
    }
    Ok(())
}

pub fn export(source: &Path, output: &Path) -> Result<Manifest> {
    let mut m = Manifest::open(source)?;
    if !m.complete() || m.rgb.is_some() {
        return Err("export needs a complete spectral asset".into());
    }
    if output.exists() {
        return Err("RGB output already exists; choose a new directory".into());
    }
    fs::create_dir_all(output)?;
    let weights = rec2020_weights(&m);
    let mut radiance: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; m.config.scattering_len()]);
    let mut solar_table: [Vec<f32>; 3] =
        std::array::from_fn(|_| vec![0.0; m.config.optical_depth_len()]);
    let mut solar = [0.0; 3];
    let mut tau_file = BufWriter::new(File::create(output.join("spectral_tau.part"))?);
    let mut tau_hash = Hash::new();
    for (i, w) in weights.iter().enumerate() {
        let band = m.read_band(source, i)?;
        write_values(&mut tau_file, &mut tau_hash, &band.optical_depth)?;
        for channel in 0..3 {
            let scale = w[channel];
            let solar_scale = scale * band.info.solar_irradiance_w_m2;
            solar[channel] += solar_scale;
            for (out, &value) in radiance[channel].iter_mut().zip(&band.radiance) {
                *out += scale * value;
            }
            for (out, &tau) in solar_table[channel].iter_mut().zip(&band.optical_depth) {
                *out += solar_scale * (-tau).exp();
            }
        }
        eprintln!("RGB export band {}/{}", i + 1, m.bands.len());
    }
    tau_file.flush()?;
    tau_file.get_ref().sync_all()?;
    drop(tau_file);
    fs::rename(
        output.join("spectral_tau.part"),
        output.join("spectral_tau.bin"),
    )?;
    let mut checksums = std::array::from_fn(|_| String::new());
    for channel in 0..3 {
        let path = output.join(format!("channel_{channel}.bin"));
        let part = path.with_extension("part");
        let mut writer = BufWriter::new(File::create(&part)?);
        let mut hash = Hash::new();
        writer.write_all(MAGIC)?;
        hash.update(MAGIC);
        write_values(&mut writer, &mut hash, &solar_table[channel])?;
        write_values(&mut writer, &mut hash, &radiance[channel])?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(part, path)?;
        checksums[channel] = hash.finish();
    }
    m.kind = RGB_KIND.into();
    m.radiance_units = "linear Rec.2020, solar-D65 transform of band-integrated radiance".into();
    m.records.fill(None);
    m.rgb = Some(RgbStorage {
        color_space: "linear Rec.2020 / solar D65".into(),
        weights,
        solar_irradiance: solar,
        channel_checksums: checksums,
        optical_depth_checksum: tau_hash.finish(),
        packed: None,
    });
    m.save(output)?;
    Ok(m)
}

pub fn read_channel(m: &Manifest, dir: &Path, channel: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let rgb = m.rgb.as_ref().ok_or("asset is not RGB")?;
    if rgb.packed.is_some() {
        return Err(
            "packed RGB: use PackedLut for random access without expanding the grid".into(),
        );
    }
    let checksum = rgb
        .channel_checksums
        .get(channel)
        .ok_or("RGB channel out of range")?;
    let path = dir.join(format!("channel_{channel}.bin"));
    let len = m.config.optical_depth_len() + m.config.scattering_len();
    if fs::metadata(&path)?.len() != 8 + len as u64 * 4 {
        return Err("RGB channel size mismatch".into());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    if &header != MAGIC {
        return Err("RGB magic mismatch".into());
    }
    let mut hash = Hash::new();
    hash.update(&header);
    let mut data = vec![0.0; len];
    for v in &mut data {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        hash.update(&bytes);
        *v = f32::from_le_bytes(bytes);
        if !v.is_finite() {
            return Err("nonfinite RGB channel".into());
        }
    }
    if hash.finish() != *checksum {
        return Err("RGB channel checksum mismatch".into());
    }
    let radiance = data.split_off(m.config.optical_depth_len());
    Ok((data, radiance))
}

pub fn verify(m: &Manifest, dir: &Path) -> Result<()> {
    if m.rgb.as_ref().is_some_and(|r| r.packed.is_some()) {
        crate::packed::PackedLut::read(m, dir)?;
        return Ok(());
    }
    for channel in 0..3 {
        read_channel(m, dir, channel)?;
    }
    let path = dir.join("spectral_tau.bin");
    if fs::metadata(&path)?.len() != m.bands.len() as u64 * m.config.optical_depth_len() as u64 * 4
    {
        return Err("spectral optical depth size mismatch".into());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Hash::new();
    let mut bytes = [0u8; 4];
    for _ in 0..m.bands.len() * m.config.optical_depth_len() {
        reader.read_exact(&mut bytes)?;
        hash.update(&bytes);
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() || value < 0.0 {
            return Err("invalid spectral optical depth".into());
        }
    }
    if hash.finish()
        != m.rgb
            .as_ref()
            .ok_or("asset is not RGB")?
            .optical_depth_checksum
    {
        return Err("spectral optical depth checksum mismatch".into());
    }
    Ok(())
}
