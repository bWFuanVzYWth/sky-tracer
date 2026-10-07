//! Container and recovery checks use synthetic arrays and never initialize a GPU.
use safetensors::{SafeTensors, tensor::Dtype};
use sky_reference::{
    Result,
    asset::{BandRecord, LutContainer, Manifest},
    config::{BakeConfig, CoordinateAllocation, MAX_TEACHER_BYTES},
    mapping::Geometry,
    model::{BandInfo, BandModel, Model},
    solver::BakedBand,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sky-spectral-container-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        assert!(self.0.starts_with(std::env::temp_dir()));
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn model(bands: usize) -> Model {
    Model {
        geometry: Geometry {
            bottom: 6360.0,
            top: 6480.0,
        },
        sun_radius: 0.00465,
        bands: (0..bands)
            .map(|i| {
                let center_nm = 500.0 + i as f32 * 10.0;
                BandModel {
                    info: BandInfo {
                        center_nm,
                        lower_nm: center_nm - 5.0,
                        upper_nm: center_nm + 5.0,
                        solar_irradiance_w_m2: 1.0,
                    },
                    profile: Vec::new(),
                    phase: Vec::new(),
                }
            })
            .collect(),
    }
}

fn band(config: &BakeConfig, scale: f32) -> BakedBand {
    BakedBand {
        optical_depth: (0..config.optical_depth_len())
            .map(|i| i as f32 * 0.001 * scale)
            .collect(),
        radiance: (0..config.scattering_len())
            .map(|i| (i as f32 + 0.5) * scale)
            .collect(),
        ground_irradiance: (0..config.ground_sun_samples)
            .map(|i| (i as f32 + 1.0) * scale)
            .collect(),
        orders: Vec::new(),
        stopped_by_tolerance: false,
    }
}

fn checksum(bytes: &[u8]) -> String {
    let value = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("{value:016x}")
}

fn replace_header(bytes: &[u8], change: impl FnOnce(&mut serde_json::Value)) -> Result<Vec<u8>> {
    let old_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + old_len])?;
    change(&mut header);
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut result = (header.len() as u64).to_le_bytes().to_vec();
    result.extend(header);
    result.extend_from_slice(&bytes[8 + old_len..]);
    Ok(result)
}

#[test]
fn spectral_tensors_preserve_axes_values_and_committed_resume_state() -> Result<()> {
    let dir = TestDirectory::new()?;
    let config = BakeConfig::smoke();
    let mut manifest = Manifest::new(&model(2), config.clone(), "CPU fixture".into())?;
    manifest.save(&dir.0)?;
    // A payload committed before its manifest update is replaced on recovery.
    fs::write(dir.0.join("band_000.safetensors"), b"interrupted payload")?;
    let expected = band(&config, 1.0);
    manifest.write_band(&dir.0, 0, band(&config, 1.0))?;
    assert!(!manifest.complete());
    let checksum = manifest.records[0]
        .as_ref()
        .unwrap()
        .checksum_fnv1a64
        .clone();
    let mut resumed = Manifest::open(&dir.0)?;
    assert_eq!(resumed.container, LutContainer::Safetensors);
    assert_eq!(resumed.config, config);
    assert!(resumed.read_band(&dir.0, 1).is_err());
    assert!(resumed.write_band(&dir.0, 0, band(&config, 2.0)).is_err());
    resumed.write_band(&dir.0, 1, band(&config, 2.0))?;
    assert!(Manifest::open(&dir.0)?.complete());
    assert_eq!(
        resumed.records[0].as_ref().unwrap().checksum_fnv1a64,
        checksum
    );
    let actual = resumed.read_band(&dir.0, 0)?;
    assert_eq!(actual.optical_depth, expected.optical_depth);
    assert_eq!(actual.radiance, expected.radiance);
    assert_eq!(actual.ground_irradiance, expected.ground_irradiance);
    let bytes = fs::read(dir.0.join("band_000.safetensors"))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    assert_eq!(tensors.names().len(), 3);
    for (name, shape) in [
        ("optical_depth", config.optical_depth.to_vec()),
        ("radiance", config.scattering.to_vec()),
        ("ground_irradiance", vec![config.ground_sun_samples]),
    ] {
        let tensor = tensors.tensor(name)?;
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), shape);
    }
    let actual_bytes = fs::read_dir(&dir.0)?.try_fold(0u64, |bytes, entry| -> Result<u64> {
        Ok(bytes + entry?.metadata()?.len())
    })?;
    assert!(actual_bytes <= config.asset_bytes(2)?);
    Ok(())
}

#[test]
fn spectral_reader_validates_tensor_schema_values_and_integrity() -> Result<()> {
    let dir = TestDirectory::new()?;
    let config = BakeConfig::smoke();
    let mut manifest = Manifest::new(&model(1), config.clone(), "CPU fixture".into())?;
    manifest.write_band(&dir.0, 0, band(&config, 1.0))?;
    let path = dir.0.join("band_000.safetensors");
    let original = fs::read(&path)?;
    let valid_checksum = manifest.records[0]
        .as_ref()
        .unwrap()
        .checksum_fnv1a64
        .clone();
    let malformed = [
        replace_header(&original, |header| {
            header["radiance"]["dtype"] = "I32".into()
        })?,
        replace_header(&original, |header| {
            header["radiance"]["shape"] = serde_json::json!([config.scattering_len()]);
        })?,
        replace_header(&original, |header| {
            let values = header.as_object_mut().unwrap();
            let radiance = values.remove("radiance").unwrap();
            values.insert("incorrect_radiance".into(), radiance);
        })?,
        replace_header(&original, |header| {
            header["radiance"]["data_offsets"][0] = 1.into()
        })?,
    ];
    for bytes in malformed {
        // Recompute integrity metadata so rejection must come from the schema.
        manifest.records[0].as_mut().unwrap().checksum_fnv1a64 = checksum(&bytes);
        fs::write(&path, bytes)?;
        assert!(manifest.read_band(&dir.0, 0).is_err());
    }
    let tensors = SafeTensors::deserialize(&original)?;
    let payload = tensors.tensor("radiance")?.data();
    let payload_start = payload.as_ptr() as usize - original.as_ptr() as usize;
    let mut negative = original.clone();
    negative[payload_start..payload_start + 4].copy_from_slice(&(-1.0f32).to_le_bytes());
    manifest.records[0].as_mut().unwrap().checksum_fnv1a64 = checksum(&negative);
    fs::write(&path, negative)?;
    assert!(manifest.read_band(&dir.0, 0).is_err());
    let mut corrupt = original.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    manifest.records[0].as_mut().unwrap().checksum_fnv1a64 = valid_checksum;
    fs::write(&path, corrupt)?;
    assert!(manifest.read_band(&dir.0, 0).is_err());
    fs::write(&path, &original[..original.len() - 1])?;
    assert!(manifest.read_band(&dir.0, 0).is_err());
    fs::write(&path, 4096u64.to_le_bytes())?;
    assert!(manifest.read_band(&dir.0, 0).is_err());
    Ok(())
}

#[test]
fn old_manifests_without_container_or_allocation_keep_binary_lookup() -> Result<()> {
    let dir = TestDirectory::new()?;
    let config = BakeConfig::smoke();
    let mut manifest = Manifest::new(&model(1), config.clone(), "legacy fixture".into())?;
    let expected = band(&config, 1.0);
    let mut bytes = b"SKYLUT01".to_vec();
    for value in expected
        .optical_depth
        .iter()
        .chain(&expected.radiance)
        .chain(&expected.ground_irradiance)
    {
        bytes.extend(value.to_le_bytes());
    }
    fs::write(dir.0.join("band_000.bin"), &bytes)?;
    manifest.records[0] = Some(BandRecord {
        checksum_fnv1a64: checksum(&bytes),
        orders: Vec::new(),
        stopped_by_tolerance: false,
    });
    let mut legacy = serde_json::to_value(&manifest)?;
    legacy.as_object_mut().unwrap().remove("container");
    legacy["config"]
        .as_object_mut()
        .unwrap()
        .remove("coordinate_allocation");
    // Historical resources can retain budgets above the new creation ceiling.
    legacy["config"]["max_asset_bytes"] = (MAX_TEACHER_BYTES + 1).into();
    fs::write(dir.0.join("asset.json"), serde_json::to_vec(&legacy)?)?;
    let mut opened = Manifest::open(&dir.0)?;
    assert_eq!(opened.container, LutContainer::LegacyBinary);
    assert_eq!(
        opened.config.coordinate_allocation,
        CoordinateAllocation::Legacy
    );
    let actual = opened.read_band(&dir.0, 0)?;
    assert_eq!(actual.optical_depth, expected.optical_depth);
    assert_eq!(actual.radiance, expected.radiance);
    assert_eq!(actual.ground_irradiance, expected.ground_irradiance);
    opened.records[0] = None;
    opened.config.max_asset_bytes = MAX_TEACHER_BYTES;
    assert!(opened.write_band(&dir.0, 0, band(&config, 1.0)).is_err());
    assert!(!dir.0.join("band_000.safetensors").exists());
    Ok(())
}

#[test]
fn teacher_creation_ceiling_cannot_be_raised_by_custom_configuration() -> Result<()> {
    let teacher = BakeConfig::current_reference();
    teacher.validate(41)?;
    assert!(teacher.asset_bytes(41)? <= MAX_TEACHER_BYTES);
    let mut custom = BakeConfig::smoke();
    custom.max_asset_bytes = MAX_TEACHER_BYTES + 1;
    assert!(custom.validate(1).is_err());
    assert!(Manifest::new(&model(1), custom, "invalid budget".into()).is_err());
    let mut oversized = teacher;
    oversized.scattering[2] *= 2;
    assert!(oversized.asset_bytes(41)? > MAX_TEACHER_BYTES);
    assert!(oversized.validate(41).is_err());
    Ok(())
}
