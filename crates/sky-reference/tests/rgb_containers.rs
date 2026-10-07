use safetensors::SafeTensors;
use sky_reference::{
    Result,
    asset::{LutContainer, Manifest},
    config::BakeConfig,
    mapping::Geometry,
    model::{BandInfo, BandModel, Model},
    packed::{PackedLut, quantize},
    solver::BakedBand,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path =
            std::env::temp_dir().join(format!("sky-rgb-container-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn spectral_fixture(dir: &Path) -> Result<Manifest> {
    fs::create_dir_all(dir)?;
    let model = Model {
        geometry: Geometry {
            bottom: 6360.0,
            top: 6460.0,
        },
        sun_radius: 0.00465,
        bands: [450.0, 550.0, 650.0]
            .into_iter()
            .map(|center_nm| BandModel {
                info: BandInfo {
                    center_nm,
                    lower_nm: center_nm - 50.0,
                    upper_nm: center_nm + 50.0,
                    solar_irradiance_w_m2: 1.0,
                },
                profile: Vec::new(),
                phase: Vec::new(),
            })
            .collect(),
    };
    let mut m = Manifest::new(&model, BakeConfig::smoke(), "CPU fixture".into())?;
    for band in 0..m.bands.len() {
        m.write_band(
            dir,
            band,
            BakedBand {
                optical_depth: (0..m.config.optical_depth_len())
                    .map(|i| i as f32 * 0.001 + band as f32)
                    .collect(),
                radiance: (0..m.config.scattering_len())
                    .map(|i| (0.25 + (i % 17) as f32) * (band + 1) as f32)
                    .collect(),
                ground_irradiance: vec![1.0; m.config.ground_sun_samples],
                orders: Vec::new(),
                stopped_by_tolerance: false,
            },
        )?;
    }
    Ok(m)
}

fn checksum(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn tensor_bytes(path: &Path, name: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    Ok(SafeTensors::deserialize(&bytes)?
        .tensor(name)?
        .data()
        .to_vec())
}

#[test]
fn rgb_and_packed_safetensors_preserve_layout_and_validate_values() -> Result<()> {
    let dir = TestDirectory::new()?;
    let source = dir.0.join("source");
    let spectral = spectral_fixture(&source)?;
    let rgb_dir = dir.0.join("rgb");
    let mut rgb = sky_reference::rgb::export(&source, &rgb_dir)?;
    assert_eq!(rgb.container, LutContainer::Safetensors);
    sky_reference::rgb::verify(&rgb, &rgb_dir)?;
    let channel_bytes = fs::read(rgb_dir.join("channel_0.safetensors"))?;
    let channel = SafeTensors::deserialize(&channel_bytes)?;
    assert_eq!(
        channel.tensor("radiance")?.shape(),
        spectral.config.scattering
    );
    assert_eq!(
        channel.tensor("solar_irradiance")?.shape(),
        spectral.config.optical_depth
    );
    let tau_bytes = fs::read(rgb_dir.join("spectral_tau.safetensors"))?;
    let tau = SafeTensors::deserialize(&tau_bytes)?;
    assert_eq!(
        tau.tensor("optical_depth")?.shape(),
        [
            spectral.bands.len(),
            spectral.config.optical_depth[0],
            spectral.config.optical_depth[1]
        ]
    );
    let values = tau.tensor("optical_depth")?;
    for band in 0..spectral.bands.len() {
        let begin = band * spectral.config.optical_depth_len() * 4;
        let expected = spectral.read_band(&source, band)?;
        assert_eq!(
            &values.data()[begin..begin + expected.optical_depth.len() * 4],
            bytemuck::cast_slice::<f32, u8>(&expected.optical_depth)
        );
    }

    let packed_dir = dir.0.join("packed");
    let packed_manifest = sky_reference::packed::compress(&rgb_dir, &packed_dir, 32)?;
    assert_eq!(packed_manifest.container, LutContainer::Safetensors);
    let packed = PackedLut::read(&packed_manifest, &packed_dir)?;
    for channel in 0..3 {
        let (solar, radiance) = sky_reference::rgb::read_channel(&rgb, &rgb_dir, channel)?;
        assert_eq!(packed.sun[channel], solar);
        for (i, value) in radiance.into_iter().enumerate() {
            assert_eq!(packed.fetch(i, channel).to_bits(), quantize(value) << 12);
        }
    }
    let sun_bytes = fs::read(packed_dir.join("sun.safetensors"))?;
    let sun = SafeTensors::deserialize(&sun_bytes)?;
    assert_eq!(
        sun.tensor("solar_irradiance")?.dtype(),
        safetensors::Dtype::F32
    );
    assert_eq!(
        sun.tensor("solar_irradiance")?.shape(),
        [
            3,
            spectral.config.optical_depth[0],
            spectral.config.optical_depth[1]
        ]
    );

    // An altered shape with the same element count must fail even when the
    // checksum is recomputed, otherwise consumers silently reinterpret axes.
    let mut corrupt = channel_bytes.clone();
    let header_len = u64::from_le_bytes(corrupt[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&corrupt[8..8 + header_len])?;
    header["radiance"]["shape"] = serde_json::json!([spectral.config.scattering_len()]);
    let new_header = serde_json::to_vec(&header)?;
    assert!(new_header.len() <= header_len);
    corrupt[8..8 + header_len].fill(b' ');
    corrupt[8..8 + new_header.len()].copy_from_slice(&new_header);
    fs::write(rgb_dir.join("channel_0.safetensors"), &corrupt)?;
    rgb.rgb.as_mut().unwrap().channel_checksums[0] = checksum(&corrupt);
    assert!(sky_reference::rgb::read_channel(&rgb, &rgb_dir, 0).is_err());
    fs::write(rgb_dir.join("channel_0.safetensors"), &channel_bytes)?;
    rgb.rgb.as_mut().unwrap().channel_checksums[0] = checksum(&channel_bytes);

    // A structurally valid container still needs physical value validation.
    let mut corrupt_tau = tau_bytes;
    let end = corrupt_tau.len();
    corrupt_tau[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(rgb_dir.join("spectral_tau.safetensors"), &corrupt_tau)?;
    rgb.rgb.as_mut().unwrap().optical_depth_checksum = checksum(&corrupt_tau);
    assert!(sky_reference::rgb::verify(&rgb, &rgb_dir).is_err());
    Ok(())
}

#[test]
fn legacy_rgb_and_packed_payloads_remain_readable() -> Result<()> {
    let dir = TestDirectory::new()?;
    let source = dir.0.join("source");
    spectral_fixture(&source)?;
    let rgb_dir = dir.0.join("rgb");
    let mut rgb = sky_reference::rgb::export(&source, &rgb_dir)?;
    let mut expected = Vec::new();
    for channel in 0..3 {
        expected.push(sky_reference::rgb::read_channel(&rgb, &rgb_dir, channel)?);
        let path = rgb_dir.join(format!("channel_{channel}.safetensors"));
        let mut bytes = b"SKYRGB01".to_vec();
        bytes.extend(tensor_bytes(&path, "solar_irradiance")?);
        bytes.extend(tensor_bytes(&path, "radiance")?);
        fs::write(rgb_dir.join(format!("channel_{channel}.bin")), &bytes)?;
        rgb.rgb.as_mut().unwrap().channel_checksums[channel] = checksum(&bytes);
        fs::remove_file(path)?;
    }
    let tau_path = rgb_dir.join("spectral_tau.safetensors");
    let tau = tensor_bytes(&tau_path, "optical_depth")?;
    fs::write(rgb_dir.join("spectral_tau.bin"), &tau)?;
    rgb.rgb.as_mut().unwrap().optical_depth_checksum = checksum(&tau);
    fs::remove_file(tau_path)?;
    rgb.container = LutContainer::LegacyBinary;
    rgb.save(&rgb_dir)?;
    // Older manifests omit the container field altogether.
    let mut json = serde_json::to_value(&rgb)?;
    json.as_object_mut().unwrap().remove("container");
    fs::write(rgb_dir.join("asset.json"), serde_json::to_vec(&json)?)?;
    let rgb = Manifest::open(&rgb_dir)?;
    assert_eq!(rgb.container, LutContainer::LegacyBinary);
    sky_reference::rgb::verify(&rgb, &rgb_dir)?;
    for (channel, expected) in expected.iter().enumerate() {
        assert_eq!(
            &sky_reference::rgb::read_channel(&rgb, &rgb_dir, channel)?,
            expected
        );
    }

    let packed_dir = dir.0.join("packed");
    let mut m = sky_reference::packed::compress(&rgb_dir, &packed_dir, 16)?;
    let packed = PackedLut::read(&m, &packed_dir)?;
    for (file, tensor) in [
        ("blocks", "blocks"),
        ("radiance", "radiance"),
        ("sun", "solar_irradiance"),
    ] {
        let path = packed_dir.join(format!("{file}.safetensors"));
        let bytes = tensor_bytes(&path, tensor)?;
        fs::write(packed_dir.join(format!("{file}.bin")), &bytes)?;
        let metadata = m.rgb.as_mut().unwrap().packed.as_mut().unwrap();
        match file {
            "blocks" => metadata.map_checksum = checksum(&bytes),
            "radiance" => metadata.data_checksum = checksum(&bytes),
            "sun" => metadata.sun_checksum = checksum(&bytes),
            _ => unreachable!(),
        }
        fs::remove_file(path)?;
    }
    m.container = LutContainer::LegacyBinary;
    m.save(&packed_dir)?;
    let legacy = PackedLut::read(&Manifest::open(&packed_dir)?, &packed_dir)?;
    assert_eq!(legacy.map, packed.map);
    assert_eq!(legacy.data, packed.data);
    assert_eq!(legacy.sun, packed.sun);
    Ok(())
}
