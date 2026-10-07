use crate::{
    Result,
    config::{BakeConfig, CoordinateAllocation, CoordinateMapping, METADATA_RESERVE},
    mapping::{Geometry, State, sample, sample_radiance_state, scattering_cosine, sun_coord, unit},
    model::{BandInfo, Model},
    solver::{BakedBand, OrderStats},
};
use glam::Vec3;
use safetensors::tensor::Dtype;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

pub const KIND: &str = "spectral_atmosphere_4d_phase_inclusive_v1";
pub const RGB_KIND: &str = "rec2020_atmosphere_4d_phase_inclusive_v1";
pub const SOLVER: &str = "spectral-transport-wgpu-f32-v6";
pub const COORDINATE_MAPPING: &str = "rho-mu-split-cubic-endpoints-mus-asinh20-nu-cubic-v1";
pub const SUN_ALIGNED_MAPPING: &str = "rho-solar-horizon-square-cone-split75-angle-mixture-v2";
pub const SUN_ANGULAR_MAPPING: &str =
    "rho-solar-horizon-square-zenith-cap-cone-split75-angle-mixture-v3";
fn mapping_name(config: &BakeConfig) -> &'static str {
    if config.coordinate_allocation == CoordinateAllocation::RealtimeFitV1 {
        return "ray-frame-realtime-fitted-height-solar-phase-inclusive-v7";
    }
    match config.mapping {
        CoordinateMapping::Legacy => COORDINATE_MAPPING,
        CoordinateMapping::SunAligned => SUN_ALIGNED_MAPPING,
        CoordinateMapping::SunAlignedAngular => SUN_ANGULAR_MAPPING,
        CoordinateMapping::HorizonAligned => {
            "height-mixture-solar-two-horizons-topology-cubic-phase-v5"
        }
        CoordinateMapping::RayAlignedReference => {
            "ray-frame-physical-twilight-corners-phase-inclusive-v6"
        }
    }
}
const MAGIC: &[u8; 8] = b"SKYLUT01";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LutContainer {
    /// Missing in older manifests, whose payloads keep their original format.
    #[default]
    LegacyBinary,
    Safetensors,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BandRecord {
    pub checksum_fnv1a64: String,
    pub orders: Vec<OrderStats>,
    pub stopped_by_tolerance: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub kind: String,
    pub solver: String,
    pub coordinate_mapping: String,
    pub scalar_format: String,
    #[serde(default)]
    pub container: LutContainer,
    pub radiance_units: String,
    pub includes_ground: bool,
    pub includes_direct_sun_disk: bool,
    pub model_fingerprint_fnv1a64: String,
    pub geometry: Geometry,
    pub sun_radius_rad: f32,
    pub config: BakeConfig,
    pub bands: Vec<BandInfo>,
    pub adapter: String,
    pub records: Vec<Option<BandRecord>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb: Option<crate::rgb::RgbStorage>,
}

impl Manifest {
    pub fn new(model: &Model, config: BakeConfig, adapter: String) -> Result<Self> {
        config.validate(model.bands.len())?;
        config.validate_top_height(model.geometry.top_height())?;
        Ok(Self {
            kind: KIND.into(),
            solver: SOLVER.into(),
            coordinate_mapping: mapping_name(&config).into(),
            scalar_format: "little-endian-f32".into(),
            container: LutContainer::Safetensors,
            radiance_units: "W m^-2 sr^-1 integrated over each input band".into(),
            includes_ground: true,
            includes_direct_sun_disk: false,
            model_fingerprint_fnv1a64: fingerprint(model)?,
            geometry: model.geometry,
            sun_radius_rad: model.sun_radius,
            config,
            bands: model.bands.iter().map(|b| b.info.clone()).collect(),
            adapter,
            records: vec![None; model.bands.len()],
            rgb: None,
        })
    }
    pub fn complete(&self) -> bool {
        self.rgb.is_some() || self.records.iter().all(Option::is_some)
    }
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("asset.json");
        if fs::metadata(&path)?.len() > METADATA_RESERVE / 2 {
            return Err("oversized LUT manifest".into());
        }
        let m: Self = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        if m.container == LutContainer::LegacyBinary || m.rgb.is_some() {
            m.config.validate_existing(m.bands.len())?;
        } else {
            m.config.validate(m.bands.len())?;
        }
        m.config.validate_top_height(m.geometry.top_height())?;
        if ![KIND, RGB_KIND].contains(&m.kind.as_str())
            || (m.kind == RGB_KIND) != m.rgb.is_some()
            || ![
                "successive-orders-wgpu-f32-v1",
                "successive-orders-wgpu-f32-v2",
                "successive-orders-wgpu-f32-v3",
                "spectral-transport-wgpu-f32-v4",
                "spectral-transport-wgpu-f32-v5",
                SOLVER,
            ]
            .contains(&m.solver.as_str())
            || m.coordinate_mapping != mapping_name(&m.config)
            || m.scalar_format
                != if m.rgb.as_ref().is_some_and(|r| r.packed.is_some()) {
                    crate::packed::FORMAT
                } else {
                    "little-endian-f32"
                }
            || !m.includes_ground
            || m.includes_direct_sun_disk
            || m.records.len() != m.bands.len()
            || !m.geometry.bottom.is_finite()
            || !m.geometry.top.is_finite()
            || m.geometry.bottom <= 0.0
            || m.geometry.top <= m.geometry.bottom
            || !m.sun_radius_rad.is_finite()
            || m.sun_radius_rad <= 0.0
            || !(0.0..0.1).contains(&m.sun_radius_rad)
        {
            return Err("unsupported or invalid LUT manifest".into());
        }
        for b in &m.bands {
            if [b.center_nm, b.lower_nm, b.upper_nm, b.solar_irradiance_w_m2]
                .iter()
                .any(|v| !v.is_finite())
                || b.lower_nm >= b.upper_nm
                || b.center_nm < b.lower_nm
                || b.center_nm > b.upper_nm
                || b.solar_irradiance_w_m2 < 0.0
            {
                return Err("invalid LUT band metadata".into());
            }
        }
        if let Some(rgb) = &m.rgb {
            rgb.validate(&m)?;
        }
        Ok(m)
    }
    pub fn save(&self, dir: &Path) -> Result<()> {
        if self.rgb.is_none() {
            self.config.validate(self.bands.len())?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        if bytes.len() as u64 > METADATA_RESERVE / 2 {
            return Err("manifest exceeds metadata budget".into());
        }
        if self.rgb.is_none() {
            self.check_storage_budget(dir, 0, None)?;
        }
        let part = dir.join("asset.json.part");
        let mut file = File::create(&part)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(part, dir.join("asset.json"))?;
        Ok(())
    }
    pub fn write_band(&mut self, dir: &Path, index: usize, band: BakedBand) -> Result<()> {
        if self.rgb.is_some() {
            return Err("cannot write a spectral band into an RGB asset".into());
        }
        if index >= self.bands.len() || self.records[index].is_some() {
            return Err("band index invalid or already committed".into());
        }
        self.config.validate(self.bands.len())?;
        if self.container != LutContainer::Safetensors {
            return Err(
                "legacy binary LUTs are read-only; bake into a new safetensors directory".into(),
            );
        }
        if band.radiance.len() != self.config.scattering_len()
            || band.optical_depth.len() != self.config.optical_depth_len()
            || band.ground_irradiance.len() != self.config.ground_sun_samples
        {
            return Err("band dimensions do not match manifest".into());
        }
        if band
            .optical_depth
            .iter()
            .chain(&band.radiance)
            .chain(&band.ground_irradiance)
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("invalid band value".into());
        }
        let path = band_path(dir, index, self.container);
        self.check_storage_budget(dir, self.config.band_bytes()?, Some(&path))?;
        // An uncommitted orphan is from an interrupted bake. Remove it before
        // replacing it so resume never retains two full copies of a band.
        if path.exists() {
            fs::remove_file(&path)?;
        }
        let checksum = crate::tensor_container::write(
            &path,
            &[
                (
                    "optical_depth",
                    Dtype::F32,
                    self.config.optical_depth.to_vec(),
                    bytemuck::cast_slice(&band.optical_depth),
                ),
                (
                    "radiance",
                    Dtype::F32,
                    self.config.scattering.to_vec(),
                    bytemuck::cast_slice(&band.radiance),
                ),
                (
                    "ground_irradiance",
                    Dtype::F32,
                    vec![self.config.ground_sun_samples],
                    bytemuck::cast_slice(&band.ground_irradiance),
                ),
            ],
        )?;
        if fs::metadata(&path)?.len() > self.config.band_bytes()? {
            return Err("band file size mismatch".into());
        }
        self.records[index] = Some(BandRecord {
            checksum_fnv1a64: checksum,
            orders: band.orders,
            stopped_by_tolerance: band.stopped_by_tolerance,
        });
        self.save(dir)
    }

    /// Include interrupted-write temporaries and reserve both manifest copies.
    /// Payloads are direct children of their asset directory.
    fn check_storage_budget(&self, dir: &Path, extra: u64, replacing: Option<&Path>) -> Result<()> {
        let mut bytes = METADATA_RESERVE
            .checked_add(extra)
            .ok_or("asset size overflow")?;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if replacing == Some(path.as_path())
                || ["asset.json", "asset.json.part"]
                    .contains(&entry.file_name().to_string_lossy().as_ref())
            {
                continue;
            }
            if entry.file_type()?.is_file() {
                bytes = bytes
                    .checked_add(entry.metadata()?.len())
                    .ok_or("asset size overflow")?;
            }
        }
        let budget = self
            .config
            .max_asset_bytes
            .min(crate::config::MAX_TEACHER_BYTES);
        if bytes > budget {
            return Err(format!("LUT directory needs up to {bytes} bytes including temporary files; budget is {budget} bytes").into());
        }
        Ok(())
    }
    pub fn read_band(&self, dir: &Path, index: usize) -> Result<BandLut> {
        if self.rgb.is_some() {
            return Err("RGB asset has no per-wavelength radiance payload".into());
        }
        let record = self
            .records
            .get(index)
            .and_then(Option::as_ref)
            .ok_or("band has not been baked")?;
        let path = band_path(dir, index, self.container);
        let (optical_depth, radiance, ground_irradiance) =
            if self.container == LutContainer::Safetensors {
                let data = crate::tensor_container::read(
                    &path,
                    &[
                        (
                            "optical_depth",
                            Dtype::F32,
                            self.config.optical_depth.to_vec(),
                        ),
                        ("radiance", Dtype::F32, self.config.scattering.to_vec()),
                        (
                            "ground_irradiance",
                            Dtype::F32,
                            vec![self.config.ground_sun_samples],
                        ),
                    ],
                    &record.checksum_fnv1a64,
                )?;
                (
                    crate::tensor_container::f32_values(&data[0], true)?,
                    crate::tensor_container::f32_values(&data[1], true)?,
                    crate::tensor_container::f32_values(&data[2], true)?,
                )
            } else {
                let legacy_bytes =
                    self.config.band_bytes()? - crate::tensor_container::HEADER_RESERVE + 8;
                if fs::metadata(&path)?.len() != legacy_bytes {
                    return Err("band file length mismatch".into());
                }
                let mut file = BufReader::new(File::open(path)?);
                let mut header = [0u8; 8];
                file.read_exact(&mut header)?;
                if &header != MAGIC {
                    return Err("bad LUT magic/version".into());
                }
                let mut hash = Hash::new();
                hash.update(&header);
                let mut values = vec![0.0; (legacy_bytes as usize - 8) / 4];
                for x in &mut values {
                    let mut bytes = [0u8; 4];
                    file.read_exact(&mut bytes)?;
                    hash.update(&bytes);
                    *x = f32::from_le_bytes(bytes);
                    if !x.is_finite() || *x < 0.0 {
                        return Err("invalid stored LUT value".into());
                    }
                }
                if hash.finish() != record.checksum_fnv1a64 {
                    return Err("LUT checksum mismatch".into());
                }
                let ground_irradiance = values
                    .split_off(self.config.optical_depth_len() + self.config.scattering_len());
                let radiance = values.split_off(self.config.optical_depth_len());
                (values, radiance, ground_irradiance)
            };
        Ok(BandLut {
            geometry: self.geometry,
            config: self.config.clone(),
            info: self.bands[index].clone(),
            sun_radius: self.sun_radius_rad,
            optical_depth,
            radiance,
            ground_irradiance,
            scattering_cosines: (0..self.config.scattering[3])
                .map(|i| scattering_cosine(unit(i, self.config.scattering[3])))
                .collect(),
        })
    }
}

pub struct BandLut {
    pub geometry: Geometry,
    pub config: BakeConfig,
    pub info: BandInfo,
    pub sun_radius: f32,
    pub optical_depth: Vec<f32>,
    pub radiance: Vec<f32>,
    pub ground_irradiance: Vec<f32>,
    pub scattering_cosines: Vec<f32>,
}

impl BandLut {
    /// Observer on or above the planet, including space. Local frame has +Z up.
    /// Direct disk is optional; all stored radiance already includes phase.
    pub fn sample(
        &self,
        altitude_km: f32,
        view: Vec3,
        sun: Vec3,
        include_sun_disk: bool,
    ) -> Result<f32> {
        if !altitude_km.is_finite()
            || altitude_km < 0.0
            || !view.is_finite()
            || !sun.is_finite()
            || view.length_squared() < 1e-20
            || sun.length_squared() < 1e-20
        {
            return Err("lookup needs a nonnegative height and finite nonzero directions".into());
        }
        let view = view.normalize();
        let sun = sun.normalize();
        let s = State {
            altitude_km,
            mu: view.z,
            mu_s: sun.z,
            nu: crate::mapping::unit_direction_cosine(view, sun),
            ground: self.geometry.hits_ground(altitude_km, view.z),
        };
        let entry = self.geometry.atmosphere_entry(s);
        let mut radiance = entry.map_or(0.0, |(point, _)| {
            sample_radiance_state(
                &self.radiance,
                self.geometry,
                &self.config,
                point,
                &self.scattering_cosines,
            )
        });
        if include_sun_disk
            && entry.is_none_or(|(point, _)| !point.ground)
            && (view - sun).length_squared() <= 4.0 * (self.sun_radius * 0.5).sin().powi(2)
        {
            let omega = 4.0 * std::f32::consts::PI * (self.sun_radius * 0.5).sin().powi(2);
            radiance += entry.map_or(1.0, |(point, _)| self.transmittance(point))
                * self.info.solar_irradiance_w_m2
                / omega;
        }
        Ok(radiance)
    }
    pub fn transmittance(&self, s: State) -> f32 {
        let coords = [
            self.geometry
                .height_coord_mapped(s.altitude_km, self.config.mapping)
                * (self.config.optical_depth[0] - 1) as f32,
            self.geometry
                .view_coord(s.altitude_km, s.mu, s.ground, self.config.optical_depth[1]),
        ];
        (-sample(&self.optical_depth, self.config.optical_depth, coords)).exp()
    }
    pub fn ground_irradiance_at(&self, mu_s: f32) -> f32 {
        sample(
            &self.ground_irradiance,
            [self.config.ground_sun_samples],
            [match self.config.mapping {
                CoordinateMapping::Legacy => sun_coord(mu_s),
                CoordinateMapping::SunAligned | CoordinateMapping::SunAlignedAngular => self
                    .geometry
                    .solar_coord_mapped(0.0, mu_s, self.config.mapping),
                CoordinateMapping::HorizonAligned | CoordinateMapping::RayAlignedReference => {
                    crate::reference_mapping::solar_coord_config(
                        self.geometry,
                        &self.config,
                        0.0,
                        mu_s,
                    )
                }
            } * (self.config.ground_sun_samples - 1) as f32],
        )
    }
}

pub fn fingerprint(model: &Model) -> Result<String> {
    let mut hash = Hash::new();
    hash.update(&serde_json::to_vec(model)?);
    Ok(hash.finish())
}
fn band_path(dir: &Path, index: usize, container: LutContainer) -> PathBuf {
    let extension = match container {
        LutContainer::LegacyBinary => "bin",
        LutContainer::Safetensors => "safetensors",
    };
    dir.join(format!("band_{index:03}.{extension}"))
}
pub(crate) struct Hash(u64);
impl Hash {
    pub(crate) fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    pub(crate) fn finish(self) -> String {
        format!("{:016x}", self.0)
    }
}
