//! Fixed-medium four-wavelength runtime resource. Angular incident moments are
//! convolved with the actual species phases during ray marching. Unlike a sky
//! radiance residual, this local source is valid for finite atmospheric segments.
use crate::{Result, mapping::Geometry};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub mod cached;
pub mod renderer;
pub mod solar;

pub const KIND: &str = "four_wave_anisotropic_source_v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resource {
    pub kind: String,
    pub model: String,
    pub source_checksums: Vec<String>,
    pub wavelengths_nm: [f32; 4],
    pub rgb_from_per_nm: [[f32; 3]; 4],
    pub sun_per_nm: [f32; 4],
    pub geometry: Geometry,
    pub sun_radius: f32,
    pub albedo: f32,
    pub heights: Vec<f32>,
    pub degrees: Vec<usize>,
    /// Word offsets of each height layer; each SH coefficient uses two u32.
    pub offsets: Vec<usize>,
    pub sun_count: usize,
    pub optical: [usize; 2],
    /// Auxiliary vec4 offsets: scales, heights, profiles, phases, moments, ground.
    pub aux_offsets: [usize; 6],
    pub profile_count: usize,
    pub ground_count: usize,
    pub angular_quadrature: [usize; 2],
    pub payload_bytes: usize,
    pub files: Vec<(String, usize, String)>,
}

pub fn checksum(bytes: &[u8]) -> String {
    let mut h = 0xcbf29ce484222325_u64;
    for &v in bytes {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

impl Resource {
    pub fn open(path: &Path) -> Result<Self> {
        if fs::metadata(path.join("resource.json"))?.len() > 65536 {
            return Err("oversized four-wave resource metadata".into());
        }
        let r: Self = serde_json::from_slice(&fs::read(path.join("resource.json"))?)?;
        if r.kind != KIND
            || r.heights.len() < 2
            || r.heights.len() != r.degrees.len()
            || r.heights.len() != r.offsets.len()
            || r.sun_count < 2
            || r.sun_count > 4096
            || r.heights.len() > 1024
            || r.heights.first() != Some(&0.0)
            || r.heights.last() != Some(&r.geometry.top_height())
            || !r.geometry.bottom.is_finite()
            || !r.geometry.top.is_finite()
            || r.geometry.bottom <= 0.0
            || r.geometry.top <= r.geometry.bottom
            || !r.sun_radius.is_finite()
            || !(0.0..0.1).contains(&r.sun_radius)
            || !r.albedo.is_finite()
            || !(0.0..=1.0).contains(&r.albedo)
            || r.optical.iter().any(|&v| !(2..=4096).contains(&v))
            || !(2..=1024).contains(&r.profile_count)
            || !(2..=4096).contains(&r.ground_count)
            || r.rgb_from_per_nm.iter().flatten().any(|v| !v.is_finite())
            || r.sun_per_nm.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || r.wavelengths_nm.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || r.heights
                .windows(2)
                .any(|w| !w[0].is_finite() || w[0] >= w[1])
            || r.degrees.iter().any(|&d| d != 2 && d != 16)
            || r.payload_bytes > 16 * 1024 * 1024
        {
            return Err("invalid four-wave source resource".into());
        }
        let mut words = 0;
        for (i, &degree) in r.degrees.iter().enumerate() {
            if r.offsets[i] != words || (r.heights[i] <= 35.0 && degree != 16) {
                return Err("invalid source layer layout".into());
            }
            words += 2 * r.sun_count * crate::anisotropic::count(degree);
        }
        let scales = r.heights.len() * r.sun_count;
        let heights = scales;
        let profile = heights + r.heights.len();
        let phases = profile + 7 * r.profile_count;
        let moments = phases + 4096;
        let ground = moments + 17 * 5;
        let sizes = [
            words * 4,
            (ground + r.ground_count) * 16,
            r.optical[0] * r.optical[1] * 8,
        ];
        if r.aux_offsets != [0, heights, profile, phases, moments, ground]
            || r.files.len() != 3
            || sizes.iter().sum::<usize>() != r.payload_bytes
            || ["moments.u32", "aux.f32", "optical.u32"]
                .iter()
                .zip(sizes)
                .any(|(name, size)| {
                    r.files
                        .iter()
                        .filter(|(n, s, _)| n == name && *s == size)
                        .count()
                        != 1
                })
        {
            return Err("invalid source payload layout".into());
        }
        Ok(r)
    }
    pub fn read(&self, dir: &Path, name: &str) -> Result<Vec<u8>> {
        let (_, size, hash) = self
            .files
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or("missing source payload")?;
        let bytes = fs::read(dir.join(name))?;
        if bytes.len() != *size || checksum(&bytes) != *hash {
            return Err("source payload size/checksum mismatch".into());
        }
        Ok(bytes)
    }
}

/// CPU mirror used to isolate source compression/interpolation from ray marching.
pub struct CpuSource {
    pub resource: Resource,
    aux: Vec<glam::Vec4>,
    packed: Vec<u32>,
}
impl CpuSource {
    pub fn open(path: &Path) -> Result<Self> {
        let resource = Resource::open(path)?;
        let bytes = resource.read(path, "aux.f32")?;
        let aux = bytes
            .chunks_exact(16)
            .map(|b| {
                glam::Vec4::from_array(std::array::from_fn(|i| {
                    f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
                }))
            })
            .collect();
        let bytes = resource.read(path, "moments.u32")?;
        let packed = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(Self {
            resource,
            aux,
            packed,
        })
    }
    fn coefficient(&self, h: usize, s: usize, i: usize) -> glam::Vec4 {
        let r = &self.resource;
        let count = crate::anisotropic::count(r.degrees[h]);
        if i >= count {
            return glam::Vec4::ZERO;
        }
        let start = r.offsets[h] + 2 * (s * count + i);
        let a = self.packed[start];
        let b = self.packed[start + 1];
        glam::Vec4::from_array(
            [a as u16, (a >> 16) as u16, b as u16, (b >> 16) as u16]
                .map(|v| half::f16::from_bits(v).to_f32()),
        )
    }
    pub fn source(
        &self,
        point: crate::mapping::State,
        models: &[crate::model::BandModel; 4],
    ) -> glam::Vec4 {
        use glam::{Vec3, Vec4};
        let r = &self.resource;
        let h = point.altitude_km;
        let hi = r
            .heights
            .partition_point(|&v| v <= h)
            .clamp(1, r.heights.len() - 1);
        let lo = hi - 1;
        let th = ((h - r.heights[lo]) / (r.heights[hi] - r.heights[lo])).clamp(0.0, 1.0);
        let solar = [lo, hi].map(|i| {
            let x = crate::reference_mapping::solar_coord(r.geometry, r.heights[i], point.mu_s)
                * (r.sun_count - 1) as f32;
            let k = (x as usize).min(r.sun_count - 2);
            (k, x - k as f32)
        });
        let degree = if h >= 35.0 { 2 } else { 16 };
        let mut normalized = Vec::new();
        for i in 0..crate::anisotropic::count(degree) {
            let vals = std::array::from_fn::<_, 2, _>(|j| {
                let layer = [lo, hi][j];
                let (s, t) = solar[j];
                self.coefficient(layer, s, i)
                    .lerp(self.coefficient(layer, s + 1, i), t)
            });
            normalized.push(vals[0].lerp(vals[1], th));
        }
        let scale = [lo, hi].map(|layer| {
            let (s, t) = solar[usize::from(layer == hi)];
            let offset = layer * r.sun_count + s;
            let log0 = self.aux[offset].to_array().map(|v| v.max(1e-30).ln());
            let log1 = self.aux[offset + 1].to_array().map(|v| v.max(1e-30).ln());
            Vec4::from_array(log0).lerp(Vec4::from_array(log1), t)
        });
        let scale = Vec4::from_array(scale[0].lerp(scale[1], th).to_array().map(f32::exp));
        let mu = point.mu.clamp(-1.0, 1.0);
        let radial = (1.0 - mu * mu).max(0.0).sqrt();
        let x = ((point.nu - mu * point.mu_s)
            / (1.0 - point.mu_s * point.mu_s).max(0.0).sqrt().max(1e-10))
        .clamp(-radial, radial);
        let direction = Vec3::new(x, (radial * radial - x * x).max(0.0).sqrt(), mu);
        let mut basis = vec![0.0; normalized.len()];
        crate::anisotropic::cosine_sh(direction, degree, &mut basis);
        let c = models.each_ref().map(|m| m.coefficients(h));
        let mut result = Vec4::ZERO;
        for l in 0..=degree {
            let mut weight = Vec4::ZERO;
            for species in 0..5 {
                weight += Vec4::from_array(c.map(|v| v.scattering[species]))
                    * self.aux[r.aux_offsets[4] + l * 5 + species];
            }
            for m in 0..=l {
                let i = crate::anisotropic::index(l, m);
                result += weight * normalized[i] * basis[i];
            }
        }
        (result * scale).max(Vec4::ZERO)
    }
}
