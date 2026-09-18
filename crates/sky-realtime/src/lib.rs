//! Four-wavelength, fixed-medium successive transport on a directional 4D grid.
//! The medium is solved afresh on the GPU. No baked radiance/SH resource is used.
//! Runtime uses direct single scattering + the local multiple-scattering source,
//! cached in an observer SkyView, with a separate perspective projection.
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub fov_y_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub altitude_km: f32,
}

use crate::{geometry::unit, model::Model};
use baseline_mapping as reference_mapping;
mod baseline_mapping;
pub mod geometry;
pub mod model;
mod phase_integral;
mod quadrature;
use std::{f32::consts::PI, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;
pub mod mapping;

pub fn shader_source() -> String {
    [
        include_str!("solver.wgsl"),
        include_str!("transport.wgsl"),
        include_str!("sky_view.wgsl"),
    ]
    .join("\n")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub heights: u32,
    pub suns: u32,
    pub phases: u32,
    pub cones: u32,
    /// Polar quadrature samples per chart.
    pub angular_mu: u32,
    /// Azimuth samples on one reflection half; full-sphere equivalent is twice this.
    pub angular_phi: u32,
    pub ray_steps: u32,
    pub iterations: u32,
    pub optical: [u32; 2],
    pub optical_steps: u32,
    pub horizon_power: f32,
    pub compact_rayleigh: bool,
    /// Ablation mask: height=1, solar=2, phase=4, cone=8, optical=16, SkyView=32.
    /// Zero reproduces the legacy coordinates.
    pub mapping_flags: u32,
    pub sky_size: u32,
    pub specialize_constants: bool,
    /// Deterministic importance quadrature diagnostics; defaults retain the
    /// calibrated Sun/horizon partition and near-density path concentration.
    pub sun_importance_width: f32,
    /// Zero selects uniform-distance steps for an ablation, not the preset.
    pub path_height_scale: f32,
    pub high_sun_weight: f32,
    /// Zero retains the full-height scratch layout for equivalence diagnostics.
    pub batch_heights: u32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            heights: 32,
            suns: 97,
            phases: 24,
            cones: 8,
            angular_mu: 12,
            angular_phi: 16,
            ray_steps: 64,
            iterations: 8,
            optical: [256, 1024],
            optical_steps: 1024,
            horizon_power: 1.0,
            compact_rayleigh: false,
            mapping_flags: 0,
            sky_size: 256,
            specialize_constants: false,
            sun_importance_width: 0.01,
            path_height_scale: 0.25,
            high_sun_weight: 1.0,
            batch_heights: 0,
        }
    }
}
impl Config {
    /// Performance-first preset, validated against dense convergence probes.
    /// Retains eight transport iterations and a 256² SkyView for twilight detail.
    pub fn balanced() -> Self {
        Self {
            heights: 40,
            suns: 160,
            phases: 20,
            cones: 12,
            angular_mu: 32,
            angular_phi: 16,
            ray_steps: 64,
            iterations: 8,
            horizon_power: 3.0,
            compact_rayleigh: true,
            mapping_flags: 63,
            specialize_constants: true,
            high_sun_weight: 0.0,
            batch_heights: 4,
            optical: [128, 512],
            optical_steps: 256,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<()> {
        if !(4..=128).contains(&self.heights)
            || !(8..=257).contains(&self.suns)
            || !(4..=64).contains(&self.phases)
            || !(4..=32).contains(&self.cones)
            || !(2..=64).contains(&self.angular_mu)
            || !(2..=64).contains(&self.angular_phi)
            || !(4..=512).contains(&self.ray_steps)
            || !(1..=32).contains(&self.iterations)
            || self.optical.iter().any(|&n| !(8..=4096).contains(&n))
            || !(16..=4096).contains(&self.optical_steps)
            || self.suns * self.phases > 8192
            || !self.horizon_power.is_finite()
            || !(1.0..=4.0).contains(&self.horizon_power)
            || self.mapping_flags > 63
            || !(64..=1024).contains(&self.sky_size)
            || self.sky_size % 8 != 0
            || !self.sun_importance_width.is_finite()
            || !(0.0001..=1.0).contains(&self.sun_importance_width)
            || !self.path_height_scale.is_finite()
            || !(0.0..=4.0).contains(&self.path_height_scale)
            || !self.high_sun_weight.is_finite()
            || !(0.0..=1.0).contains(&self.high_sun_weight)
            || self.batch_heights > 128
        {
            return Err("invalid hybrid solver dimensions".into());
        }
        Ok(())
    }
    fn samples(&self) -> u32 {
        3 * self.angular_mu * self.angular_phi
    }
    pub fn source_bytes(&self) -> u64 {
        self.heights as u64 * self.suns as u64 * self.phases as u64 * self.cones as u64 * 8
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Wavelengths {
    pub indices: [usize; 4],
    pub wavelengths_nm: [f32; 4],
    #[serde(rename = "rec2020_from_per_nm")]
    pub rgb: [[f32; 3]; 4],
    #[serde(rename = "sun_irradiance_per_nm")]
    pub solar: [f32; 4],
}
impl Wavelengths {
    pub fn optimized_four() -> Self {
        serde_json::from_str(include_str!("../configs/wavelengths.json"))
            .expect("embedded wavelength calibration")
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size_steps: [u32; 4],
    view: [f32; 4],
    sun: [f32; 4],
    dims: [u32; 4],
    offsets0: [u32; 4],
    offsets1: [u32; 4],
    medium: [f32; 4],
    solar: [f32; 4],
    rgb: [[f32; 4]; 4],
    solve: [u32; 4],
    extra: [u32; 4],
    mapping: [u32; 4],
    sky: [u32; 4],
    optical_segments: [[f32; 4]; 6],
    angular_mapping: [f32; 4],
    importance: [f32; 4],
    batch: [u32; 4],
}
struct Medium {
    data: Vec<[f32; 4]>,
    params: Params,
    key: String,
}
impl Medium {
    fn field_bytes(&self, c: &Config) -> u64 {
        let low = self.params.offsets1[1] as u64;
        low * c.suns as u64 * c.phases as u64 * c.cones as u64 * 8
            + (c.heights as u64 - low + 1) * c.suns as u64 * 5 * 16
            + c.heights as u64 * c.suns as u64 * 16
            + c.suns as u64 * 16
    }
    fn resident_bytes(&self, c: &Config) -> u64 {
        self.field_bytes(c)
            + c.optical[0] as u64 * c.optical[1] as u64 * 8
            + self.data.len() as u64 * 16
            + c.sky_size as u64 * c.sky_size as u64 * 16
            + (12 + c.sky_size as u64) * 16
    }
    fn new(model: &Model, w: &Wavelengths, c: &Config, albedo: f32) -> Result<Self> {
        c.validate()?;
        if !albedo.is_finite()
            || !(0.0..=1.0).contains(&albedo)
            || w.indices.iter().any(|&i| i >= model.bands.len())
            || w.rgb
                .iter()
                .flatten()
                .chain(w.solar.iter())
                .any(|v| !v.is_finite())
        {
            return Err("invalid medium or wavelength transform".into());
        }
        for k in 0..4 {
            if (model.bands[w.indices[k]].info.center_nm - w.wavelengths_nm[k]).abs() > 0.01 {
                return Err("wavelength candidate and physical spectrum disagree".into());
            }
            let band = &model.bands[w.indices[k]];
            if band.profile.len() < 2
                || band.phase.len() != 4096
                || band.phase.iter().any(|v| !v.is_finite() || *v < 0.0)
                || band.profile.iter().any(|(h, c)| {
                    !h.is_finite()
                        || !c.extinction.is_finite()
                        || c.extinction < 0.0
                        || c.scattering.iter().any(|v| !v.is_finite() || *v < 0.0)
                        || (*h >= 35.0 && c.scattering[1..].iter().any(|&x| x != 0.0))
                })
            {
                return Err(
                    "invalid physical profile, or aerosol above the prototype's 35 km boundary"
                        .into(),
                );
            }
        }
        let g = model.geometry;
        if g.top_height() <= 35.0 {
            return Err("prototype requires an atmosphere extending above 35 km".into());
        }
        let mut heights: Vec<_> = (0..c.heights)
            .map(|i| reference_mapping::height(g, unit(i as usize, c.heights as usize)))
            .collect();
        heights[0] = 0.0;
        *heights.last_mut().unwrap() = g.top_height();
        let closest = (1..heights.len() - 1)
            .min_by(|&i, &j| {
                (heights[i] - 35.0)
                    .abs()
                    .total_cmp(&(heights[j] - 35.0).abs())
            })
            .unwrap();
        heights[closest] = 35.0;
        heights.sort_by(f32::total_cmp);
        if c.mapping_flags & 1 != 0 {
            heights = mapping::heights(g, c.heights);
        }
        let mut data: Vec<_> = heights.iter().map(|&h| [h, 0.0, 0.0, 0.0]).collect();
        let profile_offset = data.len() as u32;
        let profile = &model.bands[w.indices[0]].profile;
        for (h, _) in profile {
            let coefficients = w.indices.map(|i| model.bands[i].coefficients(*h));
            data.push([*h, 0.0, 0.0, 0.0]);
            data.push(coefficients.map(|v| v.extinction));
            for k in 0..5 {
                data.push(coefficients.map(|v| v.scattering[k]));
            }
        }
        let phase_offset = data.len() as u32;
        for i in 0..4096 {
            data.push(w.indices.map(|k| model.bands[k].phase[i]));
        }
        // Preserve the integral of the tabulated phase interpolant. It differs
        // slightly from one; discrete energy correction must tend to that same
        // physical operator when quadrature resolution increases.
        let phase_mass_offset = data.len() as u32;
        let phase_mass = w
            .indices
            .map(|i| phase_integral::phase_moments(&model.bands[i], 0));
        for species in 1..5 {
            data.push(std::array::from_fn(|k| phase_mass[k][0][species]));
        }
        let states_offset = data.len() as u32;
        for &h in &heights {
            let coefficients = mapping::solar_coefficients(g, h);
            for si in 0..c.suns {
                let mu_s = if c.mapping_flags & 2 != 0 {
                    mapping::inverse(
                        |e| mapping::solar_coord(e, &coefficients),
                        unit(si as usize, c.suns as usize),
                        -PI * 0.5,
                        PI * 0.5,
                    )
                    .sin()
                } else {
                    reference_mapping::solar_cosine(g, h, unit(si as usize, c.suns as usize))
                };
                data.push([h, mu_s, (1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0]);
            }
        }
        let angles_offset = data.len() as u32;
        let radial = quadrature::gauss_legendre(c.angular_mu as usize);
        // One uniform azimuth rule at every altitude. Gauss nodes remain only
        // on the original mapped polar coordinate.
        for &(u, w) in &radial {
            let mu = 1.0 - 2.0 * u * u * u;
            let r = (1.0 - mu * mu).max(0.0).sqrt();
            for j in 0..c.angular_phi {
                let phi = PI * (j as f32 + 0.5) / c.angular_phi as f32;
                data.push([
                    r * phi.cos(),
                    r * phi.sin(),
                    mu,
                    w * 6.0 * u * u * 2.0 * PI / c.angular_phi as f32,
                ]);
            }
        }
        for &(u, w) in &radial {
            for j in 0..c.angular_phi {
                let phi = PI * (j as f32 + 0.5) / c.angular_phi as f32;
                data.push([u, phi.cos(), phi.sin(), w * 2.0 * PI / c.angular_phi as f32]);
            }
        }
        let phase_nodes_offset = data.len() as u32;
        for j in 0..c.phases {
            let target = unit(j as usize, c.phases as usize);
            let mut lo = 0.0;
            let mut hi = PI;
            for _ in 0..32 {
                let mid = (lo + hi) * 0.5;
                let f = mapping::phase_coord(mid.cos(), c.mapping_flags & 4 != 0);
                if f < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            data.push([((lo + hi) * 0.5).cos(), 0.0, 0.0, 0.0]);
        }
        let solar_offset = data.len() as u32;
        for &h in &heights {
            data.extend(mapping::solar_coefficients(g, h));
        }
        let optical_segments = mapping::optical_segments(g.top_height(), c.optical[0]);
        let optical_offset = data.len() as u32;
        for i in 0..c.optical[0] {
            let u = unit(i as usize, c.optical[0] as usize);
            let h = if c.mapping_flags & 16 != 0 {
                mapping::optical_height(u, &optical_segments)
            } else {
                reference_mapping::height(g, u)
            };
            data.push([h.clamp(0.0, g.top_height()), 0.0, 0.0, 0.0]);
        }
        let cone_offset = data.len() as u32;
        for i in 0..c.cones {
            let u = unit(i as usize, c.cones as usize);
            let cosine = if c.mapping_flags & 8 != 0 {
                1.0 - 2.0 * mapping::inverse(mapping::cone_coord, u, 0.0, 1.0)
            } else {
                (PI * u).cos()
            };
            data.push([cosine, (1.0 - cosine * cosine).max(0.0).sqrt(), 0.0, 0.0]);
        }
        let low_count = if c.compact_rayleigh {
            heights.partition_point(|&h| h <= 35.0) as u32
        } else {
            c.heights
        };
        let source_coeff_offset = data.len() as u32;
        for &h in &heights {
            let coeff = w.indices.map(|i| model.bands[i].coefficients(h));
            let mut sum = [0.0; 4];
            for species in 0..5 {
                let row = coeff.map(|v| v.scattering[species]);
                for k in 0..4 {
                    sum[k] += row[k];
                }
                data.push(row);
            }
            data.push(sum);
        }
        let params = Params {
            size_steps: [1, 1, c.ray_steps, 1],
            view: [0.0; 4],
            sun: [0.0, 0.0, model.sun_radius, g.bottom],
            dims: [c.heights, c.suns, c.optical[0], c.optical[1]],
            offsets0: [source_coeff_offset, 0, profile_offset, phase_offset],
            offsets1: [phase_mass_offset, low_count, profile.len() as u32, 0],
            medium: [g.top_height(), albedo, 0.0, c.horizon_power],
            solar: w.solar,
            rgb: w.rgb.map(|v| [v[0], v[1], v[2], 0.0]),
            solve: [c.samples(), c.phases, c.cones, 0],
            extra: [
                states_offset,
                angles_offset,
                phase_nodes_offset,
                c.optical_steps,
            ],
            mapping: [c.mapping_flags, solar_offset, optical_offset, cone_offset],
            sky: [
                c.sky_size,
                c.sky_size,
                c.sky_size * 3 / 8,
                c.sky_size * 3 / 4,
            ],
            optical_segments,
            angular_mapping: mapping::angular_parameters(),
            importance: [
                c.sun_importance_width,
                c.path_height_scale,
                c.high_sun_weight,
                0.0,
            ],
            batch: [0, c.heights, 0, 0],
        };
        let mut key_bytes = bytemuck::cast_slice(&data).to_vec();
        key_bytes.extend_from_slice(bytemuck::bytes_of(&params));
        let key = checksum(&key_bytes);
        Ok(Self { data, params, key })
    }
}
/// Payload estimate before creating a device; excludes screen-sized targets and
/// driver allocation overhead. Startup temporarily retains two incident fields.
pub fn estimated_resident_bytes(
    model: &Model,
    w: &Wavelengths,
    albedo: f32,
    c: &Config,
) -> Result<u64> {
    Ok(Medium::new(model, w, c, albedo)?.resident_bytes(c))
}
fn texture(
    d: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
fn uniform(d: &wgpu::Device) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hybrid parameters"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn buffer(d: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}
fn storage(d: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}
fn group(
    d: &wgpu::Device,
    p: &wgpu::ComputePipeline,
    entries: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hybrid pass resources"),
        layout: &p.get_bind_group_layout(0),
        entries: &entries
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
    })
}
fn tv(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}
struct Field {
    source: wgpu::Texture,
    source_view: wgpu::TextureView,
    _mean: wgpu::Texture,
    mean_view: wgpu::TextureView,
    _ground: wgpu::Texture,
    ground_view: wgpu::TextureView,
    _high: wgpu::Texture,
    high_view: wgpu::TextureView,
}
impl Field {
    fn new(d: &wgpu::Device, c: &Config, low_count: u32) -> Self {
        let source = texture(
            d,
            "directional MS source",
            [c.suns * c.phases, low_count * c.cones],
            wgpu::TextureFormat::Rgba16Float,
        );
        let mean = texture(
            d,
            "log incident mean",
            [c.suns, c.heights],
            wgpu::TextureFormat::Rgba32Float,
        );
        let ground = texture(
            d,
            "ground radiance",
            [c.suns, 1],
            wgpu::TextureFormat::Rgba32Float,
        );
        let source_view = source.create_view(&Default::default());
        let mean_view = mean.create_view(&Default::default());
        let ground_view = ground.create_view(&Default::default());
        let high = texture(
            d,
            "exact high-altitude Rayleigh moments",
            [c.suns * 5, c.heights - low_count + 1],
            wgpu::TextureFormat::Rgba32Float,
        );
        let high_view = high.create_view(&Default::default());
        Self {
            source,
            source_view,
            _mean: mean,
            mean_view,
            _ground: ground,
            ground_view,
            _high: high,
            high_view,
        }
    }
}
#[derive(Default, Debug, Clone, Serialize)]
pub struct CacheStats {
    pub medium_solves: u32,
    pub sky_updates: u32,
    pub projections: u32,
}
#[derive(Debug, Clone, Serialize)]
pub struct IterationTiming {
    pub iteration: u32,
    pub stage_gpu_ms: Vec<f32>,
    pub submit_wait_ms: f32,
}
#[derive(Debug, Clone, Serialize)]
pub struct SolveReport {
    pub medium_key: String,
    pub wall_seconds: f32,
    pub iterations: Vec<IterationTiming>,
    pub resident_bytes: u64,
    pub peak_payload_bytes: u64,
    pub stages: Vec<&'static str>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct SkyKey {
    height: f32,
    sun: f32,
    steps: u32,
    multiple: bool,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct FrameKey {
    view: View,
    sky: SkyKey,
    include_sun: bool,
    sky_view: bool,
    segment: f32,
    spectral: bool,
}

pub struct Renderer {
    config: Config,
    medium: Medium,
    data: wgpu::Buffer,
    pipelines: Vec<wgpu::ComputePipeline>,
    field: Option<Field>,
    _tau: wgpu::Texture,
    tau_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    _sky: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_params: wgpu::Buffer,
    sky_mapping: wgpu::Buffer,
    frame_params: wgpu::Buffer,
    sky_group: Option<wgpu::BindGroup>,
    project_group: Option<wgpu::BindGroup>,
    direct_group: Option<wgpu::BindGroup>,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    transmittance: wgpu::Texture,
    transmittance_view: wgpu::TextureView,
    size: [u32; 2],
    last_sky: Option<SkyKey>,
    last_frame: Option<FrameKey>,
    pub steps: u32,
    pub use_sky_view: bool,
    pub include_sun_disk: bool,
    pub multiple_scattering: bool,
    pub segment_km: f32,
    pub spectral_output: bool,
    pub stats: CacheStats,
}
impl Renderer {
    pub fn new(
        d: &wgpu::Device,
        model: &Model,
        w: &Wavelengths,
        albedo: f32,
        config: Config,
    ) -> Result<Self> {
        let medium = Medium::new(model, w, &config, albedo)?;
        let source = shader_source();
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hybrid four-wave solver"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = [
            "optical_depth",
            "prepare_directions",
            "trace_incident",
            "incident_moments",
            "scattering_source",
            "ground_irradiance",
            "build_sky",
            "project_sky",
            "render",
        ]
        .iter()
        .map(|&entry| {
            let constants = [
                ("SPECIALIZE", config.specialize_constants as u32 as f64),
                ("FIXED_MAPPING_FLAGS", config.mapping_flags as f64),
                (
                    "FIXED_PHASE_WEIGHT",
                    medium.params.angular_mapping[0] as f64,
                ),
                ("FIXED_CONE_WARP", medium.params.angular_mapping[1] as f64),
            ];
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        })
        .collect();
        let data = buffer(
            d,
            "physical medium and quadrature",
            bytemuck::cast_slice(&medium.data),
        );
        let tau = texture(
            d,
            "hybrid optical depth",
            [config.optical[1], config.optical[0]],
            wgpu::TextureFormat::Rgba16Float,
        );
        let tau_view = tau.create_view(&Default::default());
        let sky = texture(
            d,
            "hybrid SkyView",
            [config.sky_size, config.sky_size],
            wgpu::TextureFormat::Rgba32Float,
        );
        let sky_view = sky.create_view(&Default::default());
        let target = texture(d, "hybrid screen", [1, 1], wgpu::TextureFormat::Rgba32Float);
        let target_view = target.create_view(&Default::default());
        let transmittance = texture(
            d,
            "hybrid segment transmittance",
            [1, 1],
            wgpu::TextureFormat::Rgba32Float,
        );
        let transmittance_view = transmittance.create_view(&Default::default());
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hybrid linear interpolation"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sky_mapping = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cached SkyView coordinate rows"),
            size: (12 + config.sky_size as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            config,
            medium,
            data,
            pipelines,
            field: None,
            _tau: tau,
            tau_view,
            sampler,
            _sky: sky,
            sky_view,
            sky_params: uniform(d),
            sky_mapping,
            frame_params: uniform(d),
            sky_group: None,
            project_group: None,
            direct_group: None,
            target,
            target_view,
            transmittance,
            transmittance_view,
            size: [1, 1],
            last_sky: None,
            last_frame: None,
            steps: 96,
            use_sky_view: true,
            include_sun_disk: false,
            multiple_scattering: true,
            segment_km: 0.0,
            spectral_output: false,
            stats: CacheStats::default(),
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    fn field_bytes(&self) -> u64 {
        self.medium.field_bytes(&self.config)
    }
    pub fn resident_bytes(&self) -> u64 {
        self.medium.resident_bytes(&self.config)
    }
    /// Diagnostic export of the actual GPU optical-depth table (RGBA16F).
    pub fn export_optical(&self, d: &wgpu::Device, q: &wgpu::Queue) -> Result<Vec<u8>> {
        read_texture(d, q, &self._tau, 8)
    }
    /// Reuses the solution when all physical inputs are unchanged. Camera and
    /// Sun direction never enter the key; all solar elevations are solved.
    pub fn set_medium(
        &mut self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        model: &Model,
        w: &Wavelengths,
        albedo: f32,
    ) -> Result<Option<SolveReport>> {
        let medium = Medium::new(model, w, &self.config, albedo)?;
        if self.field.is_some() && medium.key == self.medium.key {
            return Ok(None);
        }
        self.data = buffer(
            d,
            "changed physical medium",
            bytemuck::cast_slice(&medium.data),
        );
        self.medium = medium;
        Ok(Some(self.rebuild(d, q)?))
    }
    /// Only initialization/physical changes invoke this work. Temporary ray
    /// fields, moments, and the spare 4D source are released after convergence.
    pub fn rebuild(&mut self, d: &wgpu::Device, q: &wgpu::Queue) -> Result<SolveReport> {
        let start = Instant::now();
        let c = &self.config;
        let batch_heights = if c.batch_heights == 0 {
            c.heights
        } else {
            c.batch_heights.min(c.heights)
        };
        let count = batch_heights as u64 * c.suns as u64 * c.samples() as u64;
        if count * 16 > d.limits().max_storage_buffer_binding_size as u64 {
            return Err("incident field exceeds device binding size".into());
        }
        self.field = None;
        self.sky_group = None;
        self.direct_group = None;
        let incoming = storage(d, "temporary incident radiance", count * 16);
        let directions = storage(d, "temporary angular rays", count * 16);
        let moments = storage(
            d,
            "temporary Rayleigh moments",
            batch_heights as u64 * c.suns as u64 * 7 * 16,
        );
        let low_count = self.medium.params.offsets1[1];
        let mut fields = vec![Field::new(d, c, low_count), Field::new(d, c, low_count)];
        let initial_uniform = uniform(d);
        let common = [
            (0, initial_uniform.as_entire_binding()),
            (1, self.data.as_entire_binding()),
        ];
        let tau_group = group(
            d,
            &self.pipelines[0],
            &[
                common[0].clone(),
                common[1].clone(),
                (11, tv(&self.tau_view)),
            ],
        );
        let mut batches = Vec::new();
        for first in (0..c.heights).step_by(batch_heights as usize) {
            let uniform = uniform(d);
            let common = [
                (0, uniform.as_entire_binding()),
                (1, self.data.as_entire_binding()),
            ];
            let directions_group = group(
                d,
                &self.pipelines[1],
                &[
                    common[0].clone(),
                    common[1].clone(),
                    (16, directions.as_entire_binding()),
                ],
            );
            let trace_groups: Vec<_> = fields
                .iter()
                .map(|f| {
                    group(
                        d,
                        &self.pipelines[2],
                        &[
                            common[0].clone(),
                            common[1].clone(),
                            (2, tv(&self.tau_view)),
                            (3, tv(&f.source_view)),
                            (4, tv(&f.mean_view)),
                            (5, tv(&f.ground_view)),
                            (6, wgpu::BindingResource::Sampler(&self.sampler)),
                            (7, incoming.as_entire_binding()),
                            (16, directions.as_entire_binding()),
                            (18, tv(&f.high_view)),
                        ],
                    )
                })
                .collect();
            let mean_groups: Vec<_> = fields
                .iter()
                .map(|f| {
                    group(
                        d,
                        &self.pipelines[3],
                        &[
                            common[0].clone(),
                            (7, incoming.as_entire_binding()),
                            (9, tv(&f.mean_view)),
                            (16, directions.as_entire_binding()),
                            (17, moments.as_entire_binding()),
                            (19, tv(&f.high_view)),
                        ],
                    )
                })
                .collect();
            let scatter_groups: Vec<_> = fields
                .iter()
                .map(|f| {
                    group(
                        d,
                        &self.pipelines[4],
                        &[
                            common[0].clone(),
                            common[1].clone(),
                            (4, tv(&f.mean_view)),
                            (7, incoming.as_entire_binding()),
                            (8, tv(&f.source_view)),
                            (16, directions.as_entire_binding()),
                            (17, moments.as_entire_binding()),
                        ],
                    )
                })
                .collect();
            let ground_groups: Vec<_> = fields
                .iter()
                .map(|f| {
                    group(
                        d,
                        &self.pipelines[5],
                        &[
                            common[0].clone(),
                            common[1].clone(),
                            (2, tv(&self.tau_view)),
                            (6, wgpu::BindingResource::Sampler(&self.sampler)),
                            (7, incoming.as_entire_binding()),
                            (10, tv(&f.ground_view)),
                            (16, directions.as_entire_binding()),
                        ],
                    )
                })
                .collect();
            let mut params = self.medium.params;
            params.batch = [first, batch_heights.min(c.heights - first), 0, 0];
            q.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
            batches.push((
                uniform,
                params,
                directions_group,
                trace_groups,
                mean_groups,
                scatter_groups,
                ground_groups,
            ));
        }
        q.write_buffer(&initial_uniform, 0, bytemuck::bytes_of(&self.medium.params));
        let mut encoder = d.create_command_encoder(&Default::default());
        dispatch(
            &mut encoder,
            &self.pipelines[0],
            &tau_group,
            [c.optical[1].div_ceil(8), c.optical[0].div_ceil(8)],
            None,
        );
        dispatch(
            &mut encoder,
            &self.pipelines[5],
            &batches[0].6[0],
            [c.suns.div_ceil(64), 1],
            None,
        );
        q.submit([encoder.finish()]);
        d.poll(wgpu::PollType::wait_indefinitely())?;
        let timestamps = d.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let queries = timestamps.then(|| {
            d.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("hybrid solve stages"),
                ty: wgpu::QueryType::Timestamp,
                count: 8 * batches.len() as u32,
            })
        });
        let resolved = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solve timing resolve"),
            size: 64 * batches.len() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut iterations = Vec::new();
        let mut current = 0;
        for iteration in 1..=c.iterations {
            let now = Instant::now();
            let next = 1 - current;
            let mut encoder = d.create_command_encoder(&Default::default());
            for (
                batch_index,
                (
                    uniform,
                    params,
                    directions_group,
                    trace_groups,
                    mean_groups,
                    scatter_groups,
                    ground_groups,
                ),
            ) in batches.iter_mut().enumerate()
            {
                params.solve[3] = iteration;
                q.write_buffer(uniform, 0, bytemuck::bytes_of(params));
                let batch_states = params.batch[1] * c.suns;
                dispatch(
                    &mut encoder,
                    &self.pipelines[1],
                    directions_group,
                    [c.samples().div_ceil(64), batch_states],
                    None,
                );
                for (k, (pipeline, binding, groups)) in [
                    (
                        2,
                        &trace_groups[current],
                        [c.samples().div_ceil(64), batch_states],
                    ),
                    (3, &mean_groups[next], [batch_states.div_ceil(64), 1]),
                    (
                        4,
                        &scatter_groups[next],
                        [
                            (c.suns * c.phases).div_ceil(8),
                            (low_count
                                .saturating_sub(params.batch[0])
                                .min(params.batch[1])
                                * c.cones)
                                .div_ceil(8),
                        ],
                    ),
                    (5, &ground_groups[next], [c.suns.div_ceil(64), 1]),
                ]
                .into_iter()
                .enumerate()
                {
                    dispatch(
                        &mut encoder,
                        &self.pipelines[pipeline],
                        binding,
                        groups,
                        queries
                            .as_ref()
                            .map(|set| (set, batch_index as u32 * 8 + k as u32 * 2)),
                    );
                }
            }
            if let Some(set) = &queries {
                encoder.resolve_query_set(set, 0..8 * batches.len() as u32, &resolved, 0);
            }
            q.submit([encoder.finish()]);
            d.poll(wgpu::PollType::wait_indefinitely())?;
            let elapsed = now.elapsed().as_secs_f32() * 1000.0;
            let ms = if timestamps {
                let bytes = read_buffer(d, q, &resolved, 64 * batches.len() as u64)?;
                let times: &[u64] = bytemuck::cast_slice(&bytes);
                let mut sum = vec![0.0; 4];
                for (i, t) in times.chunks_exact(2).enumerate() {
                    sum[i % 4] += (t[1] - t[0]) as f32 * q.get_timestamp_period() * 1e-6;
                }
                sum
            } else {
                Vec::new()
            };
            eprintln!(
                "hybrid iteration {iteration}/{}: {elapsed:.1} ms, GPU stages {:?}",
                c.iterations, ms
            );
            iterations.push(IterationTiming {
                iteration,
                stage_gpu_ms: ms,
                submit_wait_ms: elapsed,
            });
            current = next;
        }
        let resident_bytes = self.resident_bytes();
        let peak_payload_bytes = resident_bytes
            + self.field_bytes()
            + count * 32
            + batch_heights as u64 * c.suns as u64 * 7 * 16;
        self.field = Some(fields.swap_remove(current));
        self.last_sky = None;
        self.last_frame = None;
        self.stats.medium_solves += 1;
        self.bind_runtime(d);
        Ok(SolveReport {
            medium_key: self.medium.key.clone(),
            wall_seconds: start.elapsed().as_secs_f32(),
            iterations,
            resident_bytes,
            peak_payload_bytes,
            stages: vec![
                "trace_incident",
                "incident_moments",
                "scattering_source",
                "ground_irradiance",
            ],
        })
    }
    fn bind_runtime(&mut self, d: &wgpu::Device) {
        let Some(f) = &self.field else {
            return;
        };
        let mut common = vec![
            (0, self.sky_params.as_entire_binding()),
            (1, self.data.as_entire_binding()),
            (2, tv(&self.tau_view)),
            (3, tv(&f.source_view)),
            (4, tv(&f.mean_view)),
            (5, tv(&f.ground_view)),
            (6, wgpu::BindingResource::Sampler(&self.sampler)),
            (18, tv(&f.high_view)),
        ];
        common.push((14, tv(&self.sky_view)));
        common.push((20, self.sky_mapping.as_entire_binding()));
        self.sky_group = Some(group(d, &self.pipelines[6], &common));
        common.pop();
        common.pop();
        common[0] = (0, self.frame_params.as_entire_binding());
        common.extend([
            (12, tv(&self.target_view)),
            (13, tv(&self.transmittance_view)),
        ]);
        self.direct_group = Some(group(d, &self.pipelines[8], &common));
        self.project_group = Some(group(
            d,
            &self.pipelines[7],
            &[
                (0, self.frame_params.as_entire_binding()),
                (2, tv(&self.tau_view)),
                (6, wgpu::BindingResource::Sampler(&self.sampler)),
                (12, tv(&self.target_view)),
                (15, tv(&self.sky_view)),
                (20, self.sky_mapping.as_entire_binding()),
            ],
        ));
    }
    pub fn resize(&mut self, d: &wgpu::Device, size: [u32; 2]) -> bool {
        let ts = if self.use_sky_view && self.segment_km <= 0.0 && !self.spectral_output {
            [1, 1]
        } else {
            size
        };
        if size == self.size
            && self.transmittance.width() == ts[0]
            && self.transmittance.height() == ts[1]
        {
            return false;
        }
        self.size = size;
        self.target = texture(d, "hybrid screen", size, wgpu::TextureFormat::Rgba32Float);
        self.target_view = self.target.create_view(&Default::default());
        self.transmittance = texture(d, "hybrid segment T", ts, wgpu::TextureFormat::Rgba32Float);
        self.transmittance_view = self.transmittance.create_view(&Default::default());
        self.last_frame = None;
        self.bind_runtime(d);
        true
    }
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.target_view
    }
    pub fn target_texture(&self) -> &wgpu::Texture {
        &self.target
    }
    pub fn transmittance_texture(&self) -> &wgpu::Texture {
        &self.transmittance
    }
    pub fn render(&mut self, q: &wgpu::Queue, e: &mut wgpu::CommandEncoder, view: View) {
        assert!(
            self.field.is_some(),
            "call rebuild before rendering the hybrid atmosphere"
        );
        let sky = SkyKey {
            height: view.altitude_km.max(0.0),
            sun: view.sun_elevation_deg,
            steps: self.steps.clamp(4, 512),
            multiple: self.multiple_scattering,
        };
        let use_sky_view = self.use_sky_view && self.segment_km <= 0.0 && !self.spectral_output;
        let frame = FrameKey {
            view,
            sky,
            include_sun: self.include_sun_disk,
            sky_view: use_sky_view,
            segment: self.segment_km,
            spectral: self.spectral_output,
        };
        if self.last_frame == Some(frame) {
            return;
        }
        self.last_frame = Some(frame);
        let sky_rows = self.config.sky_size * 3 / 4;
        let lower_rows =
            if self.config.mapping_flags & 32 != 0 && sky.height >= self.medium.params.medium[0] {
                sky_rows
            } else {
                sky_rows / 2
            };
        let params = |view: View, size: [u32; 2], sun: bool| {
            let mut p = self.medium.params;
            p.size_steps = [
                size[0],
                size[1],
                sky.steps,
                self.multiple_scattering as u32
                    | ((sun as u32) << 1)
                    | ((self.spectral_output as u32) << 2),
            ];
            p.view = [
                view.yaw_deg.to_radians(),
                view.pitch_deg.to_radians(),
                view.fov_y_deg.to_radians(),
                sky.height,
            ];
            p.sun[0] = view.sun_azimuth_deg.to_radians();
            p.sun[1] = view.sun_elevation_deg.to_radians();
            p.medium[2] = self.segment_km;
            p.sky = [
                self.config.sky_size,
                self.config.sky_size,
                lower_rows,
                sky_rows,
            ];
            p
        };
        if use_sky_view && self.last_sky != Some(sky) {
            let g = geometry::Geometry {
                bottom: self.medium.params.sun[3],
                top: self.medium.params.sun[3] + self.medium.params.medium[0],
            };
            let (cache, _) = mapping::sky_cache(
                g,
                sky.height,
                view.sun_elevation_deg.to_radians(),
                self.config.sky_size,
                self.config.mapping_flags & 32 != 0,
            );
            q.write_buffer(&self.sky_mapping, 0, bytemuck::cast_slice(&cache));
            q.write_buffer(
                &self.sky_params,
                0,
                bytemuck::bytes_of(&params(
                    View {
                        sun_azimuth_deg: 0.0,
                        ..view
                    },
                    [self.config.sky_size, self.config.sky_size],
                    false,
                )),
            );
            dispatch(
                e,
                &self.pipelines[6],
                self.sky_group.as_ref().unwrap(),
                [
                    self.config.sky_size.div_ceil(8),
                    self.config.sky_size.div_ceil(8),
                ],
                None,
            );
            self.stats.sky_updates += 1;
            self.last_sky = Some(sky);
        }
        q.write_buffer(
            &self.frame_params,
            0,
            bytemuck::bytes_of(&params(view, self.size, self.include_sun_disk)),
        );
        let (p, g) = if use_sky_view {
            (&self.pipelines[7], self.project_group.as_ref().unwrap())
        } else {
            (&self.pipelines[8], self.direct_group.as_ref().unwrap())
        };
        dispatch(
            e,
            p,
            g,
            [self.size[0].div_ceil(8), self.size[1].div_ceil(8)],
            None,
        );
        self.stats.projections += 1;
    }
    /// Small diagnostic export permits auditing the actual solved source and
    /// iteration deltas without retaining any temporary incident ray fields.
    pub fn export_source(&self, d: &wgpu::Device, q: &wgpu::Queue) -> Result<Vec<u8>> {
        read_texture(d, q, &self.field.as_ref().ok_or("not solved")?.source, 8)
    }
}
fn dispatch(
    e: &mut wgpu::CommandEncoder,
    p: &wgpu::ComputePipeline,
    g: &wgpu::BindGroup,
    size: [u32; 2],
    timer: Option<(&wgpu::QuerySet, u32)>,
) {
    let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("hybrid atmosphere pass"),
        timestamp_writes: timer.map(|(query_set, index)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }),
    });
    pass.set_pipeline(p);
    pass.set_bind_group(0, g, &[]);
    pass.dispatch_workgroups(size[0], size[1], 1);
}
pub fn read_buffer(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    b: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>> {
    let target = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hybrid diagnostic readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(b, 0, &target, 0, size);
    q.submit([e.finish()]);
    let (tx, rx) = mpsc::channel();
    target.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    Ok(target.slice(..).get_mapped_range().to_vec())
}
pub fn read_texture(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    t: &wgpu::Texture,
    bpp: u32,
) -> Result<Vec<u8>> {
    let width = t.width();
    let height = t.height();
    let row = (width * bpp).div_ceil(256) * 256;
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hybrid texture readback"),
        size: (row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: t,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        t.size(),
    );
    q.submit([e.finish()]);
    let padded = read_buffer(d, q, &staging, (row * height) as u64)?;
    let mut result = Vec::with_capacity((width * height * bpp) as usize);
    for y in 0..height {
        result.extend_from_slice(&padded[(y * row) as usize..(y * row + width * bpp) as usize]);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn published_balanced_config_matches_demo() {
        let c: super::Config =
            serde_json::from_str(include_str!("../configs/balanced.json")).unwrap();
        assert_eq!(c, super::Config::balanced());
        assert_eq!(c.suns % 16, 0);
    }
    #[test]
    fn validate_shader() {
        let s = super::shader_source();
        let module =
            naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

pub mod physics;

pub fn checksum(bytes: &[u8]) -> String {
    let mut h = 0xcbf29ce484222325_u64;
    for &v in bytes {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
