//! Runtime path: immutable directional MS atlas -> observer SkyView -> screen.
//! The full SH payload is used once to expand the atlas, then released. Only
//! six Rayleigh moments per high-altitude state remain resident at runtime.
use super::{
    Resource,
    renderer::{Params, SHADER},
};
use crate::{Result, renderer::View};
use std::path::Path;
use wgpu::util::DeviceExt;

pub const PHASE_COUNT: u32 = 24;
pub const CONE_COUNT: u32 = 8;
pub const SKY_SIZE: [u32; 2] = [256, 256];
pub const SKY_ROWS: u32 = 192;
pub fn shader_source() -> String {
    let mut s = SHADER.replace("fn indirect(", "fn indirect_sh(");
    let cache = include_str!("cache.wgsl")
        .replace(
            "const PHASE_COUNT:u32=24u;",
            &format!("const PHASE_COUNT:u32={PHASE_COUNT}u;"),
        )
        .replace(
            "const CONE_COUNT:u32=8u;",
            &format!("const CONE_COUNT:u32={CONE_COUNT}u;"),
        )
        .replace(
            "const SKY_SIZE:vec2<u32>=vec2<u32>(256u,192u);",
            &format!(
                "const SKY_SIZE:vec2<u32>=vec2<u32>({}u,{}u);",
                SKY_SIZE[0], SKY_SIZE[1]
            ),
        )
        .replace(
            "const SKY_ROWS:u32=144u;",
            &format!("const SKY_ROWS:u32={SKY_ROWS}u;"),
        );
    s.push_str(&cache);
    s
}
#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub struct CacheStats {
    pub ms_builds: u32,
    pub sky_updates: u32,
    pub projections: u32,
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
}
struct Initialization {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
}
pub struct CachedRenderer {
    pub resource: Resource,
    initialization: Option<Initialization>,
    payloads: [wgpu::Buffer; 3],
    _ms: wgpu::Texture,
    ms_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    _sky: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_params: wgpu::Buffer,
    frame_params: wgpu::Buffer,
    sky_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    direct_pipeline: wgpu::ComputePipeline,
    sky_group: wgpu::BindGroup,
    project_group: wgpu::BindGroup,
    direct_group: wgpu::BindGroup,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    _transmittance: wgpu::Texture,
    transmittance_view: wgpu::TextureView,
    size: [u32; 2],
    ms_size: [u32; 2],
    last_sky: Option<SkyKey>,
    last_frame: Option<FrameKey>,
    pub steps: u32,
    pub multiple_scattering: bool,
    pub include_sun_disk: bool,
    /// False isolates the immutable MS atlas from the 2D SkyView approximation.
    pub use_sky_view: bool,
    pub stats: CacheStats,
    pub resident_lut_bytes: u64,
}
fn texture(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
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
fn buffer(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}
fn uniform(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("four-wave cache params"),
        size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("four-wave cache bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
    })
}
fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn params(
    r: &Resource,
    size: [u32; 2],
    view: View,
    steps: u32,
    multiple: bool,
    sun: bool,
) -> Params {
    Params {
        size_steps: [
            size[0],
            size[1],
            steps.clamp(4, 512),
            multiple as u32 | ((sun as u32) << 1),
        ],
        view: [
            view.yaw_deg.to_radians(),
            view.pitch_deg.to_radians(),
            view.fov_y_deg.to_radians(),
            view.altitude_km.max(0.0),
        ],
        sun: [
            view.sun_azimuth_deg.to_radians(),
            view.sun_elevation_deg.to_radians(),
            r.sun_radius,
            r.geometry.bottom,
        ],
        dims: [
            r.heights.len() as u32,
            r.sun_count as u32,
            r.optical[0] as u32,
            r.optical[1] as u32,
        ],
        offsets0: std::array::from_fn(|i| r.aux_offsets[i] as u32),
        offsets1: [
            r.aux_offsets[4] as u32,
            r.aux_offsets[5] as u32,
            r.profile_count as u32,
            r.ground_count as u32,
        ],
        medium: [r.geometry.top_height(), r.albedo, 0.0, 0.0],
        solar: r.sun_per_nm,
        rgb: r.rgb_from_per_nm.map(|v| [v[0], v[1], v[2], 0.0]),
    }
}
pub fn resident_bytes(r: &Resource) -> u64 {
    let low = r.heights.iter().filter(|&&h| h <= 35.0).count();
    let high = r.heights.iter().filter(|&&h| h >= 35.0).count();
    let auxiliary = r.files.iter().find(|(n, _, _)| n == "aux.f32").unwrap().1;
    (low * r.sun_count * PHASE_COUNT as usize * CONE_COUNT as usize * 8
        + high * r.sun_count * 6 * 8
        + auxiliary
        + r.optical[0] * r.optical[1] * 8
        + SKY_SIZE[0] as usize * SKY_SIZE[1] as usize * 16) as u64
}
impl CachedRenderer {
    pub fn new(device: &wgpu::Device, path: &Path) -> Result<Self> {
        Self::new_with_sun_model(device, path, super::solar::DEFAULT_SUN_MODEL)
    }

    pub fn new_with_sun_model(
        device: &wgpu::Device,
        path: &Path,
        model: super::solar::SunModel,
    ) -> Result<Self> {
        Self::new_with_shader_for_diagnostics(
            device,
            path,
            &super::solar::shader_for_model(&shader_source(), model),
        )
    }

    /// Cost-ablation harness only. The shader must preserve the entry points
    /// and bindings of `shader_source`; the demo always uses the default source.
    #[doc(hidden)]
    pub fn new_with_shader_for_diagnostics(
        device: &wgpu::Device,
        path: &Path,
        source: &str,
    ) -> Result<Self> {
        let resource = Resource::open(path)?;
        if !resource.heights.contains(&35.0) {
            return Err("four-wave cache requires the 35 km aerosol boundary node".into());
        }
        let full_moments = resource.read(path, "moments.u32")?;
        let full_aux = resource.read(path, "aux.f32")?;
        let optical = resource.read(path, "optical.u32")?;
        let mut aux: Vec<[f32; 4]> = full_aux
            .chunks_exact(16)
            .map(|b| {
                std::array::from_fn(|i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect();
        let mut high_moments = Vec::new();
        for (i, &h) in resource.heights.iter().enumerate() {
            let offset = high_moments.len() / 4;
            if h >= 35.0 {
                let count = crate::anisotropic::count(resource.degrees[i]);
                for s in 0..resource.sun_count {
                    let start = (resource.offsets[i] + s * count * 2) * 4;
                    high_moments.extend_from_slice(&full_moments[start..start + 6 * 8]);
                }
            }
            aux[resource.aux_offsets[1] + i][1] = 2.0;
            aux[resource.aux_offsets[1] + i][2] = offset as f32;
        }
        let payloads = [
            buffer(device, "Rayleigh-only MS moments", &high_moments),
            buffer(
                device,
                "cached medium and scales",
                bytemuck::cast_slice(&aux),
            ),
            buffer(device, "four-wave optical depth", &optical),
        ];
        let mut nodes = Vec::new();
        for &h in &resource.heights {
            for s in 0..resource.sun_count {
                nodes.push([
                    crate::reference_mapping::solar_cosine(
                        resource.geometry,
                        h,
                        s as f32 / (resource.sun_count - 1) as f32,
                    ),
                    0.0,
                    0.0,
                    0.0,
                ]);
            }
        }
        for i in 0..PHASE_COUNT {
            let u = i as f32 / (PHASE_COUNT - 1) as f32;
            let (mut lo, mut hi) = (0.0_f32, std::f32::consts::PI);
            for _ in 0..30 {
                let t = (lo + hi) * 0.5;
                let v = 0.5 * t / std::f32::consts::PI + 0.5 * ((1.0 - t.cos()) * 0.5).cbrt();
                if v < u {
                    lo = t;
                } else {
                    hi = t;
                }
            }
            nodes.push([((lo + hi) * 0.5).cos(), 0.0, 0.0, 0.0]);
        }
        let low = resource.heights.iter().filter(|&&h| h <= 35.0).count() as u32;
        let ms_size = [PHASE_COUNT * resource.sun_count as u32, CONE_COUNT * low];
        if ms_size
            .iter()
            .any(|&n| n > device.limits().max_texture_dimension_2d)
        {
            return Err("MS cache exceeds texture dimension limit".into());
        }
        let ms = texture(
            device,
            "immutable directional MS atlas",
            ms_size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let ms_view = ms.create_view(&Default::default());
        let sky = texture(
            device,
            "observer SkyView",
            SKY_SIZE,
            wgpu::TextureFormat::Rgba32Float,
        );
        let sky_view = sky.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MS angular interpolation"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("four-wave cached transport"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bake_pipeline = pipeline(device, &module, "build_ms");
        let bake_params = uniform(device);
        let bake_moments = buffer(device, "temporary source expansion SH", &full_moments);
        let bake_aux = buffer(device, "temporary source expansion metadata", &full_aux);
        let bake_nodes = buffer(
            device,
            "source expansion directions",
            bytemuck::cast_slice(&nodes),
        );
        let bake_group = group(
            device,
            &bake_pipeline,
            &[
                (0, bake_params.as_entire_binding()),
                (1, bake_moments.as_entire_binding()),
                (2, bake_aux.as_entire_binding()),
                (8, bake_nodes.as_entire_binding()),
                (9, wgpu::BindingResource::TextureView(&ms_view)),
            ],
        );
        let sky_pipeline = pipeline(device, &module, "build_sky");
        let project_pipeline = pipeline(device, &module, "project_sky");
        let direct_pipeline = pipeline(device, &module, "render");
        let sky_params = uniform(device);
        let frame_params = uniform(device);
        let sky_group = group(
            device,
            &sky_pipeline,
            &[
                (0, sky_params.as_entire_binding()),
                (1, payloads[0].as_entire_binding()),
                (2, payloads[1].as_entire_binding()),
                (3, payloads[2].as_entire_binding()),
                (6, wgpu::BindingResource::TextureView(&ms_view)),
                (7, wgpu::BindingResource::Sampler(&sampler)),
                (10, wgpu::BindingResource::TextureView(&sky_view)),
            ],
        );
        let target = texture(
            device,
            "cached sky screen",
            [1, 1],
            wgpu::TextureFormat::Rgba32Float,
        );
        let target_view = target.create_view(&Default::default());
        let transmittance = texture(
            device,
            "direct source audit transmittance",
            [1, 1],
            wgpu::TextureFormat::Rgba32Float,
        );
        let transmittance_view = transmittance.create_view(&Default::default());
        let project_group = group(
            device,
            &project_pipeline,
            &[
                (0, frame_params.as_entire_binding()),
                (3, payloads[2].as_entire_binding()),
                (4, wgpu::BindingResource::TextureView(&target_view)),
                (11, wgpu::BindingResource::TextureView(&sky_view)),
            ],
        );
        let direct_group = group(
            device,
            &direct_pipeline,
            &[
                (0, frame_params.as_entire_binding()),
                (1, payloads[0].as_entire_binding()),
                (2, payloads[1].as_entire_binding()),
                (3, payloads[2].as_entire_binding()),
                (4, wgpu::BindingResource::TextureView(&target_view)),
                (5, wgpu::BindingResource::TextureView(&transmittance_view)),
                (6, wgpu::BindingResource::TextureView(&ms_view)),
                (7, wgpu::BindingResource::Sampler(&sampler)),
            ],
        );
        let resident_lut_bytes = resident_bytes(&resource);
        eprintln!(
            "four-wave fixed MS + SkyView: {:.3} MiB resident LUTs; temporary SH released after expansion",
            resident_lut_bytes as f32 / 1048576.0
        );
        Ok(Self {
            resource,
            initialization: Some(Initialization {
                pipeline: bake_pipeline,
                group: bake_group,
                params: bake_params,
            }),
            payloads,
            _ms: ms,
            ms_view,
            sampler,
            _sky: sky,
            sky_view,
            sky_params,
            frame_params,
            sky_pipeline,
            project_pipeline,
            direct_pipeline,
            sky_group,
            project_group,
            direct_group,
            target,
            target_view,
            _transmittance: transmittance,
            transmittance_view,
            size: [1, 1],
            ms_size,
            last_sky: None,
            last_frame: None,
            steps: 128,
            multiple_scattering: true,
            include_sun_disk: false,
            use_sky_view: true,
            stats: CacheStats::default(),
            resident_lut_bytes,
        })
    }
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        let transmittance_size = if self.use_sky_view { [1, 1] } else { size };
        if self.size == size
            && [self._transmittance.width(), self._transmittance.height()] == transmittance_size
        {
            return false;
        }
        self.size = size;
        self.last_frame = None;
        self.target = texture(
            device,
            "cached sky screen",
            size,
            wgpu::TextureFormat::Rgba32Float,
        );
        self.target_view = self.target.create_view(&Default::default());
        // The production SkyView path does not need a screen-sized spectral T.
        // Allocate it only for the explicit source-atlas accuracy diagnostic.
        self._transmittance = texture(
            device,
            "direct source audit transmittance",
            transmittance_size,
            wgpu::TextureFormat::Rgba32Float,
        );
        self.transmittance_view = self._transmittance.create_view(&Default::default());
        self.project_group = group(
            device,
            &self.project_pipeline,
            &[
                (0, self.frame_params.as_entire_binding()),
                (3, self.payloads[2].as_entire_binding()),
                (4, wgpu::BindingResource::TextureView(&self.target_view)),
                (11, wgpu::BindingResource::TextureView(&self.sky_view)),
            ],
        );
        self.direct_group = group(
            device,
            &self.direct_pipeline,
            &[
                (0, self.frame_params.as_entire_binding()),
                (1, self.payloads[0].as_entire_binding()),
                (2, self.payloads[1].as_entire_binding()),
                (3, self.payloads[2].as_entire_binding()),
                (4, wgpu::BindingResource::TextureView(&self.target_view)),
                (
                    5,
                    wgpu::BindingResource::TextureView(&self.transmittance_view),
                ),
                (6, wgpu::BindingResource::TextureView(&self.ms_view)),
                (7, wgpu::BindingResource::Sampler(&self.sampler)),
            ],
        );
        true
    }
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.target_view
    }
    pub fn target_texture(&self) -> &wgpu::Texture {
        &self.target
    }
    pub fn render(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view: View) {
        if let Some(init) = self.initialization.take() {
            let p = params(&self.resource, self.ms_size, view, self.steps, true, false);
            queue.write_buffer(&init.params, 0, bytemuck::bytes_of(&p));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("initialize immutable MS once"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&init.pipeline);
            pass.set_bind_group(0, &init.group, &[]);
            pass.dispatch_workgroups(self.ms_size[0].div_ceil(8), self.ms_size[1].div_ceil(8), 1);
            drop(pass);
            // Commands retain their resources until execution completes.
            self.stats.ms_builds += 1;
        }
        let sky = SkyKey {
            height: view.altitude_km.max(0.0),
            sun: view.sun_elevation_deg,
            steps: self.steps.clamp(4, 512),
            multiple: self.multiple_scattering,
        };
        let frame = FrameKey {
            view,
            sky,
            include_sun: self.include_sun_disk,
            sky_view: self.use_sky_view,
        };
        if self.last_frame == Some(frame) {
            return;
        }
        self.last_frame = Some(frame);
        if self.use_sky_view && self.last_sky != Some(sky) {
            let canonical = View {
                sun_azimuth_deg: 0.0,
                ..view
            };
            queue.write_buffer(
                &self.sky_params,
                0,
                bytemuck::bytes_of(&params(
                    &self.resource,
                    SKY_SIZE,
                    canonical,
                    self.steps,
                    self.multiple_scattering,
                    false,
                )),
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update observer SkyView"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.sky_group, &[]);
            pass.dispatch_workgroups(SKY_SIZE[0].div_ceil(8), SKY_SIZE[1].div_ceil(8), 1);
            drop(pass);
            self.last_sky = Some(sky);
            self.stats.sky_updates += 1;
        }
        queue.write_buffer(
            &self.frame_params,
            0,
            bytemuck::bytes_of(&params(
                &self.resource,
                self.size,
                view,
                self.steps,
                self.multiple_scattering,
                self.include_sun_disk,
            )),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("four-wave sky projection"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if self.use_sky_view {
            &self.project_pipeline
        } else {
            &self.direct_pipeline
        });
        pass.set_bind_group(
            0,
            if self.use_sky_view {
                &self.project_group
            } else {
                &self.direct_group
            },
            &[],
        );
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        self.stats.projections += 1;
    }
}
