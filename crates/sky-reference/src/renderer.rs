//! GPU display of spectral assets or exported linear Rec.2020 assets.
//! Spectral assets accumulate all bands; RGB assets require three channel passes.
use crate::{Result, asset::Manifest};
use bytemuck::{Pod, Zeroable};
use std::path::Path;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub fov_y_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub altitude_km: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameParams {
    size_band: [u32; 4],
    view: [f32; 4],
    sun_height: [f32; 4],
    rgb_radius: [f32; 4],
    storage: [u32; 4],
}

pub struct SpectralRenderer {
    pub manifest: Manifest,
    pipeline: wgpu::ComputePipeline,
    bands: Vec<BandBuffers>,
    dummy: wgpu::Buffer,
    angular_nodes: wgpu::Buffer,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    rgb_sum: wgpu::Buffer,
    bindings: Vec<wgpu::BindGroup>,
    size: [u32; 2],
    last_view: Option<View>,
    packed: Option<PackedBuffers>,
}
struct PackedBuffers {
    data: wgpu::Buffer,
    blocks: wgpu::Buffer,
}
struct BandBuffers {
    params: wgpu::Buffer,
    frame: wgpu::Buffer,
    radiance: wgpu::Buffer,
    tau: wgpu::Buffer,
    weight: [f32; 3],
}

/// Largest binding and total LUT payload (frame targets/reference textures excluded).
pub fn storage_budget(manifest: &Manifest) -> (u64, u64) {
    let c = &manifest.config;
    if let Some(p) = manifest.rgb.as_ref().and_then(|r| r.packed.as_ref()) {
        (
            ((p.data_words.max(p.map_words).max(c.optical_depth_len())) * 4) as u64,
            p.gpu_bytes(manifest),
        )
    } else {
        let channels = if manifest.rgb.is_some() {
            3
        } else {
            manifest.bands.len()
        };
        (
            (c.scattering_len().max(c.optical_depth_len()) * 4) as u64,
            ((c.scattering_len() + c.optical_depth_len()) * channels * 4) as u64,
        )
    }
}

/// Used by the renderer and CPU Naga validation tests.
pub fn shader_source(packed: bool) -> String {
    let mut source = include_str!("bake.wgsl").to_string();
    if packed {
        source = source.replace(
            "fn load_radiance(index:u32)->f32 { return previous[index]; }",
            "fn load_radiance(index:u32)->f32 { return packed_radiance(index); }",
        );
        source.push_str(include_str!("packed.wgsl"));
    }
    source.push_str(include_str!("render.wgsl"));
    source
}

impl SpectralRenderer {
    /// RGB weights should include the desired spectral-to-XYZ and white balance
    /// transform. For sky-realtime-demo they produce linear Rec.2020.
    pub fn new(device: &wgpu::Device, dir: &Path, weights: &[[f32; 3]]) -> Result<Self> {
        let manifest = Manifest::open(dir)?;
        if !manifest.complete()
            || weights.len() != manifest.bands.len()
            || weights.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("renderer needs a complete LUT and one finite RGB weight per band".into());
        }
        let c = &manifest.config;
        if let Some(rgb) = &manifest.rgb
            && rgb
                .weights
                .iter()
                .flatten()
                .zip(weights.iter().flatten())
                .any(|(a, b)| (a - b).abs() > 1e-5 * a.abs().max(b.abs()).max(1.0))
        {
            return Err("RGB LUT uses a different baked colour transform".into());
        }
        let (binding_bytes, total_bytes) = storage_budget(&manifest);
        eprintln!(
            "offline LUT: {} ({:.1} MiB resident payload)",
            dir.display(),
            total_bytes as f32 / 1048576.0
        );
        if binding_bytes > device.limits().max_storage_buffer_binding_size {
            return Err("LUT band exceeds GPU binding limit".into());
        }
        let is_packed = manifest.rgb.as_ref().is_some_and(|r| r.packed.is_some());
        let source = shader_source(is_packed);
        let compact = if is_packed {
            Some(crate::packed::PackedLut::read(&manifest, dir)?)
        } else {
            None
        };
        let packed = compact.as_ref().map(|lut| PackedBuffers {
            data: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed RGB radiance"),
                contents: bytemuck::cast_slice(&lut.data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            blocks: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("packed RGB block map"),
                contents: bytemuck::cast_slice(&lut.map),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        });
        // Shared with the baker's packed phase/angle binding; display only
        // needs the cached angular nodes, avoiding inverse-CDF work per pixel.
        let mut nodes = vec![0.0_f32; 4096];
        nodes.extend(
            (0..c.scattering[3]).map(|i| {
                crate::mapping::scattering_cosine(crate::mapping::unit(i, c.scattering[3]))
            }),
        );
        crate::reference_mapping::append_display_nodes(manifest.geometry, c, &mut nodes);
        let angular_nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LUT display cached angular nodes"),
            contents: bytemuck::cast_slice(&nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spectral LUT display"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("all-band LUT display"),
            layout: None,
            module: &shader,
            entry_point: Some("render_spectral"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut bands = Vec::new();
        let channels = if manifest.rgb.is_some() {
            3
        } else {
            weights.len()
        };
        for i in 0..channels {
            let (optical_depth, radiance, irradiance, weight) = if let Some(rgb) = &manifest.rgb {
                let (sun_table, radiance) = if let Some(lut) = &compact {
                    (lut.sun[i].clone(), vec![0.0])
                } else {
                    crate::rgb::read_channel(&manifest, dir, i)?
                };
                let mut identity = [0.0; 3];
                identity[i] = 1.0;
                (sun_table, radiance, rgb.solar_irradiance[i], identity)
            } else {
                let band = manifest.read_band(dir, i)?;
                (
                    band.optical_depth,
                    band.radiance,
                    band.info.solar_irradiance_w_m2,
                    weights[i],
                )
            };
            // Mirror bake Params without exposing solver implementation to callers.
            let mut bytes = Vec::new();
            for n in c
                .scattering
                .into_iter()
                .chain([
                    c.optical_depth[0],
                    c.optical_depth[1],
                    c.ground_sun_samples,
                    0,
                ])
                .chain([0; 8])
            {
                bytes.extend_from_slice(&(n as u32).to_ne_bytes());
            }
            for v in [
                manifest.geometry.bottom,
                manifest.geometry.top_height(),
                c.ground_albedo,
                irradiance,
            ] {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
            for v in [
                c.mapping as u32,
                c.solar_interpolation as u32
                    | ((c.phase_interpolation as u32) << 8)
                    | ((c.view_interpolation as u32) << 16)
                    | (u32::from(!c.scattering_altitudes_km.is_empty()) << 24)
                    | ((c.height_interpolation as u32) << 25),
                c.angular_integration as u32
                    | ((c.iteration_scheme as u32) << 8)
                    | ((c.source_mapping as u32) << 16),
                c.cone_mapping as u32,
            ] {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LUT display band parameters"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let frame = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("LUT display view"),
                size: std::mem::size_of::<FrameParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let upload = |label, data: &[f32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            bands.push(BandBuffers {
                params,
                frame,
                radiance: upload("offline radiance", &radiance),
                tau: upload("offline sun attenuation", &optical_depth),
                weight,
            });
        }
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unused scattering density"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (target, target_view, rgb_sum) = targets(device, [1, 1]);
        let mut result = Self {
            manifest,
            pipeline,
            bands,
            dummy,
            angular_nodes,
            target,
            target_view,
            rgb_sum,
            bindings: Vec::new(),
            size: [1, 1],
            last_view: None,
            packed,
        };
        result.rebind(device);
        Ok(result)
    }
    pub fn target_view(&self) -> &wgpu::TextureView {
        &self.target_view
    }
    pub fn target(&self) -> &wgpu::Texture {
        &self.target
    }
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        let requested = size.map(|v| v.max(1));
        let limits = device.limits();
        let max_pixels = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / 16;
        let scale = ((max_pixels as f32 / requested[0] as f32 / requested[1] as f32).sqrt())
            .min(limits.max_texture_dimension_2d as f32 / requested[0] as f32)
            .min(limits.max_texture_dimension_2d as f32 / requested[1] as f32)
            .min(1.0);
        // Very large desktop surfaces are displayed from a proportionally
        // smaller image instead of creating an invalid storage binding.
        let size = requested.map(|v| ((v as f32 * scale).floor() as u32).max(1));
        if size == self.size {
            return false;
        }
        (self.target, self.target_view, self.rgb_sum) = targets(device, size);
        self.size = size;
        self.last_view = None;
        self.rebind(device);
        true
    }
    pub fn render(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view: View) {
        if self.last_view == Some(view) {
            return;
        }
        self.last_view = Some(view);
        let e = view.sun_elevation_deg.to_radians();
        let a = view.sun_azimuth_deg.to_radians();
        for (i, band) in self.bands.iter().enumerate() {
            let params = FrameParams {
                size_band: [
                    self.size[0],
                    self.size[1],
                    i as u32,
                    self.bands.len() as u32,
                ],
                view: [
                    view.yaw_deg.to_radians(),
                    view.pitch_deg.to_radians(),
                    view.fov_y_deg.to_radians(),
                    self.size[0] as f32 / self.size[1] as f32,
                ],
                sun_height: [
                    e.cos() * a.sin(),
                    e.sin(),
                    e.cos() * a.cos(),
                    view.altitude_km.max(0.0),
                ],
                rgb_radius: [
                    band.weight[0],
                    band.weight[1],
                    band.weight[2],
                    self.manifest.sun_radius_rad,
                ],
                storage: [
                    u32::from(self.manifest.rgb.is_some()),
                    self.manifest
                        .rgb
                        .as_ref()
                        .and_then(|r| r.packed.as_ref())
                        .map_or(0, |p| p.block_texels as u32),
                    0,
                    0,
                ],
            };
            queue.write_buffer(&band.frame, 0, bytemuck::bytes_of(&params));
            // Separate passes establish the required ordering of the RGB sum.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spectral band to RGB"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings[i], &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        }
    }
    fn rebind(&mut self, device: &wgpu::Device) {
        let layout = self.pipeline.get_bind_group_layout(0);
        self.bindings = self
            .bands
            .iter()
            .map(|band| {
                let mut buffers = vec![
                    (0, &band.params),
                    (2, &self.angular_nodes),
                    (4, &band.tau),
                    (6, &self.dummy),
                    (9, &band.frame),
                    (11, &self.rgb_sum),
                ];
                if let Some(p) = &self.packed {
                    buffers.extend([(12, &p.data), (13, &p.blocks)]);
                } else {
                    buffers.push((5, &band.radiance));
                }
                let entries: Vec<_> = buffers
                    .into_iter()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .chain(std::iter::once(wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&self.target_view),
                    }))
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("spectral LUT display bindings"),
                    layout: &layout,
                    entries: &entries,
                })
            })
            .collect();
    }
}

fn targets(
    device: &wgpu::Device,
    size: [u32; 2],
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Buffer) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spectral LUT HDR image"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let sum = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spectral display RGB accumulator"),
        size: size[0] as u64 * size[1] as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    (texture, view, sum)
}
