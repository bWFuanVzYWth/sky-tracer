use super::Resource;
use crate::{Result, renderer::View};
use bytemuck::{Pod, Zeroable};
use std::path::Path;
use wgpu::util::DeviceExt;

pub const SHADER: &str = include_str!("runtime.wgsl");
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Params {
    pub size_steps: [u32; 4],
    pub view: [f32; 4],
    pub sun: [f32; 4],
    pub dims: [u32; 4],
    pub offsets0: [u32; 4],
    pub offsets1: [u32; 4],
    pub medium: [f32; 4],
    pub solar: [f32; 4],
    pub rgb: [[f32; 4]; 4],
}
/// Per-pixel integration prototype. The local source also supports finite
/// segments; a SkyView/froxel cache can call the same shader integration core.
pub struct Renderer {
    pub resource: Resource,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    payloads: [wgpu::Buffer; 3],
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    transmittance: wgpu::Texture,
    transmittance_view: wgpu::TextureView,
    binding: wgpu::BindGroup,
    size: [u32; 2],
    last: Option<(View, u32, bool, f32, bool, bool)>,
    pub steps: u32,
    pub multiple_scattering: bool,
    /// <= 0 means the atmospheric boundary; positive values are km from camera.
    pub segment_km: f32,
    pub include_sun_disk: bool,
    /// Return spectral L in the output RGBA texture for fog composition tests.
    pub spectral_output: bool,
}
fn target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("four-wave transport output"),
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
    })
}
fn binding(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    params: &wgpu::Buffer,
    payloads: &[wgpu::Buffer; 3],
    target: &wgpu::TextureView,
    trans: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("four-wave source"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: payloads[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: payloads[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: payloads[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(target),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(trans),
            },
        ],
    })
}
impl Renderer {
    pub fn new(device: &wgpu::Device, path: &Path) -> Result<Self> {
        Self::new_with_sun_model(device, path, super::solar::SunModel::Finite16)
    }

    pub fn new_with_sun_model(
        device: &wgpu::Device,
        path: &Path,
        model: super::solar::SunModel,
    ) -> Result<Self> {
        let resource = Resource::open(path)?;
        let payloads =
            ["moments.u32", "aux.f32", "optical.u32"].map(|name| resource.read(path, name));
        let mut buffers = Vec::new();
        for bytes in payloads {
            buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("four-wave resource"),
                    contents: &bytes?,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
        }
        let payloads = buffers.try_into().map_err(|_| "invalid payloads")?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("four-wave frame"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("four-wave anisotropic transport"),
            source: wgpu::ShaderSource::Wgsl(super::solar::shader_for_model(SHADER, model).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("four-wave march"),
            layout: None,
            module: &shader,
            entry_point: Some("render"),
            compilation_options: Default::default(),
            cache: None,
        });
        let target = target(device, [1, 1]);
        let target_view = target.create_view(&Default::default());
        let transmittance = self::target(device, [1, 1]);
        let transmittance_view = transmittance.create_view(&Default::default());
        let binding = binding(
            device,
            &pipeline,
            &params,
            &payloads,
            &target_view,
            &transmittance_view,
        );
        eprintln!(
            "four-wave directional source: {:.3} MiB",
            resource.payload_bytes as f32 / 1048576.0
        );
        Ok(Self {
            resource,
            pipeline,
            params,
            payloads,
            target,
            target_view,
            transmittance,
            transmittance_view,
            binding,
            size: [1, 1],
            last: None,
            steps: 128,
            multiple_scattering: true,
            segment_km: 0.0,
            include_sun_disk: false,
            spectral_output: false,
        })
    }
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) -> bool {
        if self.size == size {
            return false;
        }
        self.size = size;
        self.last = None;
        self.target = target(device, size);
        self.target_view = self.target.create_view(&Default::default());
        self.transmittance = target(device, size);
        self.transmittance_view = self.transmittance.create_view(&Default::default());
        self.binding = binding(
            device,
            &self.pipeline,
            &self.params,
            &self.payloads,
            &self.target_view,
            &self.transmittance_view,
        );
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
    /// Four spectral transmittances. Apply them to matching spectral surface
    /// radiance before converting to RGB; a single RGB T is not universally exact.
    pub fn transmittance_view(&self) -> &wgpu::TextureView {
        &self.transmittance_view
    }
    pub fn render(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, view: View) {
        let key = (
            view,
            self.steps,
            self.multiple_scattering,
            self.segment_km,
            self.include_sun_disk,
            self.spectral_output,
        );
        if self.last == Some(key) {
            return;
        }
        self.last = Some(key);
        let r = &self.resource;
        let flags = (self.multiple_scattering as u32)
            | ((self.include_sun_disk as u32) << 1)
            | ((self.spectral_output as u32) << 2);
        let p = Params {
            size_steps: [self.size[0], self.size[1], self.steps.clamp(4, 512), flags],
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
            medium: [r.geometry.top_height(), r.albedo, self.segment_km, 0.0],
            solar: r.sun_per_nm,
            rgb: r.rgb_from_per_nm.map(|v| [v[0], v[1], v[2], 0.0]),
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("four-wave integrate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.binding, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
    }
}
