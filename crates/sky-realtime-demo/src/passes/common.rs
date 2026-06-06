use glam::{Mat4, Vec3, Vec4};
use sky_unreal_atmosphere::{
    HillaireAtmosphere, NonZeroRenderSize, SUN_IRRADIANCE_REC2020_W_PER_M2, Sun, ViewFrame,
};

use crate::assets::RealtimeAsset;
use crate::experiment::CompareMode;
use crate::view::ViewState;

const PRESENT_SHADER: &str = include_str!("../shaders/present_texture.wgsl");

pub(crate) struct TexturePresentPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    exposure: f32,
    reference_projection_sun_observer: [f32; 4],
}

impl TexturePresentPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        source: &wgpu::TextureView,
        reference: &ReferenceTexture,
        exposure: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("realtime_present_shader"),
            source: wgpu::ShaderSource::Wgsl(PRESENT_SHADER.into()),
        });
        let reference_projection_sun_observer = reference.projection_sun_observer();
        let uniform = PresentUniform {
            exposure_mode_ref_diff: [exposure.max(0.0), 0.0, reference.has_reference(), 4.0],
            view_yaw_pitch_fov_aspect: [
                ViewState::default().yaw_deg,
                ViewState::default().pitch_deg,
                ViewState::default().fov_y_deg,
                1.0,
            ],
            reference_projection_sun_observer,
        };
        let uniform_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("realtime_present_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("realtime_present_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let bind_group = present_bind_group(device, &layout, source, &uniform_buffer, reference);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("realtime_present_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("realtime_present_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            bind_group,
            uniform_buffer,
            exposure,
            reference_projection_sun_observer,
        }
    }

    pub(crate) fn set_source(
        &mut self,
        device: &wgpu::Device,
        source: &wgpu::TextureView,
        reference: &ReferenceTexture,
    ) {
        self.bind_group = present_bind_group(
            device,
            &self.layout,
            source,
            &self.uniform_buffer,
            reference,
        );
    }

    pub(crate) fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        compare_mode: CompareMode,
        view: ViewState,
        width: u32,
        height: u32,
        has_reference: bool,
    ) {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let uniform = PresentUniform {
            exposure_mode_ref_diff: [
                self.exposure,
                compare_mode.shader_id(),
                if has_reference { 1.0 } else { 0.0 },
                4.0,
            ],
            view_yaw_pitch_fov_aspect: [view.yaw_deg, view.pitch_deg, view.fov_y_deg, aspect],
            reference_projection_sun_observer: self.reference_projection_sun_observer,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub(crate) fn render(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("realtime_present_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn present_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    reference: &ReferenceTexture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("realtime_present_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&reference.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&reference.sampler),
            },
        ],
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PresentUniform {
    exposure_mode_ref_diff: [f32; 4],
    view_yaw_pitch_fov_aspect: [f32; 4],
    reference_projection_sun_observer: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<PresentUniform>() == 48);

pub(crate) struct ReferenceTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    available: bool,
    projection_sun_observer: [f32; 4],
}

impl ReferenceTexture {
    pub(crate) fn from_asset(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        asset: &RealtimeAsset,
    ) -> Self {
        let manifest = asset.manifest();
        let projection_sun_observer = [
            asset.reference_projection_id(),
            manifest.sun_elevation_deg,
            manifest.sun_azimuth_deg,
            manifest.observer_altitude_km,
        ];
        let path = asset.rgb_exr_path();
        match load_reference_rgba32f(&path) {
            Ok((width, height, rgba)) => {
                println!(
                    "loaded linear offline reference {}: {}x{}",
                    path.display(),
                    width,
                    height
                );
                Self::from_rgba32f(
                    device,
                    queue,
                    width,
                    height,
                    &rgba,
                    true,
                    projection_sun_observer,
                )
            }
            Err(error) => {
                eprintln!(
                    "warning: failed to load linear offline reference {}: {error}",
                    path.display()
                );
                Self::from_rgba32f(
                    device,
                    queue,
                    1,
                    1,
                    &[0.0, 0.0, 0.0, 1.0],
                    false,
                    projection_sun_observer,
                )
            }
        }
    }

    fn from_rgba32f(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[f32],
        available: bool,
        projection_sun_observer: [f32; 4],
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offline_reference_rgb_exr"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width.max(1) * 16),
                rows_per_image: Some(height.max(1)),
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("offline_reference_rgb_exr.view"),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("offline_reference_rgb_exr.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            _texture: texture,
            view,
            sampler,
            available,
            projection_sun_observer,
        }
    }

    pub(crate) const fn is_available(&self) -> bool {
        self.available
    }

    fn has_reference(&self) -> f32 {
        if self.available { 1.0 } else { 0.0 }
    }

    const fn projection_sun_observer(&self) -> [f32; 4] {
        self.projection_sun_observer
    }
}

#[derive(Debug)]
struct LinearRgbaImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

fn load_reference_rgba32f(path: &std::path::Path) -> Result<(u32, u32, Vec<f32>), String> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            let width = resolution.width();
            let height = resolution.height();
            LinearRgbaImage {
                width,
                height,
                pixels: vec![0.0; width * height * 4],
            }
        },
        |image, position, (r, g, b, a): (f32, f32, f32, f32)| {
            let index = (position.y() * image.width + position.x()) * 4;
            image.pixels[index] = r;
            image.pixels[index + 1] = g;
            image.pixels[index + 2] = b;
            image.pixels[index + 3] = a;
        },
    )
    .map_err(|error| error.to_string())?;
    let pixels = image.layer_data.channel_data.pixels;
    Ok((pixels.width as u32, pixels.height as u32, pixels.pixels))
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn atmosphere_from_asset(asset: &RealtimeAsset) -> HillaireAtmosphere {
    let mut atmosphere = HillaireAtmosphere::default();
    atmosphere.world_y0_radius_m =
        atmosphere.bottom_radius_m + asset.manifest().observer_altitude_km.max(0.0) * 1000.0;
    atmosphere
}

pub(crate) fn sun_from_asset(asset: &RealtimeAsset, elevation_deg: f32) -> Sun {
    let manifest = asset.manifest();
    let to_sun = direction_from_azimuth_elevation(manifest.sun_azimuth_deg, elevation_deg);
    Sun {
        sun_to_scene: -to_sun,
        irradiance_rec2020_w_m2: Vec3::from_array(SUN_IRRADIANCE_REC2020_W_PER_M2),
        angular_radius_rad: Sun::default().angular_radius_rad,
    }
}

pub(crate) fn view_frame_from_state(
    view: ViewState,
    size: NonZeroRenderSize,
    sun: Sun,
) -> ViewFrame {
    let aspect = size.width() as f32 / size.height() as f32;
    let yaw = view.yaw_deg.to_radians();
    let pitch = view.pitch_deg.to_radians();
    let fov_tan = (0.5 * view.fov_y_deg.to_radians()).tan();
    let forward = Vec3::new(
        yaw.sin() * pitch.cos(),
        pitch.sin(),
        yaw.cos() * pitch.cos(),
    )
    .normalize();
    let right = Vec3::new(yaw.cos(), 0.0, -yaw.sin()).normalize();
    let up = forward.cross(right).normalize();
    let relative_world_from_clip = Mat4::from_cols(
        (right * aspect * fov_tan).extend(0.0),
        (up * fov_tan).extend(0.0),
        Vec4::ZERO,
        forward.extend(1.0),
    );

    ViewFrame {
        clip_from_world: Mat4::IDENTITY.to_cols_array_2d(),
        world_from_clip: Mat4::IDENTITY.to_cols_array_2d(),
        clip_from_relative_world: Mat4::IDENTITY.to_cols_array_2d(),
        relative_world_from_clip: relative_world_from_clip.to_cols_array_2d(),
        world_position: [0.0, 0.0, 0.0, 1.0],
        world_forward: forward.extend(0.0).to_array(),
        world_right: right.extend(0.0).to_array(),
        world_up: up.extend(0.0).to_array(),
        view_params: [fov_tan, aspect, 0.1, 0.0],
        light_dir: sun.to_sun().extend(0.0).to_array(),
        viewport: [
            size.width() as f32,
            size.height() as f32,
            1.0 / size.width() as f32,
            1.0 / size.height() as f32,
        ],
    }
}

fn direction_from_azimuth_elevation(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    Vec3::new(
        azimuth.sin() * elevation.cos(),
        elevation.sin(),
        azimuth.cos() * elevation.cos(),
    )
    .normalize()
}

#[cfg(test)]
mod tests {
    use sky_unreal_atmosphere::NonZeroRenderSize;

    use super::{Sun, Vec3, ViewState, view_frame_from_state};

    #[test]
    fn default_view_frame_points_forward_at_center() {
        let size = NonZeroRenderSize::new(16, 9).expect("size");
        let frame = view_frame_from_state(ViewState::default(), size, Sun::default());
        let center_ray = Vec3::from_array([
            frame.relative_world_from_clip[3][0],
            frame.relative_world_from_clip[3][1],
            frame.relative_world_from_clip[3][2],
        ])
        .normalize();
        assert!(center_ray.z > 0.9);
    }

    #[test]
    fn present_shader_is_valid_wgsl() {
        let module =
            naga::front::wgsl::parse_str(super::PRESENT_SHADER).expect("parse present wgsl");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator.validate(&module).expect("validate present wgsl");
    }
}
