use ca_render::gpu::RenderTargets;

use super::resources::{AerialPerspectiveLut, HILLAIRE_LUT_FORMAT, RendererResources};

pub(super) struct RendererLayouts {
    pub transmittance: wgpu::BindGroupLayout,
    pub sky_view: wgpu::BindGroupLayout,
    pub aerial_perspective: wgpu::BindGroupLayout,
    pub view: wgpu::BindGroupLayout,
    pub ap_apply: wgpu::BindGroupLayout,
    pub sky: wgpu::BindGroupLayout,
}

impl RendererLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            transmittance: transmittance_layout(device),
            sky_view: sky_view_layout(device),
            aerial_perspective: aerial_perspective_layout(device),
            view: view_layout(device),
            ap_apply: ap_apply_layout(device),
            sky: sky_layout(device),
        }
    }
}

pub(super) struct RendererBindGroups {
    pub transmittance: wgpu::BindGroup,
    pub sky_view: wgpu::BindGroup,
    pub aerial_perspective: wgpu::BindGroup,
    pub view: wgpu::BindGroup,
    pub voxel_lighting: wgpu::BindGroup,
    pub ap_apply: wgpu::BindGroup,
    pub sky: wgpu::BindGroup,
}

impl RendererBindGroups {
    pub fn new(
        device: &wgpu::Device,
        layouts: &RendererLayouts,
        voxel_atmosphere_layout: &wgpu::BindGroupLayout,
        resources: &RendererResources,
        targets: &RenderTargets,
    ) -> Self {
        let transmittance = transmittance_bind_group(
            device,
            &layouts.transmittance,
            &resources.params_buffer,
            &resources.transmittance_lut.view,
        );
        let sky_view = sky_view_bind_group(
            device,
            &layouts.sky_view,
            &resources.params_buffer,
            &resources.transmittance_lut.view,
            &resources.sampler,
            &resources.sky_view_lut.view,
            &resources.aerosol_phase_lut.view,
        );
        let aerial_perspective = aerial_perspective_bind_group(
            device,
            &layouts.aerial_perspective,
            &AerialPerspectiveBindGroupInput {
                params: &resources.params_buffer,
                transmittance: &resources.transmittance_lut.view,
                sampler: &resources.sampler,
                view: &resources.view_buffer,
                ap_lut: &resources.ap_lut,
                phase_lut: &resources.aerosol_phase_lut.view,
            },
        );
        let view = view_bind_group(device, &layouts.view, &resources.view_buffer);
        let voxel_lighting = voxel_lighting_bind_group(
            device,
            voxel_atmosphere_layout,
            &resources.voxel_lighting_buffer,
            &resources.transmittance_lut.view,
            &resources.ap_lut,
            &resources.sampler,
        );
        let ap_apply = ap_apply_bind_group(
            device,
            &layouts.ap_apply,
            targets,
            &resources.ap_lut,
            &resources.sampler,
        );
        let sky = sky_bind_group(
            device,
            &layouts.sky,
            &SkyBindGroupInput {
                sky_params: &resources.sky_view_params_buffer,
                transmittance: &resources.transmittance_lut.view,
                sampler: &resources.sampler,
                sky_view: &resources.sky_view_lut.view,
                sun: &resources.sun_buffer,
                atmosphere: &resources.voxel_lighting_buffer,
            },
        );
        Self {
            transmittance,
            sky_view,
            aerial_perspective,
            view,
            voxel_lighting,
            ap_apply,
            sky,
        }
    }
}

pub(super) fn ap_apply_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    targets: &RenderTargets,
    ap_lut: &AerialPerspectiveLut,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.ap_apply.bg"),
        layout,
        entries: &[
            texture_binding(0, targets.scene_view()),
            texture_binding(1, targets.depth_view()),
            texture_binding(2, &ap_lut.inscatter_view),
            texture_binding(3, &ap_lut.transmittance_view),
            sampler_binding(4, sampler),
        ],
    })
}

fn transmittance_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.transmittance.bgl"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::COMPUTE),
            storage_2d_entry(1),
        ],
    })
}

fn sky_view_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.sky_view.bgl"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::COMPUTE),
            texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
            sampler_entry(2, wgpu::ShaderStages::COMPUTE),
            storage_2d_entry(3),
            texture_2d_array_entry(4, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn aerial_perspective_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.aerial_perspective.bgl"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::COMPUTE),
            texture_2d_entry(1, wgpu::ShaderStages::COMPUTE),
            sampler_entry(2, wgpu::ShaderStages::COMPUTE),
            uniform_entry(3, wgpu::ShaderStages::COMPUTE),
            storage_3d_entry(4),
            storage_3d_entry(5),
            texture_2d_array_entry(6, wgpu::ShaderStages::COMPUTE),
        ],
    })
}

fn view_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.view.bgl"),
        entries: &[uniform_entry(
            0,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        )],
    })
}

fn ap_apply_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.ap_apply.bgl"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            texture_3d_entry(2, wgpu::ShaderStages::FRAGMENT),
            texture_3d_entry(3, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(4, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

fn sky_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hillaire.sky.bgl"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
            texture_2d_entry(1, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(2, wgpu::ShaderStages::FRAGMENT),
            texture_2d_entry(3, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(4, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(5, wgpu::ShaderStages::FRAGMENT),
        ],
    })
}

const fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn texture_2d_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

const fn texture_2d_array_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

const fn texture_3d_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

const fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

const fn storage_2d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HILLAIRE_LUT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

const fn storage_3d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HILLAIRE_LUT_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn transmittance_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.transmittance.bg"),
        layout,
        entries: &[buffer_binding(0, params), texture_binding(1, transmittance)],
    })
}

fn sky_view_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    sky_view: &wgpu::TextureView,
    phase_lut: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.sky_view.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, transmittance),
            sampler_binding(2, sampler),
            texture_binding(3, sky_view),
            texture_binding(4, phase_lut),
        ],
    })
}

struct AerialPerspectiveBindGroupInput<'a> {
    params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    view: &'a wgpu::Buffer,
    ap_lut: &'a AerialPerspectiveLut,
    phase_lut: &'a wgpu::TextureView,
}

fn aerial_perspective_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: &AerialPerspectiveBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.aerial_perspective.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            buffer_binding(3, input.view),
            texture_binding(4, &input.ap_lut.inscatter_view),
            texture_binding(5, &input.ap_lut.transmittance_view),
            texture_binding(6, input.phase_lut),
        ],
    })
}

fn view_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.view.bg"),
        layout,
        entries: &[buffer_binding(0, view)],
    })
}

fn voxel_lighting_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    transmittance: &wgpu::TextureView,
    ap_lut: &AerialPerspectiveLut,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.voxel_lighting.bg"),
        layout,
        entries: &[
            buffer_binding(0, params),
            texture_binding(1, transmittance),
            sampler_binding(2, sampler),
            texture_binding(3, &ap_lut.inscatter_view),
            texture_binding(4, &ap_lut.transmittance_view),
        ],
    })
}

struct SkyBindGroupInput<'a> {
    sky_params: &'a wgpu::Buffer,
    transmittance: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    sky_view: &'a wgpu::TextureView,
    sun: &'a wgpu::Buffer,
    atmosphere: &'a wgpu::Buffer,
}

fn sky_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: &SkyBindGroupInput<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hillaire.sky.bg"),
        layout,
        entries: &[
            buffer_binding(0, input.sky_params),
            texture_binding(1, input.transmittance),
            sampler_binding(2, input.sampler),
            texture_binding(3, input.sky_view),
            buffer_binding(4, input.sun),
            buffer_binding(5, input.atmosphere),
        ],
    })
}

fn buffer_binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

const fn sampler_binding(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}
