//! Hillaire 大气 runtime renderer。

use std::error::Error as StdError;
use std::fmt;

use ca_render::atmo::{SkyViewParamsGpu, SkyViewSource, Sun, SunGpu, VoxelAtmosphereLightingGpu};
use ca_render::gpu::{Gpu, RenderTargets, ViewFrame};

use crate::atmosphere::HillaireAtmosphere;
use crate::params::{
    AerosolPreset, HillaireParamsGpu, HillairePhaseMode, HillaireSettings, HillaireSpeciesGpu,
    MOLECULAR_SCATTERING_BASE, OZONE_ABSORPTION_CROSS_SECTION, SUN_SPECTRAL_IRRADIANCE,
    aerosol_preset_defaults, ozone_monthly_dobson,
};

mod bindings;
mod pipelines;
mod resources;

use bindings::{RendererBindGroups, RendererLayouts, ap_apply_bind_group};
use pipelines::RendererPipelines;
use resources::{AerialPerspectiveLut, RendererResources, RuntimeViewGpu, Texture2d, TextureArray};

const M_TO_KM: f32 = 1.0e-3;

/// 单帧 Hillaire 大气输入。
#[derive(Clone, Copy, Debug)]
pub struct HillaireFrameParams {
    pub view: ViewFrame,
    pub atmosphere: HillaireAtmosphere,
    pub settings: HillaireSettings,
    pub aerosol: AerosolPreset,
    pub phase_mode: HillairePhaseMode,
    pub sun: Sun,
}

impl HillaireFrameParams {
    #[must_use]
    pub fn new(view: ViewFrame) -> Self {
        Self {
            view,
            atmosphere: HillaireAtmosphere::default(),
            settings: HillaireSettings::default(),
            aerosol: AerosolPreset::default(),
            phase_mode: HillairePhaseMode::default(),
            sun: Sun::default(),
        }
    }
}

/// Hillaire 大气 runtime renderer。
pub struct HillaireAtmosphereContext {
    pub params_buffer: wgpu::Buffer,
    pub sky_view_params_buffer: wgpu::Buffer,
    pub view_buffer: wgpu::Buffer,
    pub voxel_lighting_buffer: wgpu::Buffer,
    pub sun_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    transmittance_lut: Texture2d,
    sky_view_lut: Texture2d,
    ap_lut: AerialPerspectiveLut,
    _aerosol_phase_lut: TextureArray,
    transmittance_pipeline: wgpu::ComputePipeline,
    sky_view_pipeline: wgpu::ComputePipeline,
    aerial_perspective_pipeline: wgpu::ComputePipeline,
    ap_apply_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    transmittance_bind_group: wgpu::BindGroup,
    sky_view_bind_group: wgpu::BindGroup,
    aerial_perspective_bind_group: wgpu::BindGroup,
    view_bind_group: wgpu::BindGroup,
    voxel_lighting_bind_group: wgpu::BindGroup,
    ap_apply_layout: wgpu::BindGroupLayout,
    ap_apply_bind_group: wgpu::BindGroup,
    sky_bind_group: wgpu::BindGroup,
    sky_view_revision: u64,
    current_sun: Sun,
}

/// `compute_params()` 返回的大气上传数据。
/// 调用方通过 `queue.write_buffer` 写入对应 buffer。
#[derive(Clone, Copy, Debug)]
pub struct AtmoUploadData {
    pub hillaire: HillaireParamsGpu,
    pub sky_view: SkyViewParamsGpu,
    pub view: RuntimeViewGpu,
    pub voxel_lighting: VoxelAtmosphereLightingGpu,
    pub sun: SunGpu,
}

impl HillaireAtmosphereContext {
    /// 创建 Hillaire 大气 renderer。
    ///
    /// # Errors
    ///
    /// 当固定 LUT 尺寸无法转换为 host-side 上传缓冲尺寸时返回错误。
    pub fn new(
        gpu: &Gpu,
        targets: &RenderTargets,
        voxel_atmosphere_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, HillaireRendererError> {
        let device = gpu.device();
        let resources = RendererResources::new(device, gpu.queue())?;
        let layouts = RendererLayouts::new(device);
        let pipelines = RendererPipelines::new(device, &layouts);
        let bind_groups = RendererBindGroups::new(
            device,
            &layouts,
            voxel_atmosphere_layout,
            &resources,
            targets,
        );
        Ok(Self {
            params_buffer: resources.params_buffer,
            sky_view_params_buffer: resources.sky_view_params_buffer,
            view_buffer: resources.view_buffer,
            voxel_lighting_buffer: resources.voxel_lighting_buffer,
            sun_buffer: resources.sun_buffer,
            sampler: resources.sampler,
            transmittance_lut: resources.transmittance_lut,
            sky_view_lut: resources.sky_view_lut,
            ap_lut: resources.ap_lut,
            _aerosol_phase_lut: resources.aerosol_phase_lut,
            transmittance_pipeline: pipelines.transmittance,
            sky_view_pipeline: pipelines.sky_view,
            aerial_perspective_pipeline: pipelines.aerial_perspective,
            ap_apply_pipeline: pipelines.ap_apply,
            sky_pipeline: pipelines.sky,
            transmittance_bind_group: bind_groups.transmittance,
            sky_view_bind_group: bind_groups.sky_view,
            aerial_perspective_bind_group: bind_groups.aerial_perspective,
            view_bind_group: bind_groups.view,
            voxel_lighting_bind_group: bind_groups.voxel_lighting,
            ap_apply_layout: layouts.ap_apply,
            ap_apply_bind_group: bind_groups.ap_apply,
            sky_bind_group: bind_groups.sky,
            sky_view_revision: 0,
            current_sun: Sun::default(),
        })
    }

    #[must_use]
    pub const fn voxel_lighting_bind_group(&self) -> &wgpu::BindGroup {
        &self.voxel_lighting_bind_group
    }

    /// 返回最近一次 `prepare` 生成的 `SkyView` source。
    #[must_use]
    pub const fn sky_view_source(&self) -> SkyViewSource<'_> {
        SkyViewSource::new(
            &self.sky_view_params_buffer,
            &self.sky_view_lut.view,
            &self.sampler,
            self.sky_view_revision,
            self.sky_view_lut.size,
            self.current_sun,
        )
    }

    pub fn retarget(&mut self, gpu: &Gpu, targets: &RenderTargets) {
        self.ap_apply_bind_group = ap_apply_bind_group(
            gpu.device(),
            &self.ap_apply_layout,
            targets,
            &self.ap_lut,
            &self.sampler,
        );
    }

    /// 计算本帧大气参数（纯 CPU 计算，无 GPU I/O）。
    #[must_use]
    pub fn compute_params(&mut self, params: &HillaireFrameParams) -> AtmoUploadData {
        let hillaire_params = hillaire_params(params);
        let sky_view_params = sky_view_params(hillaire_params);
        let view = RuntimeViewGpu::from_view(&params.view);
        let voxel_lighting = voxel_lighting_params(hillaire_params);
        let sun = SunGpu::from_sun(params.sun);
        self.current_sun = params.sun;
        self.sky_view_revision = self.sky_view_revision.wrapping_add(1);
        AtmoUploadData {
            hillaire: hillaire_params,
            sky_view: sky_view_params,
            view,
            voxel_lighting,
            sun,
        }
    }

    /// 录制大气计算 pass（纯命令录制，不涉及 GPU I/O）。
    pub fn dispatch_compute(&self, frame: &mut wgpu::CommandEncoder) {
        self.dispatch_transmittance(frame);
        self.dispatch_sky_view(frame);
        self.dispatch_aerial_perspective(frame);
    }

    /// 将 aerial perspective 和天空合成到 scene radiance 之后。
    pub fn render_after_scene(&self, frame: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        self.apply_aerial_perspective(frame, targets);
        self.render_sky(frame, targets);
    }

    fn dispatch_transmittance(&self, frame: &mut wgpu::CommandEncoder) {
        let mut pass = frame.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hillaire.transmittance.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.transmittance_pipeline);
        pass.set_bind_group(0, &self.transmittance_bind_group, &[]);
        pass.dispatch_workgroups(
            self.transmittance_lut.size.x.div_ceil(8),
            self.transmittance_lut.size.y.div_ceil(8),
            1,
        );
    }

    fn dispatch_sky_view(&self, frame: &mut wgpu::CommandEncoder) {
        let mut pass = frame.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hillaire.sky_view.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sky_view_pipeline);
        pass.set_bind_group(0, &self.sky_view_bind_group, &[]);
        pass.dispatch_workgroups(
            self.sky_view_lut.size.x.div_ceil(8),
            self.sky_view_lut.size.y.div_ceil(8),
            1,
        );
    }

    fn dispatch_aerial_perspective(&self, frame: &mut wgpu::CommandEncoder) {
        let mut pass = frame.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hillaire.aerial_perspective.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.aerial_perspective_pipeline);
        pass.set_bind_group(0, &self.aerial_perspective_bind_group, &[]);
        pass.dispatch_workgroups(
            self.ap_lut.size.x.div_ceil(8),
            self.ap_lut.size.y.div_ceil(8),
            self.ap_lut.size.z,
        );
    }

    fn apply_aerial_perspective(&self, frame: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        let mut pass = frame.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hillaire.ap_apply.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: targets.post_view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.ap_apply_pipeline);
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.ap_apply_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn render_sky(&self, frame: &mut wgpu::CommandEncoder, targets: &RenderTargets) {
        let mut pass = frame.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hillaire.sky.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: targets.post_view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.sky_pipeline);
        pass.set_bind_group(0, &self.view_bind_group, &[]);
        pass.set_bind_group(1, &self.sky_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug)]
pub enum HillaireRendererError {
    ResourceSizeOverflow,
}

impl fmt::Display for HillaireRendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceSizeOverflow => f.write_str("hillaire resource size overflow"),
        }
    }
}

impl StdError for HillaireRendererError {}

// TODO 将编译期常量（sun_spectral_irradiance, molecular_scattering_base,
// ozone_absorption_cross_section, aerosol sigma_sca/sigma_abs）拆为初始化时写入一次，
// 每帧仅更新可变字段，消除 ~120 bytes 的重复上传。
fn hillaire_params(params: &HillaireFrameParams) -> HillaireParamsGpu {
    let atmosphere = params.atmosphere;
    let view_radius_m = view_radius_from_position(params.view.world_position, atmosphere);
    let earth_radius_km = atmosphere.bottom_radius_m * M_TO_KM;
    let atmosphere_thickness_km = (atmosphere.top_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let eye_distance_to_earth_center_km = view_radius_m * M_TO_KM;
    let eye_altitude_km = (view_radius_m - atmosphere.bottom_radius_m) * M_TO_KM;
    let sun_dir = params.sun.to_sun().normalize_or_zero();
    let species_defaults = aerosol_preset_defaults(params.aerosol);

    let mut p = HillaireParamsGpu::zeroed();
    p.earth_radius_km = earth_radius_km;
    p.atmosphere_thickness_km = atmosphere_thickness_km;
    p.eye_distance_to_earth_center_km = eye_distance_to_earth_center_km;
    p.eye_altitude_km = eye_altitude_km;
    p.sun_dir = sun_dir.to_array();
    p.sky_view_height_km = eye_distance_to_earth_center_km.clamp(
        earth_radius_km + 1.0e-3,
        earth_radius_km + atmosphere_thickness_km - 1.0e-3,
    );
    p.sun_spectral_irradiance = SUN_SPECTRAL_IRRADIANCE;
    p.molecular_scattering_base = MOLECULAR_SCATTERING_BASE;
    p.ozone_absorption_cross_section = OZONE_ABSORPTION_CROSS_SECTION;

    for (slot, ((base, bg, scale), (sigma_sca, sigma_abs))) in p.species.iter_mut().zip(
        species_defaults.iter().zip(
            crate::aerosol::SIGMA_SCA
                .iter()
                .zip(crate::aerosol::SIGMA_ABS.iter()),
        ),
    ) {
        *slot = HillaireSpeciesGpu {
            sigma_sca: *sigma_sca,
            sigma_abs: *sigma_abs,
            base_density: *base,
            bg_density: *bg,
            height_scale: *scale,
            pad: 0.0,
        };
    }

    p.turbidity = params.settings.aerosol_turbidity;
    p.ozone_mean_dobson = ozone_monthly_dobson(params.settings.month);
    p.mie_phase_mode = params.phase_mode.as_gpu_u32();
    p.ground_albedo_spectral = params.settings.ground_albedo_spectral;
    p
}

const fn voxel_lighting_params(params: HillaireParamsGpu) -> VoxelAtmosphereLightingGpu {
    VoxelAtmosphereLightingGpu {
        planet_radius_km: params.earth_radius_km,
        atmosphere_thickness_km: params.atmosphere_thickness_km,
        eye_distance_to_planet_center_km: params.eye_distance_to_earth_center_km,
        pad0: 0.0,
        sun_dir: params.sun_dir,
        pad1: 0.0,
        sun_spectral_irradiance: params.sun_spectral_irradiance,
    }
}

const fn sky_view_params(params: HillaireParamsGpu) -> SkyViewParamsGpu {
    SkyViewParamsGpu::new(
        params.earth_radius_km,
        params.atmosphere_thickness_km,
        params.eye_distance_to_earth_center_km,
        params.eye_altitude_km,
        params.sun_dir,
        params.sky_view_height_km,
    )
}

fn view_radius_from_position(world_position: [f32; 4], atmosphere: HillaireAtmosphere) -> f32 {
    let radius_m =
        world_position[1].mul_add(atmosphere.scene_units_to_m, atmosphere.world_y0_radius_m);
    radius_m.clamp(
        atmosphere.bottom_radius_m + 1.0,
        atmosphere.top_radius_m - 1.0,
    )
}
