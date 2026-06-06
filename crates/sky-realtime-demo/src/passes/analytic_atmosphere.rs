use glam::{Mat4, Vec3, Vec4};
use sky_analytic_atmosphere::{
    AnalyticAtmosphere, AnalyticAtmosphereContext, AnalyticFrameParams, AnalyticSun, AnalyticView,
    SCENE_RADIANCE_FORMAT,
};
use winit::dpi::PhysicalSize;

use crate::assets::RealtimeAsset;
use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext,
};
use crate::passes::common::{ReferenceTexture, TexturePresentPass};
use crate::view::ViewState;

pub struct AnalyticAtmosphereExperiment {
    atmosphere: AnalyticAtmosphereContext,
    target: SceneTarget,
    present: TexturePresentPass,
    reference: ReferenceTexture,
    view: ViewState,
    compare_mode: CompareMode,
    sun: AnalyticSun,
    atmosphere_profile: AnalyticAtmosphere,
    _display: DisplayTransform,
}

impl AnalyticAtmosphereExperiment {
    pub fn new(context: ExperimentInit<'_>) -> Result<Self, String> {
        let size = PhysicalSize::new(1, 1);
        let target = SceneTarget::new(context.device, size);
        let atmosphere = AnalyticAtmosphereContext::new(context.device);
        let reference = ReferenceTexture::from_asset(context.device, context.queue, context.asset);
        let present = TexturePresentPass::new(
            context.device,
            context.surface_format,
            &target.view,
            &reference,
            context.display.exposure,
        );

        Ok(Self {
            atmosphere,
            target,
            present,
            reference,
            view: ViewState::default(),
            compare_mode: CompareMode::default(),
            sun: analytic_sun_from_asset(context.asset, context.asset.manifest().sun_elevation_deg),
            atmosphere_profile: analytic_atmosphere_from_asset(context.asset),
            _display: context.display,
        })
    }

    fn ensure_size(&mut self, device: &wgpu::Device, surface_size: PhysicalSize<u32>) {
        let size = PhysicalSize::new(surface_size.width.max(1), surface_size.height.max(1));
        if size == self.target.size {
            return;
        }
        self.target = SceneTarget::new(device, size);
        self.present
            .set_source(device, &self.target.view, &self.reference);
    }
}

impl RealtimeExperiment for AnalyticAtmosphereExperiment {
    fn name(&self) -> &'static str {
        "analytic-radial-quadratic-atmosphere"
    }

    fn update(&mut self, context: UpdateContext<'_>) {
        self.view = context.view;
        self.compare_mode = context.compare_mode;
        self.sun = analytic_sun_from_asset(context.asset, context.sun_elevation_deg);
        self.atmosphere_profile = analytic_atmosphere_from_asset(context.asset);
    }

    fn render(&mut self, context: FrameContext<'_>) {
        self.ensure_size(context.device, context.surface_size);
        let frame_params = AnalyticFrameParams {
            view: analytic_view_from_state(self.view, self.target.size),
            atmosphere: self.atmosphere_profile,
            sun: self.sun,
        };
        self.atmosphere.prepare(context.queue, &frame_params);
        self.atmosphere.render(context.encoder, &self.target.view);
        self.present.update_uniform(
            context.queue,
            self.compare_mode,
            self.view,
            self.target.size.width,
            self.target.size.height,
            self.reference.is_available(),
        );
        self.present.render(context.encoder, context.target);
    }
}

struct SceneTarget {
    size: PhysicalSize<u32>,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl SceneTarget {
    fn new(device: &wgpu::Device, size: PhysicalSize<u32>) -> Self {
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("analytic.scene_radiance"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_RADIANCE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("analytic.scene_radiance.view"),
            ..Default::default()
        });
        Self {
            size,
            _texture: texture,
            view,
        }
    }
}

fn analytic_atmosphere_from_asset(asset: &RealtimeAsset) -> AnalyticAtmosphere {
    let mut atmosphere = AnalyticAtmosphere::default();
    atmosphere.world_y0_radius_m =
        atmosphere.bottom_radius_m + asset.manifest().observer_altitude_km.max(0.0) * 1000.0;
    atmosphere
}

fn analytic_sun_from_asset(asset: &RealtimeAsset, elevation_deg: f32) -> AnalyticSun {
    let manifest = asset.manifest();
    let to_sun = direction_from_azimuth_elevation(manifest.sun_azimuth_deg, elevation_deg);
    AnalyticSun {
        sun_to_scene: -to_sun,
        angular_radius_rad: AnalyticSun::default().angular_radius_rad,
    }
}

fn analytic_view_from_state(view: ViewState, size: PhysicalSize<u32>) -> AnalyticView {
    let width = size.width.max(1);
    let height = size.height.max(1);
    let aspect = width as f32 / height as f32;
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

    AnalyticView {
        relative_world_from_clip: relative_world_from_clip.to_cols_array_2d(),
        world_position: [0.0, 0.0, 0.0, 1.0],
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
