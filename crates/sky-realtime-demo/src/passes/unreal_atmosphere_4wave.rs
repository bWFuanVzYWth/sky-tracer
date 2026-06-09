use glam::{Mat4, Vec3, Vec4};
use sky_unreal_atmosphere_4wave::{
    AerosolPreset, Gpu, HillaireAtmosphere, HillairePhaseMode, HillaireSettings, NonZeroRenderSize,
    RenderTargets, SUN_IRRADIANCE_REC2020_W_PER_M2, Sun, UnrealAtmosphereContext,
    UnrealFrameParams, ViewFrame,
};
use winit::dpi::PhysicalSize;

use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext,
};
use crate::passes::common::{ReferenceTexture, TexturePresentPass};
use crate::view::ViewState;

pub struct UnrealAtmosphere4WaveExperiment {
    atmosphere: UnrealAtmosphereContext,
    targets: RenderTargets,
    present: TexturePresentPass,
    reference: ReferenceTexture,
    size: NonZeroRenderSize,
    view: ViewState,
    compare_mode: CompareMode,
    sun: Sun,
    atmosphere_profile: HillaireAtmosphere,
    settings: HillaireSettings,
    aerosol: AerosolPreset,
    phase_mode: HillairePhaseMode,
    _display: DisplayTransform,
}

impl UnrealAtmosphere4WaveExperiment {
    pub fn new(context: ExperimentInit<'_>) -> Result<Self, String> {
        let size = NonZeroRenderSize::new(1, 1).expect("literal non-zero render size");
        let targets = RenderTargets::new(context.device, size);
        let gpu = Gpu::borrowed(context.device, context.queue);
        let atmosphere = UnrealAtmosphereContext::new(&gpu).map_err(|error| error.to_string())?;
        let reference = ReferenceTexture::from_asset(context.device, context.queue, context.asset);
        let present = TexturePresentPass::new(
            context.device,
            context.surface_format,
            targets.post_view(),
            &reference,
            context.display.exposure,
        );

        Ok(Self {
            atmosphere,
            targets,
            present,
            reference,
            size,
            view: ViewState::default(),
            compare_mode: CompareMode::default(),
            sun: sun_from_asset(context.asset, context.asset.manifest().sun_elevation_deg),
            atmosphere_profile: atmosphere_from_asset(context.asset),
            settings: HillaireSettings::default(),
            aerosol: AerosolPreset::default(),
            phase_mode: HillairePhaseMode::default(),
            _display: context.display,
        })
    }

    fn ensure_size(&mut self, device: &wgpu::Device, surface_size: PhysicalSize<u32>) {
        let Some(size) = NonZeroRenderSize::new(surface_size.width, surface_size.height) else {
            return;
        };
        if size == self.size {
            return;
        }

        self.targets.resize(device, size);
        self.present
            .set_source(device, self.targets.post_view(), &self.reference);
        self.size = size;
    }
}

impl RealtimeExperiment for UnrealAtmosphere4WaveExperiment {
    fn name(&self) -> &'static str {
        "unreal-4wave-sky-atmosphere"
    }

    fn update(&mut self, context: UpdateContext<'_>) {
        self.view = context.view;
        self.compare_mode = context.compare_mode;
        self.sun = sun_from_asset(context.asset, context.sun_elevation_deg);
        self.atmosphere_profile = atmosphere_from_asset(context.asset);
    }

    fn render(&mut self, context: FrameContext<'_>) {
        self.ensure_size(context.device, context.surface_size);
        let frame_params = UnrealFrameParams {
            view: view_frame_from_state(self.view, self.size, self.sun),
            atmosphere: self.atmosphere_profile,
            settings: self.settings,
            aerosol: self.aerosol,
            phase_mode: self.phase_mode,
            sun: self.sun,
        };
        self.atmosphere.prepare(
            context.device,
            context.queue,
            context.encoder,
            &frame_params,
        );
        self.atmosphere.render(context.encoder, &self.targets);
        self.present.update_uniform(
            context.queue,
            self.compare_mode,
            self.view,
            self.size.width(),
            self.size.height(),
            self.reference.is_available(),
        );
        self.present.render(context.encoder, context.target);
    }
}

fn atmosphere_from_asset(asset: &crate::assets::RealtimeAsset) -> HillaireAtmosphere {
    let mut atmosphere = HillaireAtmosphere::default();
    atmosphere.world_y0_radius_m =
        atmosphere.bottom_radius_m + asset.manifest().observer_altitude_km.max(0.0) * 1000.0;
    atmosphere
}

fn sun_from_asset(asset: &crate::assets::RealtimeAsset, elevation_deg: f32) -> Sun {
    let manifest = asset.manifest();
    let to_sun = direction_from_azimuth_elevation(manifest.sun_azimuth_deg, elevation_deg);
    Sun {
        sun_to_scene: -to_sun,
        irradiance_rec2020_w_m2: Vec3::from_array(SUN_IRRADIANCE_REC2020_W_PER_M2),
        angular_radius_rad: Sun::default().angular_radius_rad,
    }
}

fn view_frame_from_state(view: ViewState, size: NonZeroRenderSize, sun: Sun) -> ViewFrame {
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
