use crate::{
    controls::RealtimeControls,
    experiment::{ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext},
    passes::common::{ReferenceTexture, TexturePresentPass},
};
use sky_atmosphere_lut::{four_wave::cached::CachedRenderer as Renderer, renderer::View};
use std::path::Path;

pub struct FourWaveExperiment {
    renderer: Renderer,
    reference: ReferenceTexture,
    present: TexturePresentPass,
    controls: RealtimeControls,
    reference_pose: [f32; 3],
    reference_transport_current: bool,
}
impl FourWaveExperiment {
    pub fn new(context: ExperimentInit<'_>, path: &Path) -> Result<Self, String> {
        let mut renderer = Renderer::new(context.device, path).map_err(|e| e.to_string())?;
        renderer.include_sun_disk = true;
        let manifest = context.asset.manifest();
        let reference = ReferenceTexture::from_asset(context.device, context.queue, context.asset);
        let present = TexturePresentPass::new(
            context.device,
            context.surface_format,
            renderer.target_view(),
            &reference,
            context.display.exposure,
        );
        Ok(Self {
            renderer,
            reference,
            present,
            controls: RealtimeControls::from_asset(context.asset),
            reference_pose: [
                manifest.sun_azimuth_deg,
                manifest.sun_elevation_deg,
                manifest.observer_altitude_km,
            ],
            reference_transport_current: manifest.transport_version.as_deref()
                == Some(sky_core::asset::LAYERED_TRANSPORT_VERSION),
        })
    }
}
impl RealtimeExperiment for FourWaveExperiment {
    fn name(&self) -> &'static str {
        "four-wave-cached-anisotropic-parallel"
    }
    fn set_linear_output(&mut self, enabled: bool) {
        self.present.set_linear_output(enabled);
    }
    fn update(&mut self, context: UpdateContext<'_>) {
        self.controls = *context.controls;
    }
    fn reference_available(&self) -> bool {
        let c = self.controls;
        self.reference.is_available()
            && self.reference_transport_current
            && ((c.sun_azimuth_deg - self.reference_pose[0] + 180.0).rem_euclid(360.0) - 180.0)
                .abs()
                < 0.001
            && (c.sun_elevation_deg - self.reference_pose[1]).abs() < 0.001
            && (c.observer_altitude_km - self.reference_pose[2]).abs() < 0.00001
    }
    fn render(&mut self, context: FrameContext<'_>) {
        let size = [
            context.viewport.width.max(1),
            context.viewport.height.max(1),
        ];
        if self.renderer.resize(context.device, size) {
            self.present
                .set_source(context.device, self.renderer.target_view(), &self.reference);
        }
        let c = self.controls;
        self.renderer.render(
            context.queue,
            context.encoder,
            View {
                yaw_deg: c.view.yaw_deg,
                pitch_deg: c.view.pitch_deg,
                fov_y_deg: c.view.fov_y_deg,
                sun_azimuth_deg: c.sun_azimuth_deg,
                sun_elevation_deg: c.sun_elevation_deg,
                altitude_km: c.observer_altitude_km,
            },
        );
        self.present.update_uniform(
            context.queue,
            c.compare_mode,
            c.view,
            size[0],
            size[1],
            self.reference_available(),
            c.exposure_multiplier(),
            c.difference_scale,
            c.tone_mapping_enabled,
            c.hdr_enabled,
            c.reinhard_overexposure,
            c.hdr_paper_white_scale(),
            c.hdr_peak_scale(),
        );
        self.present
            .render(context.encoder, context.target, context.viewport);
    }
}
