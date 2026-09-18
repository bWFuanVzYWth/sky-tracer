use crate::{
    controls::RealtimeControls,
    experiment::{ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext},
    passes::common::{ReferenceTexture, TexturePresentPass},
};
use sky_realtime::{Config, Renderer, View, Wavelengths};

pub struct HybridExperiment {
    renderer: Renderer,
    reference: ReferenceTexture,
    present: TexturePresentPass,
    controls: RealtimeControls,
    reference_pose: [f32; 3],
    reference_transport_current: bool,
    wavelengths: Wavelengths,
    applied_medium: [f32; 2],
}
impl HybridExperiment {
    pub fn new(context: ExperimentInit<'_>) -> Result<Self, String> {
        let model = sky_realtime::model::Model::earth().map_err(|e| e.to_string())?;
        let wavelengths = Wavelengths::optimized_four();
        let mut renderer = Renderer::new(
            context.device,
            &model,
            &wavelengths,
            0.18,
            Config::balanced(),
        )
        .map_err(|e| e.to_string())?;
        let report = renderer
            .rebuild(context.device, context.queue)
            .map_err(|e| e.to_string())?;
        println!(
            "hybrid fresh 4D solve: {:.3} s; {:.3} MiB resident, {:.1} MiB temporary peak payload",
            report.wall_seconds,
            report.resident_bytes as f32 / 1048576.0,
            report.peak_payload_bytes as f32 / 1048576.0
        );
        renderer.include_sun_disk = true;
        let reference = ReferenceTexture::from_asset(context.device, context.queue, context.asset);
        let present = TexturePresentPass::new(
            context.device,
            context.surface_format,
            renderer.target_view(),
            &reference,
            context.display.exposure,
        );
        let manifest = context.asset.manifest();
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
                == Some(sky_assets::asset::LAYERED_TRANSPORT_VERSION),
            wavelengths,
            applied_medium: [1.0, 0.18],
        })
    }
}
impl RealtimeExperiment for HybridExperiment {
    fn name(&self) -> &'static str {
        "hybrid-4d-four-wave-parallel"
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
            && (c.aerosol_turbidity - 1.0).abs() < 1e-6
            && (c.ground_albedo_spectral[0] - 0.18).abs() < 1e-6
            && ((c.sun_azimuth_deg - self.reference_pose[0] + 180.0).rem_euclid(360.0) - 180.0)
                .abs()
                < 0.001
            && (c.sun_elevation_deg - self.reference_pose[1]).abs() < 0.001
            && (c.observer_altitude_km - self.reference_pose[2]).abs() < 0.00001
    }
    fn render(&mut self, context: FrameContext<'_>) {
        let c = self.controls;
        let requested = [c.aerosol_turbidity, c.ground_albedo_spectral[0]];
        if requested != self.applied_medium {
            let result = sky_realtime::model::Model::earth_with_aerosol_scale(requested[0])
                .and_then(|model| {
                    self.renderer.set_medium(
                        context.device,
                        context.queue,
                        &model,
                        &self.wavelengths,
                        requested[1],
                    )
                });
            match result {
                Ok(Some(report)) => println!(
                    "hybrid medium update: {:.3} s, key {}",
                    report.wall_seconds, report.medium_key
                ),
                Ok(None) => {}
                Err(e) => eprintln!("hybrid medium update failed: {e}"),
            }
            self.applied_medium = requested;
        }
        let size = [
            context.viewport.width.max(1),
            context.viewport.height.max(1),
        ];
        if self.renderer.resize(context.device, size) {
            self.present
                .set_source(context.device, self.renderer.target_view(), &self.reference);
        }
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
