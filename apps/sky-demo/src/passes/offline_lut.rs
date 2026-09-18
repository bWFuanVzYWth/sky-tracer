use crate::{
    controls::RealtimeControls,
    experiment::{ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext},
    passes::common::{ReferenceTexture, TexturePresentPass},
};
use sky_realtime::physics::spectrum::{SpectralBand, SpectralRgbConverter, WhiteBalanceMatrix};
use sky_reference::{
    asset::Manifest,
    renderer::{SpectralRenderer, View},
};
use std::path::Path;

pub struct OfflineLutExperiment {
    renderer: SpectralRenderer,
    reference: ReferenceTexture,
    present: TexturePresentPass,
    controls: RealtimeControls,
    reference_pose: [f32; 3],
    reference_transport_current: bool,
}

impl OfflineLutExperiment {
    pub fn new(context: ExperimentInit<'_>, path: &Path) -> Result<Self, String> {
        let m = Manifest::open(path).map_err(|e| e.to_string())?;
        let reference = context.asset.manifest();
        if m.bands.len() != reference.band_centers_nm.len()
            || m.bands
                .iter()
                .zip(&reference.band_centers_nm)
                .any(|(a, b)| (a.center_nm - *b).abs() > 0.01)
        {
            return Err(
                "The LUT and path-traced reference must use the same spectral bands".into(),
            );
        }
        let bands: Vec<_> = m
            .bands
            .iter()
            .enumerate()
            .map(|(i, b)| SpectralBand {
                index: i,
                center_nm: b.center_nm,
                lower_nm: b.lower_nm,
                upper_nm: b.upper_nm,
                solar_irradiance_w_m2: b.solar_irradiance_w_m2,
                ozone_cross_section_cm2: 0.0,
            })
            .collect();
        let converter = if let Some(color) = &reference.colorimetry {
            SpectralRgbConverter::new_with_white_balance(
                &bands,
                WhiteBalanceMatrix {
                    method: "reference asset white balance",
                    source_white_xyz_y1: color.white_balance.source_white_xyz_y1,
                    target_white_xyz_y1: color.white_balance.target_white_xyz_y1,
                    xyz_from_xyz: color.white_balance.xyz_from_xyz,
                },
            )
        } else {
            SpectralRgbConverter::new_solar_d65(&bands)
        };
        let weights: Vec<_> = (0..bands.len())
            .map(|i| {
                let mut basis = vec![0.0; bands.len()];
                basis[i] = 1.0;
                let rgb = converter.to_linear_srgb(&basis);
                // The shared demo presentation pass expects linear Rec.2020.
                [
                    0.627_404 * rgb.r + 0.329_282 * rgb.g + 0.0433136 * rgb.b,
                    0.0690970 * rgb.r + 0.919_54 * rgb.g + 0.0113612 * rgb.b,
                    0.0163916 * rgb.r + 0.0880132 * rgb.g + 0.8955952 * rgb.b,
                ]
            })
            .collect();
        let renderer =
            SpectralRenderer::new(context.device, path, &weights).map_err(|e| e.to_string())?;
        let texture = ReferenceTexture::from_asset(context.device, context.queue, context.asset);
        let present = TexturePresentPass::new(
            context.device,
            context.surface_format,
            renderer.target_view(),
            &texture,
            context.display.exposure,
        );
        Ok(Self {
            renderer,
            reference: texture,
            present,
            controls: RealtimeControls::from_asset(context.asset),
            reference_pose: [
                reference.sun_azimuth_deg,
                reference.sun_elevation_deg,
                reference.observer_altitude_km,
            ],
            reference_transport_current: reference.transport_version.as_deref()
                == Some(sky_assets::asset::LAYERED_TRANSPORT_VERSION),
        })
    }
}

impl RealtimeExperiment for OfflineLutExperiment {
    fn set_linear_output(&mut self, enabled: bool) {
        self.present.set_linear_output(enabled);
    }
    fn name(&self) -> &'static str {
        if self
            .renderer
            .manifest
            .rgb
            .as_ref()
            .is_some_and(|r| r.packed.is_some())
        {
            "offline-phase-inclusive-rec2020-packed-lut"
        } else if self.renderer.manifest.rgb.is_some() {
            "offline-phase-inclusive-rec2020-lut"
        } else {
            "offline-phase-inclusive-spectral-lut"
        }
    }
    fn update(&mut self, context: UpdateContext<'_>) {
        self.controls = *context.controls;
    }
    fn reference_available(&self) -> bool {
        let c = self.controls;
        let azimuth_error =
            ((c.sun_azimuth_deg - self.reference_pose[0] + 180.0).rem_euclid(360.0) - 180.0).abs();
        self.reference.is_available()
            && self.reference_transport_current
            && azimuth_error < 0.001
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
