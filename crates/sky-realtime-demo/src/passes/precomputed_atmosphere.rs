use sky_precomputed_atmosphere::{PrecomputedAtmosphereContext, PrecomputedFrameParams};
use sky_realtime_atmosphere::gpu::{Gpu, NonZeroRenderSize, RenderTargets};
use sky_realtime_atmosphere::{AerosolPreset, HillaireAtmosphere, HillaireSettings};
use winit::dpi::PhysicalSize;

use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext,
};
use crate::view::ViewState;

use super::hillaire_atmosphere::{
    ReferenceTexture, TexturePresentPass, atmosphere_from_asset, sun_from_asset,
    view_frame_from_state,
};

pub struct PrecomputedAtmosphereExperiment {
    atmosphere: PrecomputedAtmosphereContext,
    targets: RenderTargets,
    present: TexturePresentPass,
    reference: ReferenceTexture,
    size: NonZeroRenderSize,
    view: ViewState,
    compare_mode: CompareMode,
    sun: sky_realtime_atmosphere::atmo::Sun,
    hillaire_atmosphere: HillaireAtmosphere,
    settings: HillaireSettings,
    aerosol: AerosolPreset,
    _display: DisplayTransform,
}

impl PrecomputedAtmosphereExperiment {
    pub fn new(context: ExperimentInit<'_>) -> Result<Self, String> {
        let size = NonZeroRenderSize::new(1, 1).expect("literal non-zero render size");
        let targets = RenderTargets::new(context.device, size);
        let gpu = Gpu::borrowed(context.device, context.queue);
        let atmosphere =
            PrecomputedAtmosphereContext::new(&gpu).map_err(|error| error.to_string())?;
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
            sun: sun_from_asset(context.asset),
            hillaire_atmosphere: atmosphere_from_asset(context.asset),
            settings: HillaireSettings::default(),
            aerosol: AerosolPreset::default(),
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

    fn frame_params(&self) -> PrecomputedFrameParams {
        PrecomputedFrameParams {
            view: view_frame_from_state(self.view, self.size, self.sun),
            atmosphere: self.hillaire_atmosphere,
            settings: self.settings,
            aerosol: self.aerosol,
            sun: self.sun,
        }
    }
}

impl RealtimeExperiment for PrecomputedAtmosphereExperiment {
    fn name(&self) -> &'static str {
        "eb-precomputed-spectral-atmosphere"
    }

    fn update(&mut self, context: UpdateContext<'_>) {
        self.view = context.view;
        self.compare_mode = context.compare_mode;
        self.sun = sun_from_asset(context.asset);
        self.hillaire_atmosphere = atmosphere_from_asset(context.asset);
    }

    fn render(&mut self, context: FrameContext<'_>) {
        self.ensure_size(context.device, context.surface_size);
        let frame_params = self.frame_params();
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
            self.size,
            self.reference.is_available(),
        );
        self.present.render(context.encoder, context.target);
    }
}
