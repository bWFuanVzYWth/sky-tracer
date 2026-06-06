use sky_unreal_atmosphere::{
    AerosolPreset, Gpu, HillaireAtmosphere, HillairePhaseMode, HillaireSettings, NonZeroRenderSize,
    RenderTargets, Sun, UnrealAtmosphereContext, UnrealFrameParams,
};
use winit::dpi::PhysicalSize;

use crate::color::DisplayTransform;
use crate::experiment::{
    CompareMode, ExperimentInit, FrameContext, RealtimeExperiment, UpdateContext,
};
use crate::passes::common::{
    ReferenceTexture, TexturePresentPass, atmosphere_from_asset, sun_from_asset,
    view_frame_from_state,
};
use crate::view::ViewState;

pub struct UnrealAtmosphereExperiment {
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

impl UnrealAtmosphereExperiment {
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
            sun: sun_from_asset(context.asset),
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

impl RealtimeExperiment for UnrealAtmosphereExperiment {
    fn name(&self) -> &'static str {
        "unreal-spectral-sky-atmosphere"
    }

    fn update(&mut self, context: UpdateContext<'_>) {
        self.view = context.view;
        self.compare_mode = context.compare_mode;
        self.sun = sun_from_asset(context.asset);
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
            self.size,
            self.reference.is_available(),
        );
        self.present.render(context.encoder, context.target);
    }
}
