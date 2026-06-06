//! Aggressively simplified analytic atmosphere renderer.
//!
//! This crate intentionally does not depend on the higher-quality UE-style
//! renderer. It keeps its physical and spectral constants in the WGSL header
//! so the shader can be moved as a mostly self-contained single-file model.

mod renderer;

pub use renderer::{
    AnalyticAtmosphereContext, AnalyticFrameParams, AnalyticSun, AnalyticView,
    SCENE_RADIANCE_FORMAT,
};

pub const ANALYTIC_SKY_WGSL: &str = include_str!("wgsl/analytic_sky.wgsl");

#[cfg(test)]
mod tests {
    #[test]
    fn analytic_sky_wgsl_composes() -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(crate::ANALYTIC_SKY_WGSL)
            .map_err(|error| format!("analytic_sky.wgsl: {error:?}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .map(|_| ())
            .map_err(|error| format!("analytic_sky.wgsl: {error:?}"))
    }
}
