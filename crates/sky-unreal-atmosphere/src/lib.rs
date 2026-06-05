//! Unreal SkyAtmosphere style spectral renderer.
//!
//! This crate keeps the project's current atmosphere, spectral bands and OPAC
//! aerosol phase data, but follows the Unreal/Hillaire LUT structure directly:
//! transmittance, a 2D multiple-scattering LUT, and a sky-view LUT.

mod renderer;

pub use renderer::{UnrealAtmosphereContext, UnrealFrameParams, UnrealRendererError};

pub const COMMON_WGSL: &str = include_str!("wgsl/common.wgsl");
pub const INSCATTER_WGSL: &str = include_str!("wgsl/inscatter.wgsl");

#[cfg(test)]
mod tests {
    fn compose_wgsl(source: &str, file_path: &'static str) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(source)
            .map_err(|error| format!("{file_path}: {error:?}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .map(|_| ())
            .map_err(|error| format!("{file_path}: {error:?}"))
    }

    #[test]
    fn transmittance_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}",
            crate::COMMON_WGSL,
            include_str!("wgsl/transmittance.comp.wgsl")
        );
        compose_wgsl(&source, "unreal/transmittance_combined.wgsl")
    }

    #[test]
    fn multi_scattering_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/multi_scattering.comp.wgsl")
        );
        compose_wgsl(&source, "unreal/multi_scattering_combined.wgsl")
    }

    #[test]
    fn sky_view_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/sky_view.comp.wgsl")
        );
        compose_wgsl(&source, "unreal/sky_view_combined.wgsl")
    }

    #[test]
    fn render_sky_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            sky_realtime_atmosphere::atmo::SUN_WGSL,
            include_str!("wgsl/render_sky.wgsl")
        );
        compose_wgsl(&source, "unreal/render_sky_combined.wgsl")
    }
}
