//! Bruneton-style precomputed four-wavelength spectral atmosphere renderer.
//!
//! This crate keeps the same four spectral channels and aerosol model as the
//! optimized 4-wave renderer, but stores sky scattering in Bruneton-style 4D
//! lookup tables.

pub mod aerosol;
pub mod gpu;
pub mod params;

mod atmosphere;
mod renderer;
mod sun;

pub use atmosphere::HillaireAtmosphere;
pub use gpu::{Gpu, NonZeroRenderSize, RenderTargets, ViewFrame};
pub use params::{AerosolPreset, HillairePhaseMode, HillaireSettings};
pub use renderer::{BrunetonAtmosphereContext, BrunetonFrameParams, BrunetonRendererError};
pub use sun::{SUN_IRRADIANCE_REC2020_W_PER_M2, SUN_WGSL, Sun, SunGpu};

pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE;

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
        compose_wgsl(&source, "bruneton/transmittance_combined.wgsl")
    }

    #[test]
    fn direct_irradiance_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}",
            crate::COMMON_WGSL,
            include_str!("wgsl/direct_irradiance.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/direct_irradiance_combined.wgsl")
    }

    #[test]
    fn single_scattering_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/single_scattering.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/single_scattering_combined.wgsl")
    }

    #[test]
    fn indirect_irradiance_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}",
            crate::COMMON_WGSL,
            include_str!("wgsl/indirect_irradiance.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/indirect_irradiance_combined.wgsl")
    }

    #[test]
    fn scattering_density_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/scattering_density.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/scattering_density_combined.wgsl")
    }

    #[test]
    fn multiple_scattering_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}",
            crate::COMMON_WGSL,
            include_str!("wgsl/multiple_scattering_4d.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/multiple_scattering_combined.wgsl")
    }

    #[test]
    fn sky_view_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}",
            crate::COMMON_WGSL,
            include_str!("wgsl/sky_view.comp.wgsl")
        );
        compose_wgsl(&source, "bruneton/sky_view_combined.wgsl")
    }

    #[test]
    fn render_sky_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::SUN_WGSL,
            include_str!("wgsl/render_sky_bruneton.wgsl")
        );
        compose_wgsl(&source, "bruneton/render_sky_combined.wgsl")
    }
}
