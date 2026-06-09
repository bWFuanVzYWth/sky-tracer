//! Unreal SkyAtmosphere style optimized four-wavelength spectral renderer.
//!
//! This crate keeps the copied Unreal/Hillaire LUT structure but replaces the
//! baseline four channels with the cmf-mass 410/480/560/630 nm wavelengths.

pub mod aerosol;
pub mod gpu;
pub mod params;

mod atmosphere;
mod renderer;
mod sun;

pub use atmosphere::HillaireAtmosphere;
pub use gpu::{Gpu, NonZeroRenderSize, RenderTargets, ViewFrame};
pub use params::{AerosolPreset, HillairePhaseMode, HillaireSettings};
pub use renderer::{UnrealAtmosphereContext, UnrealFrameParams, UnrealRendererError};
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
    fn ground_irradiance_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/ground_irradiance.comp.wgsl")
        );
        compose_wgsl(&source, "unreal/ground_irradiance_combined.wgsl")
    }

    #[test]
    fn render_sky_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::SUN_WGSL,
            include_str!("wgsl/render_sky.wgsl")
        );
        compose_wgsl(&source, "unreal/render_sky_combined.wgsl")
    }
}
