//! Bruneton/E.B. style precomputed spectral atmosphere.
//!
//! The LUT parameterization, integration structure and precompute pass order
//! follow `precomputed_atmospheric_scattering`. The atmospheric data is sourced
//! from `sky-realtime-atmosphere` and is kept as four spectral lanes.

mod renderer;

pub use renderer::{
    PrecomputedAtmosphereContext, PrecomputedFrameParams, PrecomputedRendererError,
};

pub const COMMON_WGSL: &str = include_str!("wgsl/common.wgsl");

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
    fn precompute_wgsl_composes() -> Result<(), String> {
        for (name, source) in [
            (
                "transmittance",
                include_str!("wgsl/transmittance.comp.wgsl"),
            ),
            (
                "direct_irradiance",
                include_str!("wgsl/direct_irradiance.comp.wgsl"),
            ),
            (
                "single_scattering",
                include_str!("wgsl/single_scattering.comp.wgsl"),
            ),
            (
                "scattering_density",
                include_str!("wgsl/scattering_density.comp.wgsl"),
            ),
            (
                "indirect_irradiance",
                include_str!("wgsl/indirect_irradiance.comp.wgsl"),
            ),
            (
                "multiple_scattering",
                include_str!("wgsl/multiple_scattering.comp.wgsl"),
            ),
        ] {
            compose_wgsl(&format!("{}\n\n{}", crate::COMMON_WGSL, source), name)?;
        }
        for (name, source) in [
            (
                "accumulate_2d",
                include_str!("wgsl/accumulate_2d.comp.wgsl"),
            ),
            (
                "accumulate_3d",
                include_str!("wgsl/accumulate_3d.comp.wgsl"),
            ),
        ] {
            compose_wgsl(source, name)?;
        }
        Ok(())
    }

    #[test]
    fn render_wgsl_composes() -> Result<(), String> {
        compose_wgsl(
            &format!(
                "{}\n\n{}\n\n{}",
                crate::COMMON_WGSL,
                sky_realtime_atmosphere::atmo::SUN_WGSL,
                include_str!("wgsl/render_sky.wgsl")
            ),
            "render_sky",
        )
    }
}
