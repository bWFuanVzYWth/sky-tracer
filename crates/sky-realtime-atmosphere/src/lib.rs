//! Hillaire 2020 风格的光谱大气渲染后端。
//!
//! 本 crate 在大气内部保持 4 波长 spectral 表示，只在天空、太阳和体素直接光
//! 的采样边界投影到项目统一的 scene-linear Rec.2020 D65 工作空间。

extern crate self as ca_render;

pub mod aerosol;
pub mod atmo;
pub mod gpu;
pub mod params;

mod atmosphere;
mod renderer;
mod sun;

pub use atmosphere::HillaireAtmosphere;
pub use params::{AerosolPreset, HillairePhaseMode, HillaireSettings};
pub use renderer::{HillaireAtmosphereContext, HillaireFrameParams, HillaireRendererError};
pub use sun::{SUN_IRRADIANCE_REC2020_W_PER_M2, Sun, SunGpu};

pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::FLOAT32_FILTERABLE;

pub const COMMON_WGSL: &str = include_str!("wgsl/common.wgsl");
pub const INSCATTER_WGSL: &str = include_str!("wgsl/inscatter.wgsl");
pub const PT_RUNTIME_WGSL: &str = include_str!("wgsl/pt_runtime.wgsl");

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
        compose_wgsl(&source, "ca_atmosphere/transmittance_combined.wgsl")
    }

    #[test]
    fn aerial_perspective_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/aerial_perspective.comp.wgsl")
        );
        compose_wgsl(&source, "ca_atmosphere/aerial_perspective_combined.wgsl")
    }

    #[test]
    fn sky_view_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            include_str!("wgsl/sky_view.comp.wgsl")
        );
        compose_wgsl(&source, "ca_atmosphere/sky_view_combined.wgsl")
    }

    #[test]
    fn raster_sky_wgsl_composes() -> Result<(), String> {
        let source = format!(
            "{}\n\n{}\n\n{}\n\n{}",
            ca_render::atmo::SKY_VIEW_WGSL,
            ca_render::atmo::SUN_WGSL,
            ca_render::atmo::VOXEL_ATMOSPHERE_LIGHTING_WGSL,
            include_str!("wgsl/raster_sky_hillaire.wgsl")
        );
        compose_wgsl(&source, "ca_atmosphere/raster_sky_combined.wgsl")
    }

    #[test]
    fn aerial_perspective_apply_wgsl_composes() -> Result<(), String> {
        compose_wgsl(
            include_str!("wgsl/aerial_perspective_apply.wgsl"),
            "ca_atmosphere/aerial_perspective_apply.wgsl",
        )
    }

    #[test]
    fn pt_runtime_wgsl_composes() -> Result<(), String> {
        const PT_RUNTIME_TEST_BINDINGS: &str = r"
struct PtCamera {
    origin: vec3<f32>,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var aerosol_phase_lut: texture_2d_array<f32>;
@group(0) @binding(4) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(5) var ap_inscatter_lut: texture_3d<f32>;
@group(0) @binding(6) var ap_transmittance_lut: texture_3d<f32>;
@group(1) @binding(0) var<uniform> camera: PtCamera;
";
        let source = format!(
            "{}\n\n{}\n\n{}\n\n{}",
            crate::COMMON_WGSL,
            crate::INSCATTER_WGSL,
            PT_RUNTIME_TEST_BINDINGS,
            crate::PT_RUNTIME_WGSL
        );
        compose_wgsl(&source, "ca_atmosphere/pt_runtime_combined.wgsl")
    }
}
