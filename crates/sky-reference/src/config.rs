use serde::{Deserialize, Serialize};

use crate::Result;

/// Reference budget after mapping and quadrature convergence studies.
pub const MAX_ASSET_BYTES: u64 = 6_000_000_000;
pub const METADATA_RESERVE: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateMapping {
    #[default]
    Legacy,
    SunAligned,
    /// Keep the tuned horizon grid and append a zenith-focused angular cap.
    SunAlignedAngular,
    /// Separate the fully visible cones from the horizon-crossing cones.
    HorizonAligned,
    /// Reference chart with ray-frame height transfer and physical twilight corners.
    RayAlignedReference,
}

impl CoordinateMapping {
    pub fn is_reference(self) -> bool {
        matches!(self, Self::HorizonAligned | Self::RayAlignedReference)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolarInterpolation {
    #[default]
    Linear,
    LogRadiance,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseInterpolation {
    #[default]
    LinearRadiance,
    LogRadiance,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewInterpolation {
    #[default]
    Linear,
    MonotoneCubic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceMapping {
    #[default]
    Radiance,
    LocalLinearHeight,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeightInterpolation {
    #[default]
    LinearHeight,
    ReferenceCdf,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IterationScheme {
    #[default]
    Orders,
    FixedPoint,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AngularIntegration {
    #[default]
    ViewOnly,
    SunAndView,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConeMapping {
    #[default]
    AngleSquare,
    OpticalDistance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BakeConfig {
    /// Missing in v1 resources, which retain their original lookup convention.
    #[serde(default)]
    pub mapping: CoordinateMapping,
    #[serde(default)]
    pub solar_interpolation: SolarInterpolation,
    #[serde(default)]
    pub phase_interpolation: PhaseInterpolation,
    #[serde(default)]
    pub view_interpolation: ViewInterpolation,
    #[serde(default)]
    pub source_mapping: SourceMapping,
    #[serde(default)]
    pub iteration_scheme: IterationScheme,
    #[serde(default)]
    pub angular_integration: AngularIntegration,
    #[serde(default)]
    pub cone_mapping: ConeMapping,
    /// Per-asset byte budget. Raise deliberately after measuring mapping/integration bias.
    #[serde(default = "default_asset_budget")]
    pub max_asset_bytes: u64,
    /// [radius, view zenith, solar zenith, view/sun cosine]. Nu varies fastest.
    pub scattering: [usize; 4],
    /// Optional scattering-only height nodes; optical depth retains its own mapping.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scattering_altitudes_km: Vec<f32>,
    #[serde(default)]
    pub height_interpolation: HeightInterpolation,
    /// [radius, view zenith], with separate ground and space halves.
    pub optical_depth: [usize; 2],
    pub ground_sun_samples: usize,
    pub ray_steps: usize,
    pub optical_depth_steps: usize,
    pub angular_mu: usize,
    pub angular_phi: usize,
    pub sun_mu: usize,
    pub sun_phi: usize,
    /// Counts both volume collisions and diffuse ground reflections.
    pub max_orders: usize,
    /// 0 disables early exit. This is an increment diagnostic, not an error bound.
    pub relative_order_tolerance: f32,
    pub min_orders: usize,
    pub ground_albedo: f32,
    /// Bounds each dispatch to keep desktop GPUs responsive. Multiple of 64.
    pub dispatch_texels: usize,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            mapping: CoordinateMapping::SunAligned,
            solar_interpolation: SolarInterpolation::LogRadiance,
            phase_interpolation: PhaseInterpolation::LinearRadiance,
            view_interpolation: ViewInterpolation::Linear,
            source_mapping: SourceMapping::Radiance,
            iteration_scheme: IterationScheme::Orders,
            angular_integration: AngularIntegration::SunAndView,
            cone_mapping: ConeMapping::OpticalDistance,
            max_asset_bytes: MAX_ASSET_BYTES,
            scattering: [64, 32, 65, 256],
            scattering_altitudes_km: Vec::new(),
            height_interpolation: HeightInterpolation::LinearHeight,
            optical_depth: [128, 1024],
            ground_sun_samples: 512,
            ray_steps: 256,
            optical_depth_steps: 2048,
            angular_mu: 16,
            angular_phi: 32,
            sun_mu: 4,
            sun_phi: 16,
            max_orders: 16,
            relative_order_tolerance: 0.0,
            min_orders: 4,
            ground_albedo: crate::physics::atmosphere::GROUND_ALBEDO,
            dispatch_texels: 65536,
        }
    }
}

impl BakeConfig {
    /// Frozen full-spectral teacher configuration; its remaining bias is documented.
    pub fn current_reference() -> Self {
        serde_json::from_str(include_str!("../configs/reference.json")).expect("reference config")
    }

    pub fn validate_top_height(&self, top: f32) -> Result<()> {
        if self
            .scattering_altitudes_km
            .last()
            .is_some_and(|h| *h != top)
        {
            return Err("custom scattering altitudes must end at the atmosphere top".into());
        }
        Ok(())
    }
    pub fn reference() -> Self {
        Self {
            mapping: CoordinateMapping::HorizonAligned,
            scattering: [80, 32, 193, 257],
            optical_depth: [384, 4096],
            ground_sun_samples: 1024,
            max_asset_bytes: 22_000_000_000,
            ..Self::default()
        }
    }
    /// Reference preset with the original twilight nodes and a denser zenith cap.
    /// The larger spectral resource is a bake intermediate; RGB export is smaller.
    pub fn reference_v4() -> Self {
        Self {
            mapping: CoordinateMapping::SunAlignedAngular,
            scattering: [64, 32, 97, 256],
            ground_sun_samples: 1024,
            max_asset_bytes: 9_000_000_000,
            ..Self::default()
        }
    }

    pub fn development() -> Self {
        Self {
            scattering: [32, 24, 65, 256],
            optical_depth: [64, 512],
            ground_sun_samples: 256,
            ray_steps: 128,
            optical_depth_steps: 512,
            angular_mu: 16,
            angular_phi: 32,
            sun_mu: 2,
            sun_phi: 8,
            max_orders: 8,
            ..Self::default()
        }
    }
    pub fn smoke() -> Self {
        Self {
            scattering: [4, 12, 6, 8],
            optical_depth: [12, 48],
            ground_sun_samples: 16,
            ray_steps: 24,
            optical_depth_steps: 64,
            angular_mu: 4,
            angular_phi: 8,
            sun_mu: 2,
            sun_phi: 4,
            max_orders: 3,
            min_orders: 2,
            ..Self::default()
        }
    }

    pub fn scattering_len(&self) -> usize {
        self.scattering.iter().product()
    }

    pub fn optical_depth_len(&self) -> usize {
        self.optical_depth.iter().product()
    }

    pub fn band_bytes(&self) -> Result<u64> {
        let product = |dims: &[usize]| -> Result<u64> {
            dims.iter().try_fold(1u64, |a, &b| {
                a.checked_mul(b as u64)
                    .ok_or_else(|| "LUT dimensions overflow".into())
            })
        };
        product(&self.scattering)?
            .checked_add(product(&self.optical_depth)?)
            .and_then(|n| n.checked_add(self.ground_sun_samples as u64))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(8))
            .ok_or_else(|| "LUT byte size overflow".into())
    }

    pub fn asset_bytes(&self, band_count: usize) -> Result<u64> {
        self.band_bytes()?
            .checked_mul(band_count as u64)
            .and_then(|n| n.checked_add(METADATA_RESERVE))
            .ok_or_else(|| "asset byte size overflow".into())
    }

    pub fn validate(&self, band_count: usize) -> Result<()> {
        if self.source_mapping != SourceMapping::Radiance
            && self.mapping != CoordinateMapping::RayAlignedReference
        {
            return Err("local source interpolation requires ray-aligned reference storage".into());
        }
        if !self.scattering_altitudes_km.is_empty()
            && (self.mapping != CoordinateMapping::RayAlignedReference
                || self.scattering_altitudes_km.len() != self.scattering[0]
                || self.scattering_altitudes_km[0] != 0.0
                || self
                    .scattering_altitudes_km
                    .iter()
                    .any(|h| !h.is_finite() || *h < 0.0)
                || self
                    .scattering_altitudes_km
                    .windows(2)
                    .any(|h| h[1] <= h[0]))
        {
            return Err("custom scattering altitudes require a strictly increasing ray-reference axis from zero".into());
        }
        if self.iteration_scheme == IterationScheme::FixedPoint
            && self.mapping != CoordinateMapping::RayAlignedReference
        {
            return Err("fixed-point iteration requires ray-aligned reference mapping".into());
        }
        if self.view_interpolation != ViewInterpolation::Linear
            && self.mapping != CoordinateMapping::RayAlignedReference
        {
            return Err(
                "cubic view interpolation requires the ray-aligned reference mapping".into(),
            );
        }
        if self.phase_interpolation != PhaseInterpolation::LinearRadiance
            && self.mapping != CoordinateMapping::RayAlignedReference
        {
            return Err("log phase interpolation requires ray-aligned reference mapping".into());
        }
        if self.dispatch_texels == 0
            || !self.dispatch_texels.is_multiple_of(64)
            || self.dispatch_texels > 65535 * 64
        {
            return Err("dispatch_texels must be a positive multiple of 64, <= 4194240".into());
        }
        if band_count == 0
            || self
                .scattering
                .iter()
                .chain(self.optical_depth.iter())
                .any(|&n| n < 2)
            || self.scattering[1] < 4
            || self.optical_depth[1] < 4
            || !self.scattering[1].is_multiple_of(2)
            || !self.optical_depth[1].is_multiple_of(2)
            || self.ground_sun_samples < 2
        {
            return Err(
                "nonempty bands and axes >= 2 required; view axes must be even and >= 4".into(),
            );
        }
        if [
            self.ray_steps,
            self.optical_depth_steps,
            self.angular_mu,
            self.angular_phi,
            self.sun_mu,
            self.sun_phi,
            self.max_orders,
            self.min_orders,
        ]
        .contains(&0)
            || self.min_orders > self.max_orders
        {
            return Err("integration counts must be positive and min_orders <= max_orders".into());
        }
        if !self.ground_albedo.is_finite()
            || !(0.0..1.0).contains(&self.ground_albedo)
            || !self.relative_order_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_order_tolerance)
        {
            return Err("albedo and order tolerance must be finite in [0, 1)".into());
        }
        for pair in [
            [self.angular_mu, self.angular_phi],
            [self.sun_mu, self.sun_phi],
        ] {
            let n = pair[0]
                .checked_mul(pair[1])
                .ok_or("quadrature count overflow")?;
            if n > 1_048_576 {
                return Err("quadrature exceeds GPU work budget".into());
            }
        }
        if [self.ray_steps, self.optical_depth_steps, self.max_orders]
            .iter()
            .any(|&x| x > 1_048_576)
        {
            return Err("integration counts exceed GPU work budget".into());
        }
        if self.mapping == CoordinateMapping::SunAlignedAngular && self.scattering[2] < 97 {
            return Err("zenith-cap mapping needs at least 97 solar nodes".into());
        }
        if self.mapping.is_reference() && !(self.scattering[3] - 1).is_multiple_of(8) {
            return Err(
                "horizon chart needs 8k+1 phase nodes to share exact region boundaries".into(),
            );
        }
        if self.mapping.is_reference() && self.cone_mapping != ConeMapping::OpticalDistance {
            return Err("horizon chart requires optical-distance cone coordinates".into());
        }
        let size = self.asset_bytes(band_count)?;
        if self.scattering_len() > u32::MAX as usize
            || self.optical_depth_len() > u32::MAX as usize
            || self.ground_sun_samples > u32::MAX as usize
            || band_count > u32::MAX as usize
        {
            return Err("LUT dimensions exceed WGSL u32 indexing".into());
        }
        if size > self.max_asset_bytes {
            return Err(format!("asset needs {size} bytes; configured budget is {} bytes; refine the mapping or adjust dimensions/budget",self.max_asset_bytes).into());
        }
        Ok(())
    }
}

fn default_asset_budget() -> u64 {
    MAX_ASSET_BYTES
}
