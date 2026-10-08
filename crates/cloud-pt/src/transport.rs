//! CPU reference for the *represented* trilinear density field.
//!
//! Extinction events use delta tracking; sun visibility uses ratio tracking.
//! There is no scattering-order or low-transmittance cutoff. Diagnostic budgets
//! reject a sample rather than turning unfinished paths into black contributions.

pub use crate::sampling::Pcg32;
use crate::sampling::{henyey_greenstein, sample_cosine_hemisphere, sample_henyey_greenstein};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::{error::Error, f64::consts::PI, fmt};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Ray {
    pub origin: DVec3,
    /// Unit propagation direction. Public entry points normalize valid inputs.
    pub direction: DVec3,
}

impl Ray {
    pub fn at(self, distance: f64) -> DVec3 {
        self.origin + self.direction * distance
    }

    pub fn normalized(self) -> Result<Self, TraceError> {
        if !self.origin.is_finite()
            || !self.direction.is_finite()
            || self.direction.length_squared() == 0.0
        {
            return Err(TraceError::InvalidRay);
        }
        let direction = self.direction.normalize();
        if !direction.is_finite() || direction.length_squared() == 0.0 {
            return Err(TraceError::InvalidRay);
        }
        Ok(Self {
            origin: self.origin,
            direction,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Bounds {
    pub min: DVec3,
    pub max: DVec3,
}

impl Bounds {
    pub fn valid(self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && (self.max - self.min).is_finite()
            && self.max.cmpgt(self.min).all()
    }

    /// Forward slab intersection. Distances are physical only for unit rays.
    pub fn ray_interval(self, ray: Ray) -> Option<(f64, f64)> {
        let mut enter: f64 = 0.0;
        let mut exit = f64::INFINITY;
        for axis in 0..3 {
            let origin = ray.origin[axis];
            let direction = ray.direction[axis];
            if direction == 0.0 {
                if origin < self.min[axis] || origin > self.max[axis] {
                    return None;
                }
            } else {
                let a = (self.min[axis] - origin) / direction;
                let b = (self.max[axis] - origin) / direction;
                enter = enter.max(a.min(b));
                exit = exit.min(a.max(b));
                if exit <= enter {
                    return None;
                }
            }
        }
        (exit > enter).then_some((enter, exit))
    }
}

/// A finite density field with a conservative maximum over its whole domain.
/// `density_world` must be nonnegative, finite, and zero outside `bounds`.
pub trait DensityField {
    fn bounds(&self) -> Bounds;
    fn density_world(&self, point: DVec3) -> f64;
    fn max_density(&self) -> f64;
    /// Local conservative bounds over a complete, ordered ray partition.
    /// Generic fields retain the global proposal; sparse grids override this.
    fn majorant_spans(
        &self,
        ray: Ray,
        endpoint: f64,
    ) -> Result<
        Box<dyn Iterator<Item = Result<crate::majorant::MajorantSpan, TraceError>> + '_>,
        TraceError,
    > {
        Ok(global_spans(
            self.bounds(),
            self.max_density(),
            ray,
            endpoint,
        ))
    }
}

fn global_spans(
    bounds: Bounds,
    maximum: f64,
    ray: Ray,
    endpoint: f64,
) -> Box<dyn Iterator<Item = Result<crate::majorant::MajorantSpan, TraceError>>> {
    let span = bounds.ray_interval(ray).and_then(|(start, end)| {
        let end = end.min(endpoint);
        (end > start).then_some(Ok(crate::majorant::MajorantSpan {
            start,
            end,
            max_density: maximum,
        }))
    });
    Box::new(span.into_iter())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GroundPlane {
    pub height: f64,
    pub albedo: DVec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransportSettings {
    /// Unbiased half-survival roulette for already-dark sun shadow estimates.
    #[serde(default = "default_shadow_roulette")]
    pub shadow_roulette: bool,
    /// Spatial proposals change efficiency, preserving the same density field.
    #[serde(default = "default_spatial_majorants")]
    pub spatial_majorants: bool,
    /// World-space extinction coefficient = density * extinction_scale.
    pub extinction_scale: f64,
    pub scattering_albedo: DVec3,
    pub phase_g: f64,
    /// Unit vector pointing from a scene point toward the directional sun.
    pub sun_direction: DVec3,
    /// Irradiance on a plane perpendicular to the sun direction, linear RGB.
    pub sun_irradiance: DVec3,
    /// Constant environment radiance, linear RGB.
    pub sky_radiance: DVec3,
    pub ground: Option<GroundPlane>,
    /// Begin unbiased roulette after this many actual scattering/surface events.
    pub roulette_start: u32,
    /// Diagnostic limit on tracking candidates + actual surface events.
    /// Exceeding it is an error, never a finite-depth approximation.
    pub event_limit: u64,
}

impl Default for TransportSettings {
    fn default() -> Self {
        Self {
            shadow_roulette: true,
            spatial_majorants: true,
            extinction_scale: 4.0,
            scattering_albedo: DVec3::ONE,
            phase_g: 0.877,
            sun_direction: DVec3::new(0.5826, 0.766, 0.2717).normalize(),
            sun_irradiance: DVec3::new(2.6, 2.5, 2.3),
            sky_radiance: DVec3::new(0.03, 0.07, 0.23),
            ground: Some(GroundPlane {
                height: -1000.0,
                albedo: DVec3::splat(0.2),
            }),
            roulette_start: 4,
            event_limit: 1_000_000,
        }
    }
}

fn default_spatial_majorants() -> bool {
    true
}

fn default_shadow_roulette() -> bool {
    true
}

impl TransportSettings {
    pub fn validate(&self) -> Result<(), TraceError> {
        if !self.extinction_scale.is_finite() || self.extinction_scale < 0.0 {
            return Err(TraceError::InvalidSettings(
                "extinction_scale must be finite and nonnegative",
            ));
        }
        if !valid_albedo(self.scattering_albedo) {
            return Err(TraceError::InvalidSettings(
                "scattering_albedo must be in [0, 1]",
            ));
        }
        if !self.phase_g.is_finite() || self.phase_g.abs() >= 1.0 {
            return Err(TraceError::InvalidSettings(
                "phase_g must be strictly between -1 and 1",
            ));
        }
        if !self.sun_direction.is_finite() || self.sun_direction.length_squared() == 0.0 {
            return Err(TraceError::InvalidSettings(
                "sun_direction must be finite and nonzero",
            ));
        }
        if !valid_nonnegative(self.sun_irradiance) || !valid_nonnegative(self.sky_radiance) {
            return Err(TraceError::InvalidSettings(
                "lighting must be finite and nonnegative",
            ));
        }
        if let Some(ground) = self.ground {
            if !ground.height.is_finite() || !valid_albedo(ground.albedo) {
                return Err(TraceError::InvalidSettings(
                    "ground height/albedo is invalid",
                ));
            }
        }
        if self.event_limit == 0 {
            return Err(TraceError::InvalidSettings("event_limit must be positive"));
        }
        Ok(())
    }
}

fn valid_nonnegative(value: DVec3) -> bool {
    value.is_finite() && value.min_element() >= 0.0
}

fn valid_albedo(value: DVec3) -> bool {
    valid_nonnegative(value) && value.max_element() <= 1.0
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub radiance: DVec3,
    pub collisions: u64,
    pub null_collisions: u64,
    pub shadow_events: u64,
    pub surface_events: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceError {
    InvalidSettings(&'static str),
    InvalidRay,
    InvalidVolume(&'static str),
    InvalidDensity { density: f64, maximum: f64 },
    EventLimit { limit: u64 },
    NumericalFailure(&'static str),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(message) => write!(f, "invalid transport settings: {message}"),
            Self::InvalidRay => write!(f, "ray is not finite with a nonzero direction"),
            Self::InvalidVolume(message) => write!(f, "invalid density field: {message}"),
            Self::InvalidDensity { density, maximum } => write!(
                f,
                "density {density} is outside conservative majorant [0, {maximum}]"
            ),
            Self::EventLimit { limit } => write!(
                f,
                "path exceeded diagnostic event limit {limit}; sample is invalid, increase the budget"
            ),
            Self::NumericalFailure(message) => write!(f, "transport numerical failure: {message}"),
        }
    }
}

impl Error for TraceError {}

#[derive(Default)]
struct Budget {
    used: u64,
}

impl Budget {
    fn charge(&mut self, limit: u64) -> Result<(), TraceError> {
        if self.used >= limit {
            return Err(TraceError::EventLimit { limit });
        }
        self.used += 1;
        Ok(())
    }
}

fn validate_field(field: &impl DensityField, scale: f64) -> Result<f64, TraceError> {
    if !field.bounds().valid() {
        return Err(TraceError::InvalidVolume(
            "bounds must be finite and nonempty",
        ));
    }
    let maximum = field.max_density();
    if !maximum.is_finite() || maximum < 0.0 {
        return Err(TraceError::InvalidVolume(
            "maximum density must be finite and nonnegative",
        ));
    }
    let majorant = maximum * scale;
    if !majorant.is_finite() {
        return Err(TraceError::InvalidVolume("extinction majorant overflow"));
    }
    if maximum > 0.0 && scale > 0.0 && majorant == 0.0 {
        return Err(TraceError::NumericalFailure(
            "extinction majorant underflow",
        ));
    }
    Ok(majorant)
}

fn density_ratio(field: &impl DensityField, point: DVec3, maximum: f64) -> Result<f64, TraceError> {
    let density = field.density_world(point);
    if !density.is_finite() || density < 0.0 || density > maximum {
        return Err(TraceError::InvalidDensity { density, maximum });
    }
    Ok(density / maximum)
}

fn next_distance(distance: f64, majorant: f64, rng: &mut Pcg32) -> Result<f64, TraceError> {
    let next = distance - rng.open01().ln() / majorant;
    if next <= distance || next.is_nan() {
        return Err(TraceError::NumericalFailure(
            "tracking distance failed to advance",
        ));
    }
    Ok(next)
}

fn ground_distance(ray: Ray, ground: Option<GroundPlane>) -> Result<Option<f64>, TraceError> {
    let Some(ground) = ground else {
        return Ok(None);
    };
    if ray.direction.y == 0.0 {
        return Ok(None);
    }
    let distance = (ground.height - ray.origin.y) / ray.direction.y;
    if !distance.is_finite() {
        return Err(TraceError::NumericalFailure(
            "nonfinite ground intersection distance",
        ));
    }
    Ok((distance > 0.0 || (distance == 0.0 && ray.direction.y < 0.0)).then_some(distance))
}

fn span_rate(span: crate::majorant::MajorantSpan, scale: f64) -> Result<f64, TraceError> {
    if !span.start.is_finite()
        || !span.end.is_finite()
        || span.start < 0.0
        || span.end <= span.start
        || !span.max_density.is_finite()
        || span.max_density < 0.0
    {
        return Err(TraceError::InvalidVolume("invalid majorant ray partition"));
    }
    let rate = span.max_density * scale;
    if !rate.is_finite() {
        return Err(TraceError::InvalidVolume(
            "local extinction majorant overflow",
        ));
    }
    if span.max_density > 0.0 && scale > 0.0 && rate == 0.0 {
        return Err(TraceError::NumericalFailure(
            "local extinction majorant underflow",
        ));
    }
    Ok(rate)
}

fn track_collision(
    field: &impl DensityField,
    ray: Ray,
    endpoint: f64,
    majorant: f64,
    settings: &TransportSettings,
    rng: &mut Pcg32,
    budget: &mut Budget,
    sample: &mut Sample,
) -> Result<Option<DVec3>, TraceError> {
    if majorant == 0.0 {
        return Ok(None);
    }
    let spans = if settings.spatial_majorants {
        field.majorant_spans(ray, endpoint)?
    } else {
        global_spans(field.bounds(), field.max_density(), ray, endpoint)
    };
    for span in spans {
        let span = span?;
        budget.charge(settings.event_limit)?;
        let rate = span_rate(span, settings.extinction_scale)?;
        if rate == 0.0 {
            continue;
        }
        let start = ray.at(span.start);
        let length = span.end - span.start;
        let mut distance = 0.0;
        loop {
            distance = next_distance(distance, rate, rng)?;
            if distance >= length {
                break;
            }
            budget.charge(settings.event_limit)?;
            let point = start + ray.direction * distance;
            if !point.is_finite() {
                return Err(TraceError::NumericalFailure("nonfinite tracking point"));
            }
            let probability = density_ratio(field, point, span.max_density)?;
            if rng.open01() < probability {
                sample.collisions += 1;
                return Ok(Some(point));
            }
            sample.null_collisions += 1;
        }
    }
    Ok(None)
}

fn track_transmittance(
    field: &impl DensityField,
    ray: Ray,
    endpoint: f64,
    majorant: f64,
    settings: &TransportSettings,
    rng: &mut Pcg32,
    budget: &mut Budget,
    sample: &mut Sample,
) -> Result<f64, TraceError> {
    if ground_distance(ray, settings.ground)?.is_some_and(|distance| distance < endpoint) {
        return Ok(0.0);
    }
    if majorant == 0.0 {
        return Ok(1.0);
    }
    let mut transmittance = 1.0;
    // Each shadow ray owns one cadence, continuing across majorant boundaries.
    let mut roulette_candidates = 0;
    let spans = if settings.spatial_majorants {
        field.majorant_spans(ray, endpoint)?
    } else {
        global_spans(field.bounds(), field.max_density(), ray, endpoint)
    };
    for span in spans {
        let span = span?;
        budget.charge(settings.event_limit)?;
        let rate = span_rate(span, settings.extinction_scale)?;
        if rate == 0.0 {
            continue;
        }
        let start = ray.at(span.start);
        let length = span.end - span.start;
        let mut distance = 0.0;
        loop {
            distance = next_distance(distance, rate, rng)?;
            if distance >= length {
                break;
            }
            budget.charge(settings.event_limit)?;
            sample.shadow_events += 1;
            roulette_candidates += 1;
            let point = start + ray.direction * distance;
            if !point.is_finite() {
                return Err(TraceError::NumericalFailure(
                    "nonfinite shadow tracking point",
                ));
            }
            transmittance *= 1.0 - density_ratio(field, point, span.max_density)?;
            if transmittance == 0.0 {
                return Ok(0.0);
            }
            if roulette_candidates == 16 {
                roulette_candidates = 0;
                if settings.shadow_roulette && transmittance < 1e-4 {
                    transmittance = shadow_roulette_weight(transmittance, rng.open01());
                    if transmittance == 0.0 {
                        return Ok(0.0);
                    }
                }
            }
        }
    }
    Ok(transmittance)
}

fn shadow_roulette_weight(transmittance: f64, uniform: f64) -> f64 {
    // Conditional expectation is (0 + 2 * Tr) / 2 = Tr. The fixed survival
    // probability preserves fractional ratio estimates rather than replacing
    // them with a binary visibility estimator.
    if uniform < 0.5 {
        0.0
    } else {
        2.0 * transmittance
    }
}

/// Unbiased ratio-tracking estimate of segment visibility, including the ground.
/// `endpoint` may be +infinity. Unit world distance is used after ray normalization.
pub fn transmittance(
    field: &impl DensityField,
    ray: Ray,
    endpoint: f64,
    settings: &TransportSettings,
    rng: &mut Pcg32,
) -> Result<f64, TraceError> {
    settings.validate()?;
    if endpoint.is_nan() || endpoint < 0.0 {
        return Err(TraceError::InvalidSettings(
            "transmittance endpoint must be nonnegative",
        ));
    }
    let majorant = validate_field(field, settings.extinction_scale)?;
    track_transmittance(
        field,
        ray.normalized()?,
        endpoint,
        majorant,
        settings,
        rng,
        &mut Budget::default(),
        &mut Sample::default(),
    )
}

/// One complete path sample. The caller must reject the whole render on error;
/// dropping failed paths or dividing by only successful samples introduces bias.
pub fn trace_sample(
    field: &impl DensityField,
    ray: Ray,
    settings: &TransportSettings,
    rng: &mut Pcg32,
) -> Result<Sample, TraceError> {
    settings.validate()?;
    let majorant = validate_field(field, settings.extinction_scale)?;
    let mut ray = ray.normalized()?;
    let sun_direction = settings.sun_direction.normalize();
    if !sun_direction.is_finite() || sun_direction.length_squared() == 0.0 {
        return Err(TraceError::InvalidSettings(
            "sun_direction normalization failed",
        ));
    }
    let mut sample = Sample::default();
    let mut budget = Budget::default();
    let mut throughput = DVec3::ONE;
    let mut vertices: u64 = 0;
    loop {
        let surface = ground_distance(ray, settings.ground)?;
        let collision = track_collision(
            field,
            ray,
            surface.unwrap_or(f64::INFINITY),
            majorant,
            settings,
            rng,
            &mut budget,
            &mut sample,
        )?;
        if let Some(point) = collision {
            throughput *= settings.scattering_albedo;
            if throughput.max_element() == 0.0 {
                break;
            }
            if settings.sun_irradiance.max_element() > 0.0 {
                let shadow = Ray {
                    origin: point,
                    direction: sun_direction,
                };
                let visibility = track_transmittance(
                    field,
                    shadow,
                    f64::INFINITY,
                    majorant,
                    settings,
                    rng,
                    &mut budget,
                    &mut sample,
                )?;
                let phase = henyey_greenstein(ray.direction.dot(sun_direction), settings.phase_g);
                sample.radiance += throughput * settings.sun_irradiance * (phase * visibility);
            }
            ray = Ray {
                origin: point,
                direction: sample_henyey_greenstein(ray.direction, settings.phase_g, rng),
            };
        } else if let Some(distance) = surface {
            budget.charge(settings.event_limit)?;
            sample.surface_events += 1;
            // The opaque plane has a diffuse upper side and an absorbing lower side.
            if ray.direction.y >= 0.0 {
                break;
            }
            let ground = settings.ground.expect("surface distance requires ground");
            let mut point = ray.at(distance);
            point.y = ground.height; // Exact plane origin, with no epsilon offset.
            if !point.is_finite() {
                return Err(TraceError::NumericalFailure(
                    "nonfinite ground intersection point",
                ));
            }
            throughput *= ground.albedo;
            if throughput.max_element() == 0.0 {
                break;
            }
            if sun_direction.y > 0.0 && settings.sun_irradiance.max_element() > 0.0 {
                let visibility = track_transmittance(
                    field,
                    Ray {
                        origin: point,
                        direction: sun_direction,
                    },
                    f64::INFINITY,
                    majorant,
                    settings,
                    rng,
                    &mut budget,
                    &mut sample,
                )?;
                sample.radiance +=
                    throughput * settings.sun_irradiance * (sun_direction.y * visibility / PI);
            }
            ray = Ray {
                origin: point,
                direction: sample_cosine_hemisphere(DVec3::Y, rng),
            };
        } else {
            sample.radiance += throughput * settings.sky_radiance;
            break;
        }
        vertices += 1;
        if !sample.radiance.is_finite()
            || !throughput.is_finite()
            || !ray.origin.is_finite()
            || !ray.direction.is_finite()
        {
            return Err(TraceError::NumericalFailure(
                "nonfinite path weight/radiance/ray",
            ));
        }
        if vertices >= u64::from(settings.roulette_start) {
            let survival = throughput.max_element().min(1.0);
            // Survival may equal one: conservative albedo-one clouds are allowed.
            if rng.open01() >= survival {
                break;
            }
            throughput /= survival;
        }
    }
    if !sample.radiance.is_finite() {
        return Err(TraceError::NumericalFailure("nonfinite final radiance"));
    }
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantField {
        density: f64,
        maximum: f64,
    }
    impl DensityField for ConstantField {
        fn bounds(&self) -> Bounds {
            Bounds {
                min: DVec3::ZERO,
                max: DVec3::ONE,
            }
        }
        fn density_world(&self, _: DVec3) -> f64 {
            self.density
        }
        fn max_density(&self) -> f64 {
            self.maximum
        }
    }

    fn dark_settings() -> TransportSettings {
        TransportSettings {
            extinction_scale: 1.0,
            sun_irradiance: DVec3::ZERO,
            ground: None,
            ..TransportSettings::default()
        }
    }

    fn through_cube() -> Ray {
        Ray {
            origin: DVec3::new(-1.0, 0.5, 0.5),
            direction: DVec3::X,
        }
    }

    #[test]
    fn bounds_handle_parallel_axes_and_inside_origins() {
        let bounds = Bounds {
            min: DVec3::ZERO,
            max: DVec3::ONE,
        };
        assert_eq!(bounds.ray_interval(through_cube()), Some((1.0, 2.0)));
        assert_eq!(
            bounds.ray_interval(Ray {
                origin: DVec3::splat(0.5),
                direction: DVec3::Y
            }),
            Some((0.0, 0.5))
        );
        assert_eq!(
            bounds.ray_interval(Ray {
                origin: DVec3::new(-1.0, 2.0, 0.5),
                direction: DVec3::X
            }),
            None
        );
    }

    #[test]
    fn ratio_tracking_matches_beer_lambert_with_loose_majorant() {
        let field = ConstantField {
            density: 0.7,
            maximum: 2.1,
        };
        let settings = dark_settings();
        let count = 20_000;
        let mut sum = 0.0;
        let mut sum_squared = 0.0;
        for i in 0..count {
            let value = transmittance(
                &field,
                through_cube(),
                f64::INFINITY,
                &settings,
                &mut Pcg32::for_sample(9, 0, i),
            )
            .unwrap();
            sum += value;
            sum_squared += value * value;
        }
        let mean = sum / count as f64;
        let variance = sum_squared / count as f64 - mean * mean;
        let standard_error = (variance / count as f64).sqrt();
        assert!((mean - (-0.7_f64).exp()).abs() < 5.0 * standard_error);
    }

    #[test]
    fn absorbing_medium_matches_beer_lambert() {
        let field = ConstantField {
            density: 0.7,
            maximum: 2.1,
        };
        let settings = TransportSettings {
            scattering_albedo: DVec3::ZERO,
            sky_radiance: DVec3::ONE,
            ..dark_settings()
        };
        let count = 20_000;
        let mut sum = 0.0;
        for i in 0..count {
            sum += trace_sample(
                &field,
                through_cube(),
                &settings,
                &mut Pcg32::for_sample(19, 0, i),
            )
            .unwrap()
            .radiance
            .x;
        }
        let expected = (-0.7_f64).exp();
        let standard_error = (expected * (1.0 - expected) / count as f64).sqrt();
        assert!((sum / count as f64 - expected).abs() < 5.0 * standard_error);
    }

    #[test]
    fn conservative_cloud_preserves_constant_environment_exactly() {
        let field = ConstantField {
            density: 3.0,
            maximum: 3.0,
        };
        let settings = dark_settings();
        for i in 0..128 {
            let sample = trace_sample(
                &field,
                through_cube(),
                &settings,
                &mut Pcg32::for_sample(29, 0, i),
            )
            .unwrap();
            assert_eq!(sample.radiance, settings.sky_radiance);
        }
    }

    #[test]
    fn vacuum_ground_matches_lambertian_environment_and_sun() {
        let field = ConstantField {
            density: 0.0,
            maximum: 0.0,
        };
        let settings = TransportSettings {
            ground: Some(GroundPlane {
                height: -1.0,
                albedo: DVec3::splat(0.2),
            }),
            roulette_start: 100,
            ..TransportSettings::default()
        };
        let ray = Ray {
            origin: DVec3::new(2.0, 2.0, 2.0),
            direction: -DVec3::Y,
        };
        let sample = trace_sample(&field, ray, &settings, &mut Pcg32::new(7, 11)).unwrap();
        let expected = 0.2
            * (settings.sky_radiance
                + settings.sun_irradiance * (settings.sun_direction.normalize().y / PI));
        assert!((sample.radiance - expected).abs().max_element() < 1e-14);
    }

    #[test]
    fn ground_visibility_handles_surface_origins_and_underside() {
        let field = ConstantField {
            density: 0.0,
            maximum: 0.0,
        };
        let settings = TransportSettings {
            ground: Some(GroundPlane {
                height: -1.0,
                albedo: DVec3::ONE,
            }),
            ..dark_settings()
        };
        let on_plane = DVec3::new(2.0, -1.0, 2.0);
        assert_eq!(
            transmittance(
                &field,
                Ray {
                    origin: on_plane,
                    direction: -DVec3::Y
                },
                f64::INFINITY,
                &settings,
                &mut Pcg32::new(1, 1)
            )
            .unwrap(),
            0.0
        );
        assert_eq!(
            transmittance(
                &field,
                Ray {
                    origin: on_plane,
                    direction: DVec3::Y
                },
                f64::INFINITY,
                &settings,
                &mut Pcg32::new(1, 1)
            )
            .unwrap(),
            1.0
        );
        let underneath = Ray {
            origin: DVec3::new(2.0, -2.0, 2.0),
            direction: DVec3::Y,
        };
        assert_eq!(
            trace_sample(&field, underneath, &settings, &mut Pcg32::new(1, 1))
                .unwrap()
                .radiance,
            DVec3::ZERO
        );
    }

    #[test]
    fn invalid_majorant_and_watchdog_are_errors() {
        let settings = dark_settings();
        let broken = ConstantField {
            density: 200.0,
            maximum: 100.0,
        };
        assert!(matches!(
            transmittance(
                &broken,
                through_cube(),
                f64::INFINITY,
                &settings,
                &mut Pcg32::new(3, 5)
            ),
            Err(TraceError::InvalidDensity { .. })
        ));
        let dense = ConstantField {
            density: 0.0,
            maximum: 100.0,
        };
        let settings = TransportSettings {
            event_limit: 1,
            ..settings
        };
        assert!(matches!(
            trace_sample(&dense, through_cube(), &settings, &mut Pcg32::new(3, 5)),
            Err(TraceError::EventLimit { limit: 1 })
        ));
    }

    #[test]
    fn shadow_roulette_preserves_conditional_mean_exactly() {
        for transmittance in [f64::from_bits(1), 1e-309, 1e-5, 0.00009999] {
            let terminated = shadow_roulette_weight(transmittance, 0.25);
            let survived = shadow_roulette_weight(transmittance, 0.75);
            assert_eq!(terminated, 0.0);
            assert_eq!(survived, 2.0 * transmittance);
            assert_eq!(0.5 * terminated + 0.5 * survived, transmittance);
        }
    }

    #[test]
    fn thick_shadow_roulette_reduces_candidates_across_short_spans() {
        struct LongShadow {
            split: bool,
        }
        impl DensityField for LongShadow {
            fn bounds(&self) -> Bounds {
                Bounds {
                    min: DVec3::ZERO,
                    max: DVec3::new(1000.0, 1.0, 1.0),
                }
            }
            fn density_world(&self, point: DVec3) -> f64 {
                if point.cmpge(self.bounds().min).all() && point.cmple(self.bounds().max).all() {
                    0.1
                } else {
                    0.0
                }
            }
            fn max_density(&self) -> f64 {
                1.0
            }
            fn majorant_spans(
                &self,
                ray: Ray,
                endpoint: f64,
            ) -> Result<
                Box<dyn Iterator<Item = Result<crate::majorant::MajorantSpan, TraceError>> + '_>,
                TraceError,
            > {
                if !self.split {
                    return Ok(global_spans(self.bounds(), 1.0, ray, endpoint));
                }
                let mut spans = Vec::new();
                for x in 0..1000 {
                    let bounds = Bounds {
                        min: DVec3::new(x as f64, 0.0, 0.0),
                        max: DVec3::new(x as f64 + 1.0, 1.0, 1.0),
                    };
                    if let Some((start, end)) = bounds.ray_interval(ray) {
                        let end = end.min(endpoint);
                        if end > start {
                            spans.push(crate::majorant::MajorantSpan {
                                start,
                                end,
                                max_density: 1.0,
                            });
                        }
                    }
                }
                spans.sort_by(|a, b| a.start.total_cmp(&b.start));
                Ok(Box::new(spans.into_iter().map(Ok)))
            }
        }
        for split in [false, true] {
            let field = LongShadow { split };
            let mut events = [0_u64; 2];
            for (index, enabled) in [false, true].into_iter().enumerate() {
                let settings = TransportSettings {
                    shadow_roulette: enabled,
                    ..dark_settings()
                };
                for path in 0..256 {
                    let mut stats = Sample::default();
                    let value = track_transmittance(
                        &field,
                        through_cube(),
                        f64::INFINITY,
                        1.0,
                        &settings,
                        &mut Pcg32::for_sample(0x5AD0_2026, u64::from(split), path),
                        &mut Budget::default(),
                        &mut stats,
                    )
                    .unwrap();
                    assert!(value >= 0.0 && value.is_finite());
                    events[index] += stats.shadow_events;
                }
            }
            eprintln!(
                "tau=100, split={split}: no roulette {:.2} candidates/path, roulette {:.2}",
                events[0] as f64 / 256.0,
                events[1] as f64 / 256.0
            );
            assert!(
                events[1] < events[0] / 4,
                "roulette cadence must continue across short spans"
            );
        }
    }
}
