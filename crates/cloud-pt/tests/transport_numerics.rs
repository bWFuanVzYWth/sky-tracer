//! CPU-only counterexamples for invalid finite-precision geometry/proposals.

use cloud_pt::{
    config::Camera,
    majorant::MajorantSpan,
    sampling::Pcg32,
    transport::{
        Bounds, DensityField, GroundPlane, Ray, TraceError, TransportSettings, trace_sample,
        transmittance,
    },
};
use glam::DVec3;

struct Vacuum;
impl DensityField for Vacuum {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: DVec3::ZERO,
            max: DVec3::ONE,
        }
    }
    fn density_world(&self, _: DVec3) -> f64 {
        0.0
    }
    fn max_density(&self) -> f64 {
        0.0
    }
}

struct SuppliedSpan(MajorantSpan);
impl DensityField for SuppliedSpan {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: DVec3::ZERO,
            max: DVec3::ONE,
        }
    }
    fn density_world(&self, _: DVec3) -> f64 {
        0.0
    }
    fn max_density(&self) -> f64 {
        1.0
    }
    fn majorant_spans(
        &self,
        _: Ray,
        _: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<MajorantSpan, TraceError>> + '_>, TraceError> {
        Ok(Box::new(std::iter::once(Ok(self.0))))
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

fn cube_ray() -> Ray {
    Ray {
        origin: DVec3::new(-1.0, 0.5, 0.5),
        direction: DVec3::X,
    }
}

#[test]
fn overflowing_ground_distance_is_rejected_even_in_vacuum() {
    let ray = Ray {
        origin: DVec3::new(0.5, 1.0, 0.5),
        direction: DVec3::new(1.0, -1e-309, 0.0),
    };
    let settings = TransportSettings {
        ground: Some(GroundPlane {
            height: -1.0,
            albedo: DVec3::ONE,
        }),
        ..dark_settings()
    };
    assert!(matches!(
        trace_sample(&Vacuum, ray, &settings, &mut Pcg32::new(1, 2)),
        Err(TraceError::NumericalFailure(
            "nonfinite ground intersection distance"
        ))
    ));
    assert!(matches!(
        transmittance(
            &Vacuum,
            ray,
            f64::INFINITY,
            &settings,
            &mut Pcg32::new(1, 2)
        ),
        Err(TraceError::NumericalFailure(
            "nonfinite ground intersection distance"
        ))
    ));
}

#[test]
fn overflowing_ground_point_cannot_become_a_valid_ambient_sample() {
    // The ground distance is finite (1e308), but its x coordinate overflows.
    let ray = Ray {
        origin: DVec3::new(1.5e308, 8e307, 0.0),
        direction: DVec3::new(0.6, -0.8, 0.0),
    };
    let settings = TransportSettings {
        ground: Some(GroundPlane {
            height: 0.0,
            albedo: DVec3::ONE,
        }),
        ..dark_settings()
    };
    assert!(matches!(
        trace_sample(&Vacuum, ray, &settings, &mut Pcg32::new(1, 2)),
        Err(TraceError::NumericalFailure(
            "nonfinite ground intersection point"
        ))
    ));
}

#[test]
fn camera_rejects_overflow_after_finite_input_validation() {
    let mut camera = Camera {
        origin: DVec3::new(-1e308, 0.0, 0.0),
        target: DVec3::new(1e308, 0.0, 0.0),
        up: DVec3::Y,
        horizontal_fov_deg: 60.0,
    };
    assert!(
        camera.basis().is_err(),
        "finite endpoints can overflow during subtraction"
    );
    camera.origin = DVec3::ZERO;
    assert!(
        camera.basis().is_err(),
        "finite direction can overflow its normalization length"
    );
    camera.target = -DVec3::Z;
    camera.up = DVec3::new(0.0, 1e308, 0.0);
    assert!(camera.basis().is_err());
    camera.up = DVec3::Y;
    assert!(camera.basis().is_ok());
    assert!(camera.ray(1, 1, 1e308, 0.5).is_err());
}

#[test]
fn malformed_majorant_spans_fail_before_zero_rate_skip() {
    let valid = MajorantSpan {
        start: 1.0,
        end: 2.0,
        max_density: 0.0,
    };
    let malformed = [
        MajorantSpan {
            start: f64::NAN,
            ..valid
        },
        MajorantSpan {
            start: f64::INFINITY,
            ..valid
        },
        MajorantSpan {
            start: -1.0,
            ..valid
        },
        MajorantSpan {
            end: f64::NAN,
            ..valid
        },
        MajorantSpan {
            end: f64::INFINITY,
            ..valid
        },
        MajorantSpan { end: 1.0, ..valid },
        MajorantSpan {
            max_density: f64::NAN,
            ..valid
        },
        MajorantSpan {
            max_density: f64::INFINITY,
            ..valid
        },
        MajorantSpan {
            max_density: -1.0,
            ..valid
        },
    ];
    for span in malformed {
        let field = SuppliedSpan(span);
        assert!(
            matches!(
                trace_sample(&field, cube_ray(), &dark_settings(), &mut Pcg32::new(1, 2)),
                Err(TraceError::InvalidVolume("invalid majorant ray partition"))
            ),
            "delta tracking accepted {span:?}"
        );
        assert!(
            matches!(
                transmittance(
                    &field,
                    cube_ray(),
                    f64::INFINITY,
                    &dark_settings(),
                    &mut Pcg32::new(1, 2)
                ),
                Err(TraceError::InvalidVolume("invalid majorant ray partition"))
            ),
            "ratio tracking accepted {span:?}"
        );
    }
}

#[test]
fn local_rate_underflow_is_a_diagnostic_error_not_empty_space() {
    let field = SuppliedSpan(MajorantSpan {
        start: 1.0,
        end: 2.0,
        max_density: 1e-200,
    });
    // Global rate is representable, but the positive local proposal underflows.
    let settings = TransportSettings {
        extinction_scale: 1e-200,
        ..dark_settings()
    };
    assert!(matches!(
        trace_sample(&field, cube_ray(), &settings, &mut Pcg32::new(1, 2)),
        Err(TraceError::NumericalFailure(
            "local extinction majorant underflow"
        ))
    ));
    assert!(matches!(
        transmittance(
            &field,
            cube_ray(),
            f64::INFINITY,
            &settings,
            &mut Pcg32::new(1, 2)
        ),
        Err(TraceError::NumericalFailure(
            "local extinction majorant underflow"
        ))
    ));
}
