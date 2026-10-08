//! Independent CPU statistical checks of spatial proposal rates. No test here
//! creates an adapter, GPU device, window, or loads the Disney density asset.

use cloud_pt::{
    config::{Camera, RenderConfig},
    film::render_cpu,
    majorant::MajorantSpan,
    sampling::Pcg32,
    transport::{
        Bounds, DensityField, Ray, TraceError, TransportSettings, trace_sample, transmittance,
    },
};
use glam::DVec3;

/// Three unit-thickness slabs with one exactly empty interval in the middle.
struct Slabs;

impl DensityField for Slabs {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: DVec3::ZERO,
            max: DVec3::new(3.0, 1.0, 1.0),
        }
    }

    fn density_world(&self, point: DVec3) -> f64 {
        if point.x < 0.0
            || point.x >= 3.0
            || point.y < 0.0
            || point.y > 1.0
            || point.z < 0.0
            || point.z > 1.0
        {
            0.0
        } else if point.x < 1.0 {
            0.2
        } else if point.x < 2.0 {
            0.0
        } else {
            1.2
        }
    }

    fn max_density(&self) -> f64 {
        1.2
    }

    fn majorant_spans(
        &self,
        ray: Ray,
        endpoint: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<MajorantSpan, TraceError>> + '_>, TraceError> {
        let mut spans = Vec::new();
        for (slab, maximum) in [0.2, 0.0, 1.2].into_iter().enumerate() {
            let bounds = Bounds {
                min: DVec3::new(slab as f64, 0.0, 0.0),
                max: DVec3::new(slab as f64 + 1.0, 1.0, 1.0),
            };
            if let Some((start, end)) = bounds.ray_interval(ray) {
                let end = end.min(endpoint);
                if end > start {
                    spans.push(MajorantSpan {
                        start,
                        end,
                        max_density: maximum,
                    });
                }
            }
        }
        spans.sort_by(|a, b| a.start.total_cmp(&b.start));
        Ok(Box::new(spans.into_iter().map(Ok)))
    }
}

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

fn settings(spatial: bool) -> TransportSettings {
    TransportSettings {
        spatial_majorants: spatial,
        extinction_scale: 1.0,
        scattering_albedo: DVec3::ZERO,
        sun_irradiance: DVec3::ZERO,
        sky_radiance: DVec3::ONE,
        ground: None,
        ..TransportSettings::default()
    }
}

fn axis_ray(reverse: bool) -> Ray {
    Ray {
        origin: DVec3::new(if reverse { 4.0 } else { -1.0 }, 0.5, 0.5),
        direction: if reverse { -DVec3::X } else { DVec3::X },
    }
}

#[derive(Default)]
struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
}
impl Moments {
    fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }
    fn standard_error(&self) -> f64 {
        (self.m2 / (self.count - 1) as f64 / self.count as f64).sqrt()
    }
    fn matches(&self, expected: f64, label: &str) {
        assert!(
            (self.mean - expected).abs() <= 5.0 * self.standard_error() + 1e-12,
            "{label}: estimate {} ± {} (1 SE), expected {expected}",
            self.mean,
            self.standard_error()
        );
    }
}

#[test]
fn spatial_and_global_tracking_match_piecewise_beer_lambert_both_directions() {
    // 4 configurations x 2 estimators x 12,500 = 100,000 tiny CPU paths.
    // Optical thickness is 0.2*1 + 0*1 + 1.2*1, independent of direction.
    const COUNT: u64 = 12_500;
    let expected = (-1.4_f64).exp();
    let mut global_nulls = 0;
    let mut local_nulls = 0;
    for reverse in [false, true] {
        let ray = axis_ray(reverse);
        let mut estimates = Vec::new();
        for spatial in [false, true] {
            let settings = settings(spatial);
            let mut ratio = Moments::default();
            let mut delta = Moments::default();
            for sample_index in 0..COUNT {
                let key = u64::from(reverse) * 2 + u64::from(spatial);
                let mut rng = Pcg32::for_sample(0xACCE_1E12, key, sample_index);
                ratio.add(transmittance(&Slabs, ray, f64::INFINITY, &settings, &mut rng).unwrap());
                let mut rng = Pcg32::for_sample(0xDE17_A123, key, sample_index);
                let sample = trace_sample(&Slabs, ray, &settings, &mut rng).unwrap();
                assert!(sample.radiance == DVec3::ZERO || sample.radiance == DVec3::ONE);
                assert_eq!(sample.collisions, u64::from(sample.radiance == DVec3::ZERO));
                assert_eq!(sample.shadow_events, 0);
                if spatial {
                    local_nulls += sample.null_collisions;
                } else {
                    global_nulls += sample.null_collisions;
                }
                delta.add(sample.radiance.x);
            }
            ratio.matches(
                expected,
                &format!("ratio, reverse={reverse}, spatial={spatial}"),
            );
            delta.matches(
                expected,
                &format!("delta, reverse={reverse}, spatial={spatial}"),
            );
            estimates.push((ratio, delta));
        }
        for estimator in 0..2 {
            let (a, b) = if estimator == 0 {
                (&estimates[0].0, &estimates[1].0)
            } else {
                (&estimates[0].1, &estimates[1].1)
            };
            let error = a.standard_error().hypot(b.standard_error());
            assert!(
                (a.mean - b.mean).abs() <= 5.0 * error,
                "spatial/global estimates disagree"
            );
        }
    }
    assert_eq!(
        local_nulls, 0,
        "exact local slab proposals cannot have null collisions"
    );
    assert!(
        global_nulls > 1_000,
        "global proposal should waste candidates in the empty/low-density slabs"
    );
}

#[test]
fn spatial_partition_clips_at_empty_intervals_and_exact_endpoints() {
    for reverse in [false, true] {
        let ray = axis_ray(reverse);
        let spans: Vec<_> = Slabs
            .majorant_spans(ray, 3.0)
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].start, spans[0].end), (1.0, 2.0));
        assert_eq!(
            (spans[1].start, spans[1].end, spans[1].max_density),
            (2.0, 3.0, 0.0)
        );
        assert_eq!(spans[0].max_density, if reverse { 1.2 } else { 0.2 });
        let empty_ray = Ray {
            origin: DVec3::new(1.5, 0.5, 0.5),
            direction: ray.direction,
        };
        for spatial in [false, true] {
            for endpoint in [0.0, 0.25, 0.5] {
                for index in 0..32 {
                    let value = transmittance(
                        &Slabs,
                        empty_ray,
                        endpoint,
                        &settings(spatial),
                        &mut Pcg32::for_sample(7, 0, index),
                    )
                    .unwrap();
                    assert_eq!(value, 1.0);
                }
            }
            for endpoint in [0.0, 0.5, 1.0] {
                assert_eq!(
                    transmittance(
                        &Slabs,
                        ray,
                        endpoint,
                        &settings(spatial),
                        &mut Pcg32::new(5, 9)
                    )
                    .unwrap(),
                    1.0
                );
            }
        }
    }
}

#[test]
fn finite_shadow_endpoints_preserve_integrated_extinction() {
    const COUNT: u64 = 3_000;
    for reverse in [false, true] {
        let ray = axis_ray(reverse);
        // Both endpoint 2 and 3 have crossed the first dense slab only; the
        // second endpoint adds an exactly empty unit interval.
        let expected = (if reverse { -1.2_f64 } else { -0.2_f64 }).exp();
        for spatial in [false, true] {
            for endpoint in [2.0, 3.0] {
                let mut moments = Moments::default();
                for index in 0..COUNT {
                    moments.add(
                        transmittance(
                            &Slabs,
                            ray,
                            endpoint,
                            &settings(spatial),
                            &mut Pcg32::for_sample(
                                0xEC11_5EED,
                                u64::from(reverse) * 2 + u64::from(spatial),
                                index,
                            ),
                        )
                        .unwrap(),
                    );
                }
                moments.matches(expected, "finite endpoint");
            }
        }
    }
}

#[test]
fn tiny_vacuum_film_has_exact_mean_zero_variance_and_progress() {
    let camera = Camera {
        origin: DVec3::new(-2.0, 0.5, 0.5),
        target: DVec3::splat(0.5),
        up: DVec3::Y,
        horizontal_fov_deg: 50.0,
    };
    let config = RenderConfig {
        width: 4,
        height: 3,
        spp: 4,
        seed: 19,
        sample_batch_size: 0,
    };
    for spatial in [false, true] {
        let settings = TransportSettings {
            sky_radiance: DVec3::new(0.25, 0.5, 1.0),
            ..settings(spatial)
        };
        let mut progress = Vec::new();
        let film = render_cpu(&Vacuum, &camera, &settings, &config, |n| progress.push(n)).unwrap();
        film.validate().unwrap();
        assert_eq!((film.width, film.height, film.samples_per_pixel), (4, 3, 4));
        assert_eq!(progress, [1, 2, 3, 4]);
        assert!(film.mean.iter().all(|value| *value == [0.25, 0.5, 1.0]));
        assert!(film.sample_variance.iter().all(|value| *value == [0.0; 3]));
    }
}

#[test]
fn camera_uses_horizontal_fov_symmetric_rays_and_image_aspect() {
    let camera = Camera {
        origin: DVec3::new(2.0, 3.0, 4.0),
        target: DVec3::new(2.0, 3.0, 3.0),
        up: DVec3::Y,
        horizontal_fov_deg: 60.0,
    };
    let (forward, right, up) = camera.basis().unwrap();
    let center = camera.ray(8, 4, 4.0, 2.0).unwrap();
    assert!((center.direction - forward).length() < 1e-14);
    let horizontal = camera.ray(8, 4, 8.0, 2.0).unwrap().direction;
    let opposite = camera.ray(8, 4, 0.0, 2.0).unwrap().direction;
    assert!((horizontal.dot(right) + opposite.dot(right)).abs() < 1e-14);
    assert!((horizontal.dot(forward) - opposite.dot(forward)).abs() < 1e-14);
    let tan_half_fov = (30.0_f64).to_radians().tan();
    assert!((horizontal.dot(right) / horizontal.dot(forward) - tan_half_fov).abs() < 1e-14);
    let top = camera.ray(8, 4, 4.0, 0.0).unwrap().direction;
    assert!((top.dot(up) / top.dot(forward) - tan_half_fov * 0.5).abs() < 1e-14);
}
