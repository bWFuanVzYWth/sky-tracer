//! CPU checks of geometric invariants, independent of noisy transport images.
use sky_reference::{
    config::BakeConfig,
    mapping::{Geometry, State, unit},
    reference_mapping as m,
};

fn geometry() -> Geometry {
    Geometry {
        bottom: 6360.0,
        top: 6480.0,
    }
}

#[test]
fn cubic_phase_is_bounded_and_has_shared_cell_derivatives() {
    for values in [
        [0.0, 1.0, 4.0, 9.0, 16.0],
        [-3.0, -1.0, 0.0, 0.5, 2.0],
        [0.0, 2.0, 1.0, 4.0, 3.0],
        [1.0; 5],
    ] {
        let left = [values[0], values[1], values[2], values[3]];
        let right = [values[1], values[2], values[3], values[4]];
        for i in 0..101 {
            let value = m::cubic_phase(left, i as f32 / 100.0);
            assert!(value >= values[1].min(values[2]) && value <= values[1].max(values[2]));
        }
        let eps = 0.001;
        let a = (m::cubic_phase(left, 1.0) - m::cubic_phase(left, 1.0 - eps)) / eps;
        let b = (m::cubic_phase(right, eps) - m::cubic_phase(right, 0.0)) / eps;
        assert!((a - b).abs() < 0.015, "{values:?}: {a} vs {b}");
    }
}

#[test]
fn log_phase_preserves_exponentials_signed_fallback_and_black() {
    for k in [-12.0_f32, -3.0, 0.0, 3.0, 12.0] {
        let values = [(-k).exp(), 1.0, k.exp(), (2.0 * k).exp()];
        for i in 0..65 {
            let t = i as f32 / 64.0;
            let expected = (k * t).exp();
            assert!((m::log_cubic_phase(values, t) / expected - 1.0).abs() < 4e-6);
        }
    }
    assert_eq!(m::log_cubic_phase([0.0; 4], 0.3), 0.0);
    let signed = [-3.0, -1.0, 2.0, 4.0];
    assert_eq!(m::log_cubic_phase(signed, 0.6), m::cubic_phase(signed, 0.6));
}

#[test]
fn synthesis_rejects_invalid_query_coordinates() {
    let bad = sky_reference::synthesis::SamplePoint {
        altitude_km: 0.0,
        sun_elevation_deg: 91.0,
        view_elevation_deg: 0.0,
        relative_azimuth_deg: 0.0,
    };
    assert!(bad.packed().is_err());
}

#[test]
fn reference_axes_are_monotone_refinable_and_resolve_both_altitude_ends() {
    let g = geometry();
    let c = BakeConfig::reference();
    c.validate(41).unwrap();
    assert!(m::height(g, unit(1, c.optical_depth[0])) < 0.0001);
    for i in 0..256 {
        let a = m::height(g, unit(i, 257));
        let b = m::height(g, unit(i + 1, 257));
        let mid = m::height(g, unit(2 * i + 1, 513));
        assert!(a < mid && mid < b);
        assert!((m::height_coord(g, a) - unit(i, 257)).abs() < 2e-6);
    }
    for h in [0.0, 0.2, 10.0, 60.0, 120.0] {
        let mut prev = -1.0;
        for i in 0..193 {
            let mu = m::solar_cosine(g, h, unit(i, 193));
            assert!(mu >= prev);
            prev = mu;
            let restored = m::solar_cosine(g, h, m::solar_coord(g, h, mu));
            assert!((restored - mu).abs() < 2e-6);
        }
    }
}

#[test]
fn horizon_chart_has_no_solar_rings_in_a_pure_air_mass_field() {
    let g = geometry();
    let c = BakeConfig::reference();
    let air_mass = |s: State| {
        let x = (s.mu - g.horizon(s.altitude_km)).abs();
        0.05 / (0.05 + x)
    };
    for sun in [20.0_f32, 45.0, 60.0, 80.0, 85.0, 89.0, 90.0] {
        for az in [0.0_f32, 30.0, 90.0, 180.0] {
            for above in [0.0_f32, 0.002, 0.1, 1.0, 5.0] {
                let e = g.horizon(0.2).asin() + above.to_radians();
                let se = sun.to_radians();
                let s = State {
                    altitude_km: 0.2,
                    mu: e.sin(),
                    mu_s: se.sin(),
                    nu: e.sin() * se.sin() + e.cos() * se.cos() * az.to_radians().cos(),
                    ground: false,
                };
                let actual = m::sample_with(g, &c, s, |i| air_mass(g.state_config(i, &c)));
                let error = (actual / air_mass(s) - 1.0).abs();
                assert!(error < 0.01, "sun={sun}, az={az}, above={above}: {error}");
            }
        }
    }
}

#[test]
fn reference_branches_are_disconnected_and_nodes_are_realizable() {
    let g = geometry();
    let c = BakeConfig::reference();
    let [_, nm, ns, nn] = c.scattering;
    let ng = nm / 4;
    for h in [0.0, 0.2, 30.0, 120.0] {
        for sun in [-90.0_f32, -12.0, -6.0, 0.0, 60.0, 89.0, 90.0] {
            for ground in [false, true] {
                let mu = g.horizon(h);
                let mu_s = sun.to_radians().sin();
                for k in 0..nn {
                    let mut s = State {
                        altitude_km: h,
                        mu,
                        mu_s,
                        nu: 0.0,
                        ground,
                    };
                    s.nu = m::phase_cosine(g, s, unit(k, nn));
                    for index in if ground { 0..ng } else { ng..nm } {
                        s.mu = g.optical_cone_view(s, index, nm);
                        let (view, solar) = s.directions();
                        assert!(
                            (view.dot(solar) - s.nu).abs() < 0.0003,
                            "h={h} sun={sun} k={k} mi={index} {s:?} dot={}",
                            view.dot(solar)
                        );
                        assert!(if ground {
                            s.mu <= g.horizon(h) + 2e-6
                        } else {
                            s.mu >= g.horizon(h) - 2e-6
                        });
                    }
                }
                let s = State {
                    altitude_km: h,
                    mu,
                    mu_s,
                    nu: mu * mu_s,
                    ground,
                };
                let actual =
                    m::sample_with(
                        g,
                        &c,
                        s,
                        |i| if i / (ns * nn) % nm < ng { 37.0 } else { 2.0 },
                    );
                assert!((actual - if ground { 37.0 } else { 2.0 }).abs() < 0.0001);
            }
        }
    }
}

#[test]
fn cubic_view_stencil_never_reads_across_the_ground_sky_boundary() {
    let g = geometry();
    let c = BakeConfig {
        mapping: sky_reference::config::CoordinateMapping::RayAlignedReference,
        view_interpolation: sky_reference::config::ViewInterpolation::MonotoneCubic,
        ..BakeConfig::reference()
    };
    let [_, nm, ns, nn] = c.scattering;
    for h in [0.0, 0.2, 30.0, 120.0] {
        for sun in [-12.0_f32, -6.0, 0.0, 60.0, 89.0] {
            for above in [-0.1_f32, -0.001, 0.001, 0.1] {
                let mu = (g.horizon(h).asin() + above.to_radians()).sin();
                let mu_s = sun.to_radians().sin();
                let s = State {
                    altitude_km: h,
                    mu,
                    mu_s,
                    nu: mu * mu_s,
                    ground: above < 0.0,
                };
                let expected = if s.ground { 37.0 } else { 2.0 };
                let value = m::ReferenceStencil::new(g, &c, s, None).sample_with(|i| {
                    let ground = i / (ns * nn) % nm < nm / 4;
                    assert_eq!(ground, s.ground);
                    if ground { 37.0 } else { 2.0 }
                });
                assert!((value - expected).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn explicit_scattering_heights_roundtrip_and_leave_tau_axis_independent() {
    let g = geometry();
    let mut c = BakeConfig {
        mapping: sky_reference::config::CoordinateMapping::RayAlignedReference,
        scattering: [6, 12, 9, 17],
        scattering_altitudes_km: vec![0.0, 0.2, 1.0, 2.0, 35.0, 120.0],
        ..BakeConfig::reference()
    };
    c.validate(41).unwrap();
    c.validate_top_height(120.0).unwrap();
    let [_, nm, ns, nn] = c.scattering;
    for i in 0..c.scattering[0] {
        let h = m::radius(g, &c, i);
        assert_eq!(m::radius_coord(g, &c, h), i as f32);
        let s = g.state_config(((i * nm + nm - 1) * ns + 4) * nn + 6, &c);
        assert_eq!(s.altitude_km, h);
        let value = sky_reference::ray_mapping::RayStencil::new_local_source(g, &c, s, None)
            .with_linear_radius()
            .sample_with(|index| m::radius(g, &c, index / (nm * ns * nn)));
        assert!((value - h).abs() < 1e-4);
    }
    assert_eq!(m::radius_coord(g, &c, 1.5), 2.5);
    assert_eq!(g.height_mapped(0.5, c.mapping), m::height(g, 0.5));
    assert!(c.validate_top_height(100.0).is_err());
    c.scattering_altitudes_km[3] = 1.0;
    assert!(c.validate(41).is_err());
}
