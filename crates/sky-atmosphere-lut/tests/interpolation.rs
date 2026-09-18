//! CPU-only regression tests; safe to run while another application owns the GPU.
use sky_atmosphere_lut::{
    config::{BakeConfig, CoordinateMapping},
    mapping::{Geometry, State, sample_radiance, sample_radiance_state, scattering_cosine, unit},
};

#[test]
fn zenith_cap_preserves_twilight_nodes_and_resolves_noon_air_mass() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    let mapping = CoordinateMapping::SunAlignedAngular;
    for h in [0.0, 0.2, 15.0, 80.0, 120.0] {
        for i in 0..=62 {
            let old = g.solar_cosine_mapped(h, unit(i, 65), CoordinateMapping::SunAligned);
            let new = g.solar_cosine_mapped(h, unit(i, 97), mapping);
            assert!(
                (old - new).abs() < 2e-6,
                "twilight node moved at h={h} i={i}"
            );
        }
        let mut previous = -1.01;
        for i in 0..97 {
            let mu = g.solar_cosine_mapped(h, unit(i, 97), mapping);
            assert!(mu >= previous);
            previous = mu;
            // The very last angles round to mu=1 in f32. Check physical
            // direction roundtrips instead of demanding impossible index accuracy.
            let u = g.solar_coord_mapped(h, mu, mapping);
            let restored = g.solar_cosine_mapped(h, u, mapping);
            assert!((mu - restored).abs() < 2e-6);
        }
        for i in 0..96 {
            let left = g.solar_cosine_mapped(h, unit(i, 97), mapping);
            let right = g.solar_cosine_mapped(h, unit(i + 1, 97), mapping);
            let middle = g.solar_cosine_mapped(h, unit(2 * i + 1, 193), mapping);
            assert!(middle >= left && middle <= right);
            if i < 93 {
                assert!(middle > left && middle < right);
            }
        }
    }
    let analytic = |s: State| {
        let scale = (16.0 / (g.bottom + s.altitude_km)).sqrt();
        scale / (scale + (s.mu - g.horizon(s.altitude_km)).abs())
    };
    let mut errors = Vec::new();
    for (mapping, ns) in [(CoordinateMapping::SunAligned, 65), (mapping, 97)] {
        let c = BakeConfig {
            mapping,
            scattering: [2, 32, ns, 256],
            ..BakeConfig::default()
        };
        let values: Vec<_> = (0..c.scattering_len())
            .map(|i| analytic(g.state_config(i, &c)))
            .collect();
        let nodes: Vec<_> = (0..256).map(|i| scattering_cosine(unit(i, 256))).collect();
        let mut max_error = 0.0_f32;
        for sun in [80.0_f32, 85.0, 88.0, 89.0] {
            for j in 0..21 {
                let e = (j as f32 * 0.05).to_radians();
                let solar = sun.to_radians();
                let s = State {
                    altitude_km: 0.0,
                    mu: e.sin(),
                    mu_s: solar.sin(),
                    nu: (solar - e).cos(),
                    ground: false,
                };
                let actual = sample_radiance_state(&values, g, &c, s, &nodes);
                max_error = max_error.max((actual / analytic(s) - 1.0).abs());
            }
        }
        errors.push(max_error);
    }
    assert!(
        errors[0] > 0.1,
        "regression must expose old noon distortion: {errors:?}"
    );
    // This artificial field has a sharper horizon cusp than the measured
    // transport profile. Demand a substantial reduction, not transport's bound.
    assert!(
        errors[1] < 0.1 && errors[1] < errors[0] * 0.15,
        "zenith cap did not sufficiently reduce noon distortion: {errors:?}"
    );
}

#[test]
fn physical_view_stays_smooth_across_the_solar_cone_horizon() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    let c = BakeConfig {
        scattering: [2, 32, 65, 256],
        ..BakeConfig::default()
    };
    // A smooth, finite air-mass field independent of solar direction must not
    // acquire a ring when the parameterized cone starts intersecting Earth.
    let analytic = |s: State| {
        let scale = (16.0 / (g.bottom + s.altitude_km)).sqrt();
        scale / (scale + (s.mu - g.horizon(s.altitude_km)).abs())
    };
    let values: Vec<_> = (0..c.scattering_len())
        .map(|i| {
            let s = g.state_config(i, &c);
            analytic(s)
        })
        .collect();
    let nodes: Vec<_> = (0..c.scattering[3])
        .map(|i| scattering_cosine(unit(i, c.scattering[3])))
        .collect();
    let mut old_max = 0.0_f32;
    let mut new_max = 0.0_f32;
    for i in 430..=510 {
        let theta = (i as f32 * 0.1).to_radians();
        let mu_s = 47.0_f32.to_radians().sin();
        let s = State {
            altitude_km: 0.0,
            mu: mu_s * theta.cos(),
            mu_s,
            nu: theta.cos(),
            ground: false,
        };
        let exact = analytic(s);
        let old = sample_radiance(
            &values,
            c.scattering,
            g.coords_config(s, &c),
            c.solar_interpolation,
        );
        let new = sample_radiance_state(&values, g, &c, s, &nodes);
        old_max = old_max.max((old / exact - 1.0).abs());
        new_max = new_max.max((new / exact - 1.0).abs());
    }
    assert!(
        old_max > 0.04,
        "test must expose the original ring: {old_max}"
    );
    assert!(new_max < 0.004, "smooth field distorted: {new_max}");
}

#[test]
fn corner_projection_keeps_ground_and_sky_disconnected() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    let c = BakeConfig {
        scattering: [2, 12, 9, 32],
        ..BakeConfig::default()
    };
    let values: Vec<_> = (0..c.scattering_len())
        .map(|i| if i / (9 * 32) % 12 < 3 { 37.0 } else { 2.0 })
        .collect();
    let nodes: Vec<_> = (0..32).map(|i| scattering_cosine(unit(i, 32))).collect();
    for h in [0.0, 0.2, 40.0, 120.0] {
        for mu_s in [-1.0_f32, -0.1, 0.7, 1.0] {
            for ground in [false, true] {
                let s = State {
                    altitude_km: h,
                    mu: g.horizon(h),
                    mu_s,
                    nu: 0.0,
                    ground,
                };
                let value = sample_radiance_state(&values, g, &c, s, &nodes);
                assert!((value - if ground { 37.0 } else { 2.0 }).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn bake_and_display_wgsl_pass_cpu_validation() {
    let source = format!(
        "{}\n{}",
        include_str!("../src/bake.wgsl"),
        include_str!("../src/render.wgsl")
    );
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
}
