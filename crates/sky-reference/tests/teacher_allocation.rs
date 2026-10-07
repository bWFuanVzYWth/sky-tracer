//! Fitted teacher coordinates retain boundaries and the same CPU/GPU caches.
use sky_reference::{
    asset::BandLut,
    config::{BakeConfig, CoordinateAllocation, CoordinateMapping},
    mapping::{Geometry, unit},
    model::BandInfo,
    reference_mapping as m,
};

fn geometry() -> Geometry {
    Geometry {
        bottom: 6360.0,
        top: 6480.0,
    }
}

#[test]
fn teacher_fit_preserves_surface_and_material_boundaries_within_budget() {
    let g = geometry();
    let c = BakeConfig::current_reference();
    c.validate(41).unwrap();
    c.validate_top_height(g.top_height()).unwrap();
    assert_eq!(c.coordinate_allocation, CoordinateAllocation::RealtimeFitV1);
    assert_eq!(c.scattering, [72, 32, 193, 257]);
    assert!(c.asset_bytes(41).unwrap() <= 20_000_000_000);
    let heights = &c.scattering_altitudes_km;
    assert_eq!(heights.len(), c.scattering[0]);
    assert_eq!((heights[0], heights[heights.len() - 1]), (0.0, 120.0));
    assert!(heights.windows(2).all(|h| h[0] < h[1]));
    assert!(
        heights[1] < 0.001,
        "retain sub-metre boundary radiance resolution"
    );
    assert!(heights.iter().filter(|&&h| h < 1.0).count() >= 14);
    for h in [1.0, 2.0, 11.0, 12.0, 35.0] {
        assert!(heights.contains(&h), "material boundary {h} km");
    }
    for (actual, fitted) in heights
        .iter()
        .zip(m::realtime_fit_heights(g, heights.len()))
    {
        assert!((actual - fitted).abs() < 2e-5);
    }
    for (i, &h) in heights.iter().enumerate() {
        assert_eq!(m::radius_coord(g, &c, h), i as f32);
    }
    let mut bad = c.clone();
    bad.mapping = CoordinateMapping::HorizonAligned;
    assert!(bad.validate(41).is_err());
    let mut bad = c;
    bad.scattering_altitudes_km.clear();
    assert!(bad.validate(41).is_err());
}

#[test]
fn fitted_solar_coordinates_are_monotone_and_shared_by_all_node_caches() {
    let g = geometry();
    let c = BakeConfig::current_reference();
    let [_, nm, ns, nn] = c.scattering;
    for h in [0.0, 0.2, 1.0, 12.0, 35.0, 120.0] {
        let mut previous = -1.0;
        for si in 0..ns {
            let u = unit(si, ns);
            let mu = m::solar_cosine_config(g, &c, h, u);
            assert!(mu >= previous);
            previous = mu;
            let restored = m::solar_cosine_config(g, &c, h, m::solar_coord_config(g, &c, h, mu));
            assert!((restored - mu).abs() < 2e-6, "h={h}, si={si}");
        }
        assert_eq!(m::solar_cosine_config(g, &c, h, 0.0), -1.0);
        assert_eq!(m::solar_cosine_config(g, &c, h, 1.0), 1.0);
    }
    let mut display = Vec::new();
    m::append_display_nodes(g, &c, &mut display);
    let mut baked = Vec::new();
    m::append_nodes(g, &c, &mut baked);
    assert_eq!(display, baked[..display.len()]);
    for ri in [0, 1, 20, c.scattering[0] - 1] {
        for si in [0, 1, ns / 2, ns - 2, ns - 1] {
            let s = g.state_config(((ri * nm + nm - 1) * ns + si) * nn + nn / 4, &c);
            assert_eq!(s.mu_s, display[c.scattering[0] + ri * ns + si]);
            let restored = m::solar_cosine_config(
                g,
                &c,
                s.altitude_km,
                g.coords_config(s, &c)[2] / (ns - 1) as f32,
            );
            assert!((restored - s.mu_s).abs() < 2e-6);
        }
    }
}

#[test]
fn pre_fit_configs_retain_the_original_solar_mapping() {
    let c = BakeConfig::current_reference();
    let mut json = serde_json::to_value(&c).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("coordinate_allocation");
    let old: BakeConfig = serde_json::from_value(json).unwrap();
    assert_eq!(old.coordinate_allocation, CoordinateAllocation::Legacy);
    let g = geometry();
    for h in [0.0, 35.0, 120.0] {
        for mu in [-1.0, -0.2, 0.0, 0.1, 1.0] {
            assert_eq!(
                m::solar_coord_config(g, &old, h, mu),
                m::solar_coord(g, h, mu)
            );
            assert_eq!(
                m::solar_cosine_config(g, &old, h, 0.37),
                m::solar_cosine(g, h, 0.37)
            );
        }
        assert!((m::solar_coord_config(g, &c, h, 0.5) - m::solar_coord(g, h, 0.5)).abs() > 0.01);
    }
}

#[test]
fn fitted_ground_irradiance_queries_recover_the_baked_solar_nodes() {
    let g = geometry();
    let mut c = BakeConfig::current_reference();
    c.ground_sun_samples = 65;
    let n = c.ground_sun_samples;
    let lut = BandLut {
        geometry: g,
        config: c.clone(),
        info: BandInfo {
            center_nm: 550.0,
            lower_nm: 545.0,
            upper_nm: 555.0,
            solar_irradiance_w_m2: 1.0,
        },
        sun_radius: 0.00465,
        optical_depth: Vec::new(),
        radiance: Vec::new(),
        ground_irradiance: (0..n).map(|i| unit(i, n)).collect(),
        scattering_cosines: Vec::new(),
    };
    for i in 0..n {
        let u = unit(i, n);
        let mu = m::solar_cosine_config(g, &c, 0.0, u);
        assert!(
            (lut.ground_irradiance_at(mu) - u).abs() < 2e-6,
            "solar node {i}"
        );
    }
}
