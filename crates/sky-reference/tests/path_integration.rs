use sky_reference::{
    config::{BakeConfig, CoordinateMapping, RayStepMapping},
    mapping::{Geometry, State},
    model::{BandInfo, BandModel, Coefficients, Model},
    path_integration::path_edges,
    solver::GpuBaker,
};
use std::f32::consts::PI;

fn state(g: Geometry, h: f32, mu: f32) -> State {
    State {
        altitude_km: h,
        mu,
        mu_s: 1.0,
        nu: mu,
        ground: g.hits_ground(h, mu),
    }
}

#[test]
fn cpu_path_edges_cover_monotone_rays_and_exact_tangent_split() {
    let g = Geometry {
        bottom: 6360.0,
        top: 6480.0,
    };
    for h in [
        0.0, 0.00025, 0.002, 0.2, 1.0, 12.0, 35.0, 60.0, 119.99, 120.0,
    ] {
        let horizon = g.horizon(h);
        for mu in [
            -1.0,
            horizon - 0.0001,
            horizon,
            horizon + 0.0001,
            -0.01,
            0.0,
            0.01,
            1.0,
        ] {
            let s = state(g, h, mu.clamp(-1.0, 1.0));
            let length = g.distance(h, s.mu, s.ground);
            for steps in [1, 32, 128, 192, 256, 768] {
                for mapping in [RayStepMapping::UniformDistance, RayStepMapping::LogHeightV1] {
                    let edges = path_edges(g, s, steps, mapping);
                    assert_eq!(edges.len(), steps + 1);
                    assert_eq!(edges[0], 0.0);
                    assert_eq!(edges[steps].to_bits(), length.to_bits());
                    assert!(
                        edges
                            .iter()
                            .all(|v| v.is_finite() && *v >= 0.0 && *v <= length)
                    );
                    if length > 0.0 {
                        assert!(
                            edges.windows(2).all(|e| e[1] > e[0]),
                            "nonmonotone: h={h}, mu={}, steps={steps}, mapping={mapping:?}",
                            s.mu
                        );
                    } else {
                        assert!(edges.iter().all(|&v| v == 0.0));
                    }
                    let sum = edges.windows(2).map(|e| (e[1] - e[0]) as f64).sum::<f64>();
                    assert!((sum - length as f64).abs() <= length as f64 * 1e-6);
                }
            }
        }
    }
    let s = state(g, 10.0, -0.02);
    let length = g.distance(s.altitude_km, s.mu, s.ground);
    let middle = (-(g.bottom + s.altitude_km) * s.mu).clamp(0.0, length);
    let edges = path_edges(g, s, 256, RayStepMapping::LogHeightV1);
    assert!(edges.iter().any(|e| e.to_bits() == middle.to_bits()));
}

#[test]
fn cpu_legacy_config_defaults_to_uniform_distance() {
    let original = BakeConfig::smoke();
    let mut json = serde_json::to_value(&original).unwrap();
    json.as_object_mut().unwrap().remove("ray_step_mapping");
    let loaded: BakeConfig = serde_json::from_value(json).unwrap();
    assert_eq!(loaded.ray_step_mapping, RayStepMapping::UniformDistance);
    assert_eq!(loaded, original);
    assert_eq!(
        BakeConfig::current_reference().ray_step_mapping,
        RayStepMapping::LogHeightV1
    );
}

#[test]
fn cpu_bake_path_shader_validates_without_gpu() {
    let source = include_str!("../src/bake.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
}

fn homogeneous(beta: f32) -> Model {
    let c = Coefficients {
        scattering: [beta, 0.0, 0.0, 0.0, 0.0],
        extinction: beta,
    };
    Model {
        geometry: Geometry {
            bottom: 6360.0,
            top: 6460.0,
        },
        sun_radius: 0.00465047,
        bands: vec![BandModel {
            info: BandInfo {
                center_nm: 550.0,
                lower_nm: 545.0,
                upper_nm: 555.0,
                solar_irradiance_w_m2: 1.0,
            },
            profile: vec![(0.0, c), (100.0, c)],
            phase: vec![1.0 / (4.0 * PI); 4096],
        }],
    }
}

#[test]
fn gpu_both_path_modes_preserve_vacuum_and_thin_transport() -> sky_reference::Result<()> {
    let gpu = GpuBaker::new()?;
    for mapping in [RayStepMapping::UniformDistance, RayStepMapping::LogHeightV1] {
        let config = BakeConfig {
            mapping: CoordinateMapping::RayAlignedReference,
            ray_step_mapping: mapping,
            scattering: [4, 12, 6, 9],
            optical_depth: [12, 48],
            angular_mu: 4,
            angular_phi: 8,
            ray_steps: 128,
            max_orders: 2,
            min_orders: 2,
            ground_albedo: 0.0,
            ..BakeConfig::smoke()
        };
        let vacuum = gpu.bake_band(&homogeneous(0.0), &config, 0, |_| {})?;
        assert!(
            vacuum
                .radiance
                .iter()
                .chain(&vacuum.optical_depth)
                .all(|&v| v == 0.0)
        );
        // Vacuum still delivers direct sunlight to the unreflecting ground.
        assert!(
            vacuum
                .ground_irradiance
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert!(*vacuum.ground_irradiance.last().unwrap() > 0.99);
        let model = homogeneous(1e-8);
        let band = gpu.bake_band(&model, &config, 0, |_| {})?;
        // Ground-level, zenith-Sun nodes have no occultation. Both direct and
        // second-order pipelines must preserve the analytic optically thin limit.
        for mi in config.scattering[1] / 4..config.scattering[1] {
            for ni in 0..config.scattering[3] {
                let i = (mi * config.scattering[2] + config.scattering[2] - 1)
                    * config.scattering[3]
                    + ni;
                let s = model.geometry.state_config(i, &config);
                let expected = 1e-8
                    * model.geometry.distance(s.altitude_km, s.mu, s.ground)
                    * model.bands[0].phases(s.nu)[0];
                assert!(
                    (band.radiance[i] - expected).abs() <= expected.abs() * 0.003 + 1e-11,
                    "{mapping:?}, node {i}: {} vs {expected}",
                    band.radiance[i]
                );
            }
        }
    }
    Ok(())
}
