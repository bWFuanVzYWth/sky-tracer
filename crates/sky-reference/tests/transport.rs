use glam::Vec3;
use sky_reference::{
    asset::Manifest,
    config::{BakeConfig, CoordinateMapping, MAX_ASSET_BYTES},
    mapping::{
        Geometry, State, sample, sample_radiance, sample_radiance_state, scattering_cosine, unit,
    },
    model::{BandInfo, BandModel, Coefficients, Model},
    quadrature,
    renderer::{SpectralRenderer, View},
    solver::GpuBaker,
};
use std::{f32::consts::PI, fs};

fn query(values: &[f32], g: Geometry, c: &BakeConfig, s: State) -> f32 {
    let nodes: Vec<_> = (0..c.scattering[3])
        .map(|i| scattering_cosine(unit(i, c.scattering[3])))
        .collect();
    sample_radiance_state(values, g, c, s, &nodes)
}

fn homogeneous(beta: f32) -> Model {
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
            profile: vec![
                (
                    0.0,
                    Coefficients {
                        scattering: [beta, 0.0, 0.0, 0.0, 0.0],
                        extinction: beta,
                    },
                ),
                (
                    100.0,
                    Coefficients {
                        scattering: [beta, 0.0, 0.0, 0.0, 0.0],
                        extinction: beta,
                    },
                ),
            ],
            phase: vec![1.0 / (4.0 * PI); 4096],
        }],
    }
}

#[test]
fn gpu_horizon_reference_chart_preserves_thin_transport_and_vacuum() -> sky_reference::Result<()> {
    let gpu = GpuBaker::new()?;
    let c = BakeConfig {
        scattering: [8, 32, 33, 65],
        optical_depth: [32, 128],
        max_orders: 1,
        min_orders: 1,
        ground_albedo: 0.0,
        ray_steps: 128,
        ..BakeConfig::reference()
    };
    let vacuum = homogeneous(0.0);
    let zero = gpu.bake_band(&vacuum, &c, 0, |_| {})?;
    assert!(
        zero.radiance
            .iter()
            .chain(&zero.optical_depth)
            .all(|&v| v == 0.0)
    );
    let model = homogeneous(1e-8);
    let band = gpu.bake_band(&model, &c, 0, |_| {})?;
    let g = model.geometry;
    for above in [0.0_f32, 0.1, 1.0, 10.0, 45.0, 89.0] {
        let mu = above.to_radians().sin();
        let s = State {
            altitude_km: 0.0,
            mu,
            mu_s: 1.0,
            nu: mu,
            ground: false,
        };
        let expected = 1e-8 * g.distance(0.0, mu, false) * model.bands[0].phases(mu)[0];
        let value = query(&band.radiance, g, &c, s);
        assert!(
            (value / expected - 1.0).abs() < 0.035,
            "{above}: {value} vs {expected}"
        );
    }
    Ok(())
}

#[test]
fn gpu_ray_reference_direct_first_preserves_thin_transport() -> sky_reference::Result<()> {
    let gpu = GpuBaker::new()?;
    let model = homogeneous(1e-8);
    let c = BakeConfig {
        mapping: CoordinateMapping::RayAlignedReference,
        phase_interpolation: sky_reference::config::PhaseInterpolation::LogRadiance,
        view_interpolation: sky_reference::config::ViewInterpolation::MonotoneCubic,
        iteration_scheme: sky_reference::config::IterationScheme::FixedPoint,
        scattering: [5, 32, 33, 65],
        scattering_altitudes_km: vec![0.0, 1.0, 2.0, 30.0, 100.0],
        height_interpolation: sky_reference::config::HeightInterpolation::ReferenceCdf,
        source_mapping: sky_reference::config::SourceMapping::LocalLinearHeight,
        sun_mu: 3,
        sun_phi: 23,
        optical_depth: [32, 128],
        max_orders: 3,
        min_orders: 3,
        ground_albedo: 0.0,
        ray_steps: 128,
        ..BakeConfig::reference()
    };
    let band = gpu.bake_band(&model, &c, 0, |_| {})?;
    for above in [0.0_f32, 0.1, 1.0, 10.0, 45.0, 89.0] {
        let mu = above.to_radians().sin();
        let s = State {
            altitude_km: 0.0,
            mu,
            mu_s: 1.0,
            nu: mu,
            ground: false,
        };
        let expected =
            1e-8 * model.geometry.distance(0.0, mu, false) * model.bands[0].phases(mu)[0];
        let value = query(&band.radiance, model.geometry, &c, s);
        assert!(
            (value / expected - 1.0).abs() < 0.035,
            "{above}: {value} vs {expected}"
        );
    }
    let root = std::env::temp_dir().join(format!("sky-ray-chart-test-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let mut manifest = Manifest::new(&model, c, gpu.adapter_name.clone())?;
    manifest.write_band(&root, 0, band)?;
    let loaded = manifest.read_band(&root, 0)?;
    let mut renderer = SpectralRenderer::new(gpu.device(), &root, &[[1.0, 0.0, 0.0]])?;
    for (h, pitch, sun, fov) in [
        (0.2_f32, 1.0_f32, 85.0_f32, 15.0_f32),
        (30.0, -5.5, -5.5, 2.0),
        (30.0, 15.0, -3.55, 60.0),
        (400.0, -70.0, 70.0, 45.0),
    ] {
        let values = read_render(
            &gpu,
            &mut renderer,
            View {
                yaw_deg: 0.0,
                pitch_deg: pitch,
                fov_y_deg: fov,
                sun_azimuth_deg: 0.0,
                sun_elevation_deg: sun,
                altitude_km: h,
            },
        )?;
        for y in 0..8 {
            for x in 0..8 {
                let px = (x as f32 + 0.5) / 8.0 * 2.0 - 1.0;
                let py = 1.0 - (y as f32 + 0.5) / 8.0 * 2.0;
                let forward = Vec3::new(0.0, pitch.to_radians().sin(), pitch.to_radians().cos());
                let v = (forward
                    + (Vec3::X * px + forward.cross(Vec3::X) * py)
                        * (fov * 0.5).to_radians().tan())
                .normalize();
                let cpu = loaded.sample(
                    h,
                    Vec3::new(v.z, v.x, v.y),
                    Vec3::new(sun.to_radians().cos(), 0.0, sun.to_radians().sin()),
                    true,
                )?;
                assert!(
                    (values[(y * 8 + x) * 4] - cpu).abs() < cpu.abs() * 0.005 + 1e-11,
                    "ray chart GPU/CPU at h={h}: {} vs {cpu}",
                    values[(y * 8 + x) * 4]
                );
            }
        }
    }
    for file in ["band_000.bin", "asset.json"] {
        fs::remove_file(root.join(file))?;
    }
    fs::remove_dir(root)?;
    Ok(())
}

#[test]
fn gpu_fixed_point_keeps_direct_surface_light_once() -> sky_reference::Result<()> {
    let gpu = GpuBaker::new()?;
    let model = homogeneous(0.0);
    let c = BakeConfig {
        mapping: CoordinateMapping::RayAlignedReference,
        phase_interpolation: sky_reference::config::PhaseInterpolation::LogRadiance,
        iteration_scheme: sky_reference::config::IterationScheme::FixedPoint,
        scattering: [3, 12, 9, 17],
        optical_depth: [8, 32],
        ray_steps: 16,
        ground_sun_samples: 32,
        angular_mu: 4,
        angular_phi: 8,
        max_orders: 3,
        min_orders: 3,
        ..BakeConfig::reference()
    };
    let a = gpu.bake_band(
        &model,
        &BakeConfig {
            max_orders: 1,
            min_orders: 1,
            ..c.clone()
        },
        0,
        |_| {},
    )?;
    let b = gpu.bake_band(&model, &c, 0, |_| {})?;
    assert!(a.radiance.iter().any(|v| *v > 0.0));
    assert_eq!(a.radiance, b.radiance);
    assert_eq!(a.ground_irradiance, b.ground_irradiance);
    assert_eq!(
        b.orders.last().unwrap().max_local_relative_increment,
        Some(0.0)
    );
    Ok(())
}

#[test]
fn cpu_synthesis_streams_spectra_and_matches_band_queries() -> sky_reference::Result<()> {
    use sky_reference::{
        rgb::rec2020_weights,
        solver::{BakedBand, OrderStats},
        synthesis::{SamplePoint, sample_rec2020},
    };
    let mut model = homogeneous(1e-7);
    let original = model.bands[0].clone();
    model.bands = [440.0, 550.0, 680.0]
        .into_iter()
        .map(|nm| {
            let mut b = original.clone();
            b.info.center_nm = nm;
            b.info.lower_nm = nm - 5.0;
            b.info.upper_nm = nm + 5.0;
            b
        })
        .collect();
    let c = BakeConfig {
        scattering: [5, 16, 17, 33],
        optical_depth: [8, 32],
        max_orders: 1,
        min_orders: 1,
        ..BakeConfig::reference()
    };
    let root = std::env::temp_dir().join(format!(
        "sky-lut-synthesis-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    fs::create_dir(&root)?;
    let mut manifest = Manifest::new(&model, c.clone(), "analytic dataset test".into())?;
    for i in 0..3 {
        let radiance = (0..c.scattering_len())
            .map(|index| {
                let s = model.geometry.state_config(index, &c);
                let chart = if s.ground { 0.7 } else { 1.0 };
                chart
                    * ((i + 1) as f32
                        + 0.2 * s.mu
                        + 0.3 * s.mu_s
                        + 0.1 * (i + 1) as f32 * s.nu * s.nu)
                    * (1.0 + s.altitude_km * 0.003)
            })
            .collect();
        manifest.write_band(
            &root,
            i,
            BakedBand {
                radiance,
                optical_depth: vec![0.0; c.optical_depth_len()],
                ground_irradiance: vec![0.0; c.ground_sun_samples],
                stopped_by_tolerance: false,
                orders: vec![OrderStats {
                    stage_wall_seconds: None,
                    max_local_relative_increment: None,
                    order: 1,
                    max_radiance_increment: 1.0,
                    relative_max_increment: 1.0,
                    relative_texel_sum_increment: 1.0,
                    elapsed_seconds: 0.0,
                }],
            },
        )?;
    }
    let mut points = Vec::new();
    for h in [0.0, 0.2, 30.0, 100.0, 400.0] {
        for sun in [-6.3, 47.2, 89.5] {
            for (view, az) in [(-20.0, 0.0), (0.2, 180.0), (31.0, 67.0), (89.7, 0.0)] {
                points.push(SamplePoint {
                    altitude_km: h,
                    sun_elevation_deg: sun,
                    view_elevation_deg: view,
                    relative_azimuth_deg: az,
                });
            }
        }
    }
    let actual = sample_rec2020(&root, &points)?;
    let weights = rec2020_weights(&manifest);
    let mut expected = vec![[0.0; 3]; points.len()];
    for (i, w) in weights.iter().enumerate() {
        let band = manifest.read_band(&root, i)?;
        for (j, p) in points.iter().enumerate() {
            let e = p.view_elevation_deg.to_radians();
            let a = p.relative_azimuth_deg.to_radians();
            let se = p.sun_elevation_deg.to_radians();
            let v = band.sample(
                p.altitude_km,
                Vec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin()),
                Vec3::new(se.cos(), 0.0, se.sin()),
                false,
            )?;
            for channel in 0..3 {
                expected[j][channel] += w[channel] * v;
            }
        }
    }
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        let error = Vec3::from_array(*a).distance(Vec3::from_array(*b))
            / Vec3::from_array(*b).length().max(1e-6);
        assert!(
            error < 0.002,
            "point {i} {:?}: {a:?} vs {b:?}, {error}",
            points[i]
        );
    }
    assert!(sample_rec2020(&root, &[]).is_err());
    // The CPU-only production path preserves the grid and the original Sun
    // table, and refuses incomplete/corrupted packed payloads.
    let rgb_root = root.join("rgb");
    let packed_root = root.join("packed");
    let rgb_manifest = sky_reference::rgb::export(&root, &rgb_root)?;
    let packed_manifest = sky_reference::packed::compress(&rgb_root, &packed_root, 16)?;
    let reopened = Manifest::open(&packed_root)?;
    assert_eq!(reopened.config, rgb_manifest.config);
    let packed = sky_reference::packed::PackedLut::read(&reopened, &packed_root)?;
    for channel in 0..3 {
        let (sun, radiance) = sky_reference::rgb::read_channel(&rgb_manifest, &rgb_root, channel)?;
        assert_eq!(sun, packed.sun[channel]);
        for (i, &v) in radiance.iter().enumerate() {
            assert_eq!(
                packed.fetch(i, channel).to_bits(),
                sky_reference::packed::quantize(v) << 12
            );
        }
    }
    assert!(sky_reference::packed::compress(&rgb_root, &packed_root, 16).is_err());
    assert!(sky_reference::packed::compress(&packed_root, &root.join("invalid"), 16).is_err());
    let map_path = packed_root.join("blocks.bin");
    let mut bytes = fs::read(&map_path)?;
    bytes[0] ^= 1;
    fs::write(&map_path, &bytes)?;
    assert!(sky_reference::packed::PackedLut::read(&packed_manifest, &packed_root).is_err());
    assert!(root.starts_with(std::env::temp_dir()));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn budgets_and_invalid_input_are_checked_before_gpu_work() {
    let reference = BakeConfig::reference();
    reference.validate(41).unwrap();
    assert_eq!(reference.mapping, CoordinateMapping::HorizonAligned);
    assert_eq!(reference.scattering, [80, 32, 193, 257]);
    assert!(reference.asset_bytes(41).unwrap() < 22_000_000_000);
    let c = BakeConfig::default();
    c.validate(41).unwrap();
    assert!(c.asset_bytes(41).unwrap() < MAX_ASSET_BYTES);
    let mut larger = c.clone();
    larger.scattering[3] = 320;
    assert!(larger.validate(41).is_err());
    larger.max_asset_bytes = 8_000_000_000;
    assert!(larger.validate(41).is_ok());
    let mut c = c;
    c.scattering = [usize::MAX; 4];
    assert!(c.validate(41).is_err());
    let mut c = BakeConfig::smoke();
    c.angular_mu = 0;
    assert!(c.validate(41).is_err());
    let mut c = BakeConfig::smoke();
    c.ground_albedo = f32::NAN;
    assert!(c.validate(41).is_err());
    let mut c = BakeConfig::smoke();
    c.dispatch_texels = 65;
    assert!(c.validate(41).is_err());
}

#[test]
fn height_geometry_resolves_sub_meter_paths_and_both_horizon_sides() {
    let g = homogeneous(0.0).geometry;
    assert!((g.distance(0.0001, -1.0, true) - 0.0001).abs() < 1e-9);
    let dims = [9, 32, 12, 20];
    for r in 0..dims[0] {
        let h = g.height(r as f32 / (dims[0] - 1) as f32);
        for i in 0..dims[1] {
            let (mu, ground) = g.view(h, i, dims[1]);
            let coord = g.view_coord(h, mu, ground, dims[1]);
            assert!(
                (coord - i as f32).abs() < 0.003,
                "h={h}, i={i}, coord={coord}"
            );
            let end = g.advanced(
                State {
                    altitude_km: h,
                    mu,
                    mu_s: 0.5,
                    nu: 0.0,
                    ground,
                },
                g.distance(h, mu, ground),
            );
            let expected = if ground { 0.0 } else { g.top_height() };
            // At top, tangent sky rays have zero path length.
            assert!(
                (end.altitude_km - expected).abs() < 0.015,
                "{end:?} != {expected}"
            );
        }
    }
    let table: Vec<f32> = (0..32).map(|i| if i < 16 { 2.0 } else { 7.0 }).collect();
    for ground in [false, true] {
        let x = g.view_coord(0.2, g.horizon(0.2), ground, 32);
        assert_eq!(sample(&table, [32], [x]), if ground { 2.0 } else { 7.0 });
    }
}

#[test]
fn quadrature_integrates_solid_angle_and_normalized_rayleigh() {
    let q = quadrature::sphere(32, 64);
    let area: f32 = q.iter().map(|q| q.1).sum();
    assert!((area - 4.0 * PI).abs() < 0.0001);
    let norm: f32 = q
        .iter()
        .map(|(v, w)| w * 3.0 * (1.0 + v.z * v.z) / (16.0 * PI))
        .sum();
    assert!((norm - 1.0).abs() < 0.0001);
    let cos: f32 = quadrature::hemisphere(16, 32)
        .iter()
        .map(|(v, w)| v.z * w)
        .sum();
    assert!((cos - PI).abs() < 0.0001);
}

#[test]
fn mapped_solar_interpolation_preserves_positive_exponential_attenuation() {
    use sky_reference::config::SolarInterpolation;
    let dims = [2, 2, 9, 2];
    let data: Vec<_> = (0..dims.iter().product())
        .map(|i| (-20.0 * (i / 2 % 9) as f32 / 8.0).exp())
        .collect();
    for solar in [0.2, 1.7, 3.4, 6.1, 7.8] {
        let actual = sample_radiance(
            &data,
            dims,
            [0.3, 0.8, solar, 0.5],
            SolarInterpolation::LogRadiance,
        );
        let expected = (-20.0 * solar / 8.0).exp();
        assert!((actual / expected - 1.0).abs() < 3e-6);
    }
    assert_eq!(
        sample_radiance(
            &vec![0.0; 72],
            dims,
            [0.3, 0.8, 4.2, 0.5],
            SolarInterpolation::LogRadiance
        ),
        0.0
    );
    // Linear Rec.2020 can contain negative channels outside its gamut.
    // Such endpoints must remain finite and use a signed linear fallback.
    let signed: Vec<_> = (0..72).map(|i| (i / 2 % 9) as f32 - 4.0).collect();
    let actual = sample_radiance(
        &signed,
        dims,
        [0.3, 0.8, 2.5, 0.5],
        SolarInterpolation::LogRadiance,
    );
    assert!((actual + 1.5).abs() < 1e-6);
}

#[test]
fn actual_spectral_forward_peaks_survive_the_sun_aligned_grid() {
    let scene = sky_reference::physics::data::load_scene_data(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data"),
        20.0,
        0.0,
    )
    .unwrap();
    let model = Model::from_scene(&scene).unwrap();
    let config = BakeConfig {
        scattering: [2, 8, 9, 256],
        ..BakeConfig::default()
    };
    let disk = quadrature::sun_disk(4, 16, model.sun_radius);
    for band_index in [6, 17, 30] {
        let band = &model.bands[band_index];
        let coeff = band.coefficients(0.2);
        let phase = |nu: f32| {
            let sun = Vec3::new((1.0 - nu * nu).max(0.0).sqrt(), 0.0, nu);
            disk.iter()
                .map(|&(v, w)| {
                    w * sky_reference::model::phase_weight(
                        coeff,
                        band.phases(quadrature::rotate(v, sun).z),
                    )
                })
                .sum::<f32>()
        };
        let values: Vec<_> = (0..config.scattering_len())
            .map(|i| phase(model.geometry.state_config(i, &config).nu))
            .collect();
        for angle in [0.0_f32, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let solar = 20.0_f32.to_radians();
            let elevation = (20.0 + angle).to_radians();
            let nu = angle.to_radians().cos();
            let state = State {
                altitude_km: 0.2,
                mu: elevation.sin(),
                mu_s: solar.sin(),
                nu,
                ground: false,
            };
            let actual = query(&values, model.geometry, &config, state);
            let expected = phase(nu);
            assert!(
                (actual / expected - 1.0).abs() < 0.012,
                "{} nm, {angle} degrees: {actual} vs {expected}",
                band.info.center_nm
            );
        }
    }
}

#[test]
fn gpu_transport_analytic_limits_io_and_render_agree()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // No silent adapter skip: this integration test must execute a real wgpu backend.
    let gpu = GpuBaker::new()?;
    let mut c = BakeConfig::smoke();
    c.ground_albedo = 0.0;
    c.mapping = CoordinateMapping::SunAlignedAngular;
    c.scattering[2] = 97;
    let vacuum = homogeneous(0.0);
    let black = gpu.bake_band(&vacuum, &c, 0, |_| {})?;
    assert!(
        black
            .radiance
            .iter()
            .chain(&black.optical_depth)
            .all(|v| *v == 0.0)
    );

    c.ground_albedo = 0.18;
    let reflected = gpu.bake_band(&vacuum, &c, 0, |_| {})?;
    let nadir = vacuum.geometry.coords_config(
        State {
            altitude_km: 0.0,
            mu: -1.0,
            mu_s: 1.0,
            nu: -1.0,
            ground: true,
        },
        &c,
    );
    assert!((sample(&reflected.radiance, c.scattering, nadir) / (0.18 / PI) - 1.0).abs() < 0.0001);
    assert!(reflected.orders[1].max_radiance_increment == 0.0);
    c.ground_albedo = 0.0;
    // A phase-inclusive LUT must resolve a zenith forward peak in the view
    // axis as well as nu: at a zenith sun the physical domain has nu == mu.
    let mut forward = homogeneous(0.0);
    let beta = 1e-6;
    for (_, coefficients) in &mut forward.bands[0].profile {
        coefficients.scattering = [0.0, beta, 0.0, 0.0, 0.0];
        coefficients.extinction = beta;
    }
    for (bin, value) in forward.bands[0].phase.iter_mut().take(1024).enumerate() {
        let u = (bin as f32 + 0.5) / 1024.0;
        let mu = 1.0 - 2.0 * u * u * u;
        let g = 0.9_f32;
        *value = (1.0 - g * g) / (4.0 * PI * (1.0 + g * g - 2.0 * g * mu).powf(1.5));
    }
    let mut forward_config = c.clone();
    forward_config.mapping = CoordinateMapping::SunAligned;
    forward_config.scattering = [4, 32, 17, 128];
    forward_config.max_orders = 1;
    forward_config.min_orders = 1;
    let forward_band = gpu.bake_band(&forward, &forward_config, 0, |_| {})?;
    for degrees in [5.0_f32, 10.0, 20.0] {
        let mu = degrees.to_radians().cos();
        let state = State {
            altitude_km: 0.0,
            mu,
            mu_s: 1.0,
            nu: mu,
            ground: false,
        };
        let value = query(
            &forward_band.radiance,
            forward.geometry,
            &forward_config,
            state,
        );
        let thin_limit =
            beta * forward.geometry.distance(0.0, mu, false) * forward.bands[0].phases(mu)[1];
        assert!(
            (value / thin_limit - 1.0).abs() < 0.15,
            "forward peak at {degrees} deg: {value} vs {thin_limit}"
        );
    }
    // Earth shadow has a geometric single-scattering thin-medium limit:
    // only the portion of the ray above the solar shadow contributes. This
    // exercises negative solar elevations independently of path-tracer noise.
    let shadow = homogeneous(1e-7);
    let shadow_config = BakeConfig {
        scattering: [40, 96, 64, 20],
        ray_steps: 256,
        max_orders: 1,
        min_orders: 1,
        ground_albedo: 0.0,
        ..BakeConfig::development()
    };
    let shadow_band = gpu.bake_band(&shadow, &shadow_config, 0, |_| {})?;
    for elevation in [45.0_f32, 90.0] {
        let e = elevation.to_radians();
        let s = (-6.0_f32).to_radians();
        let ray = Vec3::new(-e.cos(), 0.0, e.sin());
        let sun = Vec3::new(s.cos(), 0.0, s.sin());
        let state = State {
            altitude_km: 0.2,
            mu: ray.z,
            mu_s: sun.z,
            nu: ray.dot(sun),
            ground: false,
        };
        let step = shadow.geometry.distance(0.2, ray.z, false) / 32768.0;
        let mut lit_length = 0.0;
        for j in 0..32768 {
            let p = shadow.geometry.advanced(state, (j as f32 + 0.5) * step);
            if !shadow.geometry.hits_ground(p.altitude_km, p.mu_s) {
                lit_length += step;
            }
        }
        let expected = lit_length * 1e-7 * shadow.bands[0].phases(state.nu)[0];
        let actual = query(
            &shadow_band.radiance,
            shadow.geometry,
            &shadow_config,
            state,
        );
        assert!(
            (actual / expected - 1.0).abs() < 0.12,
            "Earth shadow at view {elevation}: {actual} vs {expected}"
        );
        let night = State {
            mu_s: -1.0,
            nu: -ray.z,
            ..state
        };
        assert_eq!(
            query(
                &shadow_band.radiance,
                shadow.geometry,
                &shadow_config,
                night,
            ),
            0.0
        );
    }
    let mut medium = homogeneous(1e-5);
    let mut second = medium.bands[0].clone();
    second.info.center_nm += 10.0;
    second.info.lower_nm += 10.0;
    second.info.upper_nm += 10.0;
    second.info.solar_irradiance_w_m2 *= 2.0;
    medium.bands.push(second);
    c.max_orders = 1;
    c.min_orders = 1;
    c.ray_steps = 128;
    c.optical_depth_steps = 256;
    c.sun_mu = 4;
    c.sun_phi = 16;
    let band = gpu.bake_band(&medium, &c, 0, |_| {})?;
    // Zenith with zenith sun: attenuation along incoming+outgoing path is constant.
    let i = ((c.scattering[1] - 1) * c.scattering[2] + c.scattering[2] - 1) * c.scattering[3];
    let expected = 1e-5 * 100.0 * (-0.001_f32).exp() * 3.0 / (8.0 * PI);
    assert!(
        (band.radiance[i] / expected - 1.0).abs() < 0.004,
        "{} vs {expected}",
        band.radiance[i]
    );
    let tau_i = c.optical_depth[1] - 1;
    assert!((band.optical_depth[tau_i] - 0.001).abs() < 1e-7);

    let root = std::env::temp_dir().join(format!("sky-lut-test-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let mut manifest = Manifest::new(&medium, c.clone(), gpu.adapter_name.clone())?;
    manifest.save(&root)?;
    manifest.write_band(&root, 0, band)?;
    manifest.write_band(&root, 1, gpu.bake_band(&medium, &c, 1, |_| {})?)?;
    let manifest = Manifest::open(&root)?;
    assert!(manifest.complete());
    let loaded = manifest.read_band(&root, 0)?;
    assert!((loaded.sample(0.0, Vec3::Z, Vec3::Z, false)? / expected - 1.0).abs() < 0.004);
    assert!(loaded.sample(-1.0, Vec3::Z, Vec3::Z, false).is_err());
    assert_eq!(loaded.sample(400.0, Vec3::Z, Vec3::X, false)?, 0.0);
    assert!(
        (loaded.sample(400.0, -Vec3::Z, Vec3::Z, false)?
            / loaded.sample(medium.geometry.top_height(), -Vec3::Z, Vec3::Z, false)?
            - 1.0)
            .abs()
            < 1e-6
    );

    let mut renderer =
        SpectralRenderer::new(gpu.device(), &root, &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    renderer.resize(gpu.device(), [8, 8]);
    for (test_altitude, test_pitch, test_solar, test_fov) in [
        (0.2_f32, 30.0_f32, 70.0_f32, 45.0_f32),
        (0.2, 30.0, 47.0, 120.0),
        (0.2, 1.0, 85.0, 15.0),
        (400.0, -70.0, 70.0, 45.0),
    ] {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        renderer.render(
            gpu.queue(),
            &mut encoder,
            View {
                yaw_deg: 0.0,
                pitch_deg: test_pitch,
                fov_y_deg: test_fov,
                sun_azimuth_deg: 0.0,
                sun_elevation_deg: test_solar,
                altitude_km: test_altitude,
            },
        );
        let output = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("render test readback"),
            size: 256 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: renderer.target(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        output.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let data = output.slice(..).get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        for y in 0..8 {
            for x in 0..8 {
                let base = y * 64 + x * 4;
                let px = (x as f32 + 0.5) / 8.0 * 2.0 - 1.0;
                let py = 1.0 - (y as f32 + 0.5) / 8.0 * 2.0;
                let forward = Vec3::new(
                    0.0,
                    test_pitch.to_radians().sin(),
                    test_pitch.to_radians().cos(),
                );
                let right = Vec3::X;
                let ray = (forward
                    + (right * px + forward.cross(right) * py)
                        * ((test_fov * 0.5).to_radians().tan()))
                .normalize();
                let sun = Vec3::new(
                    0.0,
                    test_solar.to_radians().sin(),
                    test_solar.to_radians().cos(),
                );
                let cpu = loaded.sample(
                    test_altitude,
                    Vec3::new(ray.z, ray.x, ray.y),
                    Vec3::new(sun.z, sun.x, sun.y),
                    true,
                )?;
                assert!(
                    (values[base] / cpu - 1.0).abs() < 0.003,
                    "GPU {} CPU {cpu}",
                    values[base]
                );
                assert!((values[base + 1] - 2.0 * values[base]).abs() < 1e-7);
            }
        }
        drop(data);
        output.unmap();
    }
    // Spectral conversion happens once at export. Proportional input spectra
    // commute with interpolation, giving a strict GPU equivalence check;
    // actual multi-band off-grid differences are measured separately.
    let rgb_root = root.join("rgb");
    let rgb_manifest = sky_reference::rgb::export(&root, &rgb_root)?;
    sky_reference::rgb::verify(&rgb_manifest, &rgb_root)?;
    assert!(rgb_manifest.read_band(&rgb_root, 0).is_err());
    assert!(sky_reference::rgb::export(&rgb_root, &root.join("reexport")).is_err());
    let weights = sky_reference::rgb::rec2020_weights(&manifest);
    for channel in 0..3 {
        let (solar, values) = sky_reference::rgb::read_channel(&rgb_manifest, &rgb_root, channel)?;
        for (i, &value) in values.iter().enumerate() {
            let expected = loaded.radiance[i] * (weights[0][channel] + 2.0 * weights[1][channel]);
            assert!((value - expected).abs() < 1e-6 * expected.abs().max(1.0));
        }
        assert!(solar.iter().all(|v| v.is_finite()));
    }
    let mut spectral = SpectralRenderer::new(gpu.device(), &root, &weights)?;
    let mut rgb = SpectralRenderer::new(gpu.device(), &rgb_root, &weights)?;
    for (altitude, pitch, sun, fov) in [
        (0.2, 1.0, 85.0, 15.0),
        (0.2, 85.0, 85.0, 0.1),
        (400.0, 85.0, 85.0, 0.1),
    ] {
        let view = View {
            yaw_deg: 0.0,
            pitch_deg: pitch,
            fov_y_deg: fov,
            sun_azimuth_deg: 0.0,
            sun_elevation_deg: sun,
            altitude_km: altitude,
        };
        let a = read_render(&gpu, &mut spectral, view)?;
        let b = read_render(&gpu, &mut rgb, view)?;
        for (&a, &b) in a.iter().zip(&b) {
            assert!(
                (a - b).abs() < 2e-4 * a.abs().max(1e-6),
                "RGB {b} spectral {a}"
            );
        }
    }
    let path = rgb_root.join("channel_0.bin");
    let mut bytes = fs::read(&path)?;
    bytes[10] ^= 1;
    fs::write(&path, bytes)?;
    assert!(sky_reference::rgb::verify(&rgb_manifest, &rgb_root).is_err());
    for file in [
        "channel_0.bin",
        "channel_1.bin",
        "channel_2.bin",
        "spectral_tau.bin",
        "asset.json",
    ] {
        fs::remove_file(rgb_root.join(file))?;
    }
    fs::remove_dir(rgb_root)?;
    // Detect a corrupt payload, even if it remains finite and the length matches.
    let path = root.join("band_000.bin");
    let mut bytes = fs::read(&path)?;
    bytes[10] ^= 1;
    fs::write(&path, bytes)?;
    assert!(manifest.read_band(&root, 0).is_err());
    fs::remove_file(path)?;
    fs::remove_file(root.join("band_001.bin"))?;
    fs::remove_file(root.join("asset.json"))?;
    fs::remove_dir(root)?;
    Ok(())
}

fn read_render(
    gpu: &GpuBaker,
    renderer: &mut SpectralRenderer,
    view: View,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    renderer.resize(gpu.device(), [8, 8]);
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    renderer.render(gpu.queue(), &mut encoder, view);
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("RGB regression readback"),
        size: 256 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: renderer.target(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(8),
            },
        },
        wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let data = buffer.slice(..).get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let mut result = Vec::new();
    for y in 0..8 {
        result.extend_from_slice(&values[y * 64..y * 64 + 32]);
    }
    drop(data);
    buffer.unmap();
    Ok(result)
}
