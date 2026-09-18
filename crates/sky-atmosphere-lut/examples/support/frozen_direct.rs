//! CPU counterpart of the frozen reference direct-light integral.
use sky_atmosphere_lut::{
    asset::BandLut,
    mapping::State,
    model::{BandModel, phase_weight},
    quadrature,
};

/// The ground boundary uses *total* irradiance. The remainder therefore holds
/// indirect atmospheric scattering, including scattering of ground-reflected light.
/// Uniform 256 steps and the finite solar disk match the frozen v6 integrator.
pub fn direct(
    lut: &BandLut,
    band: &BandModel,
    s: State,
    steps: usize,
    point_sun: bool,
    log_height: bool,
) -> [f32; 2] {
    let g = lut.geometry;
    let (ray, sun) = s.directions();
    let disk = if point_sun {
        vec![(glam::Vec3::Z, 1.0)]
    } else {
        quadrature::sun_disk(lut.config.sun_mu, lut.config.sun_phi, lut.sun_radius)
    };
    let lights: Vec<_> = disk
        .into_iter()
        .map(|(local, w)| {
            let v = quadrature::rotate(local, sun);
            let nu = ray.dot(v).clamp(-1.0, 1.0);
            (v.z, nu, band.phases(nu), w * lut.info.solar_irradiance_w_m2)
        })
        .collect();
    let length = g.distance(s.altitude_km, s.mu, s.ground);
    let edges = if log_height {
        height_edges(g, s, steps)
    } else {
        (0..=steps)
            .map(|j| length * j as f32 / steps as f32)
            .collect()
    };
    let mut single = 0.0_f32;
    let mut trans = 1.0_f32;
    for j in 0..steps {
        let dx = if log_height {
            edges[j + 1] - edges[j]
        } else {
            length / steps as f32
        };
        let d = if log_height {
            (edges[j] + edges[j + 1]) * 0.5
        } else {
            (j as f32 + 0.5) * dx
        };
        let point = g.advanced(s, d);
        let c = band.coefficients(point.altitude_km);
        let mut source = 0.0;
        for &(z, nu, phase, w) in &lights {
            let mu = ((g.bottom + s.altitude_km) * z + d * nu) / (g.bottom + point.altitude_km);
            if !g.hits_ground(point.altitude_km, mu) {
                source += phase_weight(c, phase)
                    * w
                    * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..point
                    });
            }
        }
        let tau = c.extinction * dx;
        let weight = if tau < 0.001 {
            dx * (1.0 - tau * 0.5 + tau * tau / 6.0)
        } else {
            (1.0 - (-tau).exp()) / c.extinction
        };
        single += trans * source * weight;
        trans *= (-tau).exp();
    }
    let boundary = if s.ground {
        trans * lut.config.ground_albedo / std::f32::consts::PI
            * lut.ground_irradiance_at(g.advanced(s, length).mu_s)
    } else {
        0.0
    };
    [single, boundary]
}

/// A bounded geometric warp: spend samples near minimum altitude, on both
/// sides of the tangent point. No source evaluations are used to place samples.
fn height_edges(g: sky_atmosphere_lut::mapping::Geometry, s: State, steps: usize) -> Vec<f32> {
    let length = g.distance(s.altitude_km, s.mu, s.ground);
    let b = (g.bottom + s.altitude_km) * s.mu;
    let middle = (-b).clamp(0.0, length);
    let h_min = g.advanced(s, middle).altitude_km;
    let h_end = g.advanced(s, length).altitude_km;
    let scale = 0.25_f32;
    let before = ((s.altitude_km - h_min).max(0.0) / scale).ln_1p();
    let after = ((h_end - h_min).max(0.0) / scale).ln_1p();
    if length <= 0.0 || before + after < 1e-5 || steps < 2 {
        return (0..=steps)
            .map(|j| length * j as f32 / steps as f32)
            .collect();
    }
    let split = if middle <= 0.0 {
        0
    } else if middle >= length {
        steps
    } else {
        ((steps as f32 * before / (before + after)).round() as usize).clamp(1, steps - 1)
    };
    (0..=steps)
        .map(|j| {
            if j == 0 {
                return 0.0;
            }
            if j == steps {
                return length;
            }
            if j == split {
                return middle;
            }
            let incoming = j < split;
            let log_h = if incoming {
                before * (1.0 - j as f32 / split as f32)
            } else {
                after * (j - split) as f32 / (steps - split) as f32
            };
            let h = h_min + scale * log_h.exp_m1();
            let c = (s.altitude_km - h) * (2.0 * g.bottom + s.altitude_km + h);
            let root = (b * b - c).max(0.0).sqrt();
            let distance = if incoming {
                c / (-b + root).max(1e-20)
            } else if b > 0.0 {
                -c / (root + b)
            } else {
                -b + root
            };
            distance.clamp(0.0, length)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn height_warp_covers_ground_grazing_and_space_paths() {
        let g = sky_atmosphere_lut::mapping::Geometry {
            bottom: 6360.0,
            top: 6480.0,
        };
        for h in [0.0, 0.001, 0.2, 12.0, 30.0, 119.9, 120.0] {
            for mu in [
                -1.0,
                g.horizon(h) - 1e-5,
                g.horizon(h) + 1e-5,
                0.0,
                0.5,
                1.0,
            ] {
                let s = State {
                    altitude_km: h,
                    mu,
                    mu_s: 0.0,
                    nu: 0.0,
                    ground: g.hits_ground(h, mu),
                };
                for n in [1, 16, 32, 64] {
                    let e = height_edges(g, s, n);
                    assert_eq!(e.len(), n + 1);
                    assert_eq!(e[0], 0.0);
                    assert_eq!(e[n], g.distance(h, mu, s.ground));
                    assert!(e.iter().all(|v| v.is_finite()));
                    assert!(
                        e.windows(2).all(|w| w[1] >= w[0]),
                        "h={h}, mu={mu}, n={n}: {e:?}"
                    );
                }
            }
        }
    }
}
