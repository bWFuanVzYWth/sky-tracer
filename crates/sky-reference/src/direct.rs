//! CPU counterpart of the frozen reference direct-light integral.
use crate::{
    asset::BandLut,
    mapping::State,
    model::{BandModel, phase_weight},
    quadrature,
};

/// The ground boundary uses *total* irradiance. The remainder therefore holds
/// indirect atmospheric scattering, including scattering of ground-reflected light.
/// Uniform 256 steps and the finite solar disk match the frozen v6 integrator.
pub fn direct(lut: &BandLut, band: &BandModel, s: State, steps: usize) -> [f32; 2] {
    let g = lut.geometry;
    let (ray, sun) = s.directions();
    let disk = quadrature::sun_disk(lut.config.sun_mu, lut.config.sun_phi, lut.sun_radius);
    let lights: Vec<_> = disk
        .into_iter()
        .map(|(local, w)| {
            let v = quadrature::rotate(local, sun);
            let nu = ray.dot(v).clamp(-1.0, 1.0);
            (v.z, nu, band.phases(nu), w * lut.info.solar_irradiance_w_m2)
        })
        .collect();
    let length = g.distance(s.altitude_km, s.mu, s.ground);
    let mut single = 0.0_f32;
    let mut trans = 1.0_f32;
    for j in 0..steps {
        let dx = length / steps as f32;
        let d = (j as f32 + 0.5) * dx;
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
