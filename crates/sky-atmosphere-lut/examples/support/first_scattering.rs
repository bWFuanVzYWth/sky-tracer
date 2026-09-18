//! Independent f32 first-order integral using the stored optical-depth table.
use sky_atmosphere_lut::{
    asset::BandLut,
    mapping::State,
    model::{BandModel, phase_weight},
    quadrature,
};

pub fn first(lut: &BandLut, model: &BandModel, s: State) -> f32 {
    let g = lut.geometry;
    let (ray, sun) = s.directions();
    let disk: Vec<_> = quadrature::sun_disk(8, 32, lut.sun_radius)
        .into_iter()
        .map(|(v, w)| {
            let v = quadrature::rotate(v, sun);
            (v, model.phases(v.dot(ray)), w)
        })
        .collect();
    let dx = g.distance(s.altitude_km, s.mu, s.ground) / 512.0;
    let mut sum = 0.0;
    let mut tr = 1.0;
    for i in 0..512 {
        let d = (i as f32 + 0.5) * dx;
        let p = g.advanced(s, d);
        let c = model.coefficients(p.altitude_km);
        let mut source = 0.0;
        for &(v, phase, w) in &disk {
            let mu =
                ((g.bottom + s.altitude_km) * v.z + d * ray.dot(v)) / (g.bottom + p.altitude_km);
            if !g.hits_ground(p.altitude_km, mu) {
                source += w
                    * phase_weight(c, phase)
                    * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..p
                    });
            }
        }
        let tau = c.extinction * dx;
        let weight = if tau < 0.001 {
            dx * (1.0 - tau * 0.5 + tau * tau / 6.0)
        } else {
            (1.0 - (-tau).exp()) / c.extinction
        };
        sum += tr * source * weight * lut.info.solar_irradiance_w_m2;
        tr *= (-tau).exp();
    }
    sum
}
