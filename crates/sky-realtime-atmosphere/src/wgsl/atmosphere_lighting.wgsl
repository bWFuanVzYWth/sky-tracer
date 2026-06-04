struct VoxelAtmosphereLighting {
    planet_radius_km: f32,
    atmosphere_thickness_km: f32,
    eye_distance_to_planet_center_km: f32,
    _pad0: f32,
    sun_dir: vec3<f32>,
    _pad1: f32,
    sun_spectral_irradiance: vec4<f32>,
}

fn ca_atmosphere_linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    let m = mat4x3<f32>(
        vec3<f32>(83.460, 1.554, -0.043),
        vec3<f32>(49.968, 86.062, -2.182),
        vec3<f32>(-11.823, 29.205, 29.153),
        vec3<f32>(6.811, -8.283, 104.377),
    );
    return (m * l) * vec3<f32>(0.9441, 0.9888, 1.0761);
}

fn ca_atmosphere_transmittance_uv(
    p: VoxelAtmosphereLighting,
    cos_theta: f32,
    normalized_altitude: f32,
) -> vec2<f32> {
    let r_km = p.planet_radius_km
        + clamp(normalized_altitude, 0.0, 1.0) * p.atmosphere_thickness_km;
    let mu = clamp(cos_theta, -1.0, 1.0);
    let bottom = p.planet_radius_km;
    let top = bottom + p.atmosphere_thickness_km;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = sqrt(max(r_km * r_km - bottom * bottom, 0.0));
    let discriminant = r_km * r_km * (mu * mu - 1.0) + top * top;
    let d = max(0.0, -r_km * mu + sqrt(max(discriminant, 0.0)));
    let d_min = top - r_km;
    let d_max = rho + h;
    let x_mu = clamp((d - d_min) / max(d_max - d_min, 1.0e-6), 0.0, 1.0);
    let x_r = clamp(rho / max(h, 1.0e-6), 0.0, 1.0);
    return vec2<f32>(x_mu, x_r);
}

fn ca_atmosphere_transmittance_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    p: VoxelAtmosphereLighting,
    cos_theta: f32,
    normalized_altitude: f32,
) -> vec4<f32> {
    let uv = ca_atmosphere_transmittance_uv(p, cos_theta, normalized_altitude);
    return textureSampleLevel(lut, samp, uv, 0.0);
}

fn ca_atmosphere_sun_transmittance_rec2020(
    lut: texture_2d<f32>,
    samp: sampler,
    p: VoxelAtmosphereLighting,
    cos_theta: f32,
    normalized_altitude: f32,
) -> vec3<f32> {
    let t_sun = ca_atmosphere_transmittance_from_lut(
        lut,
        samp,
        p,
        cos_theta,
        normalized_altitude,
    );
    let irradiance_through = ca_atmosphere_linear_rec2020_from_spectral(
        p.sun_spectral_irradiance * t_sun,
    );
    let irradiance_total = ca_atmosphere_linear_rec2020_from_spectral(
        p.sun_spectral_irradiance,
    );
    return clamp(
        irradiance_through / max(irradiance_total, vec3<f32>(1.0e-6)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}
