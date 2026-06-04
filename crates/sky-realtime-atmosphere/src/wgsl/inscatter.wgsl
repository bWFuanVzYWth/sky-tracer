// Single-scattering ray march with three aerosol phase LUT species.

fn phase_lut_uv(cos_theta: f32) -> f32 {
    return pow(max(0.5 - 0.5 * clamp(cos_theta, -1.0, 1.0), 0.0), 1.0 / 3.0);
}

fn aerosol_phase_at(species: u32, cos_theta: f32) -> vec4<f32> {
    if (hp.mie_phase_mode == ATM_MIE_PHASE_MODE_CS) {
        let cs = cornette_shanks_phase(ATM_CS_G, -cos_theta);
        return vec4<f32>(cs);
    }
    let mu = -cos_theta;
    return textureSampleLevel(
        aerosol_phase_lut,
        lut_sampler,
        vec2<f32>(phase_lut_uv(mu), 0.5),
        i32(species),
        0.0,
    );
}

const HILLAIRE_INSCATTER_STEPS: u32 = 32u;

struct HillaireInScatterResult {
    radiance: vec4<f32>,
    transmittance: vec4<f32>,
}

fn compute_inscattering(
    transmittance_lut: texture_2d<f32>,
    samp: sampler,
    ray_origin_km: vec3<f32>,
    ray_dir: vec3<f32>,
    t_max_km: f32,
) -> HillaireInScatterResult {
    let cos_theta = dot(-ray_dir, hp.sun_dir);
    let molecular_phase = molecular_phase_function(cos_theta);

    var phase_per_species: array<vec4<f32>, 3>;
    phase_per_species[0] = aerosol_phase_at(0u, cos_theta);
    phase_per_species[1] = aerosol_phase_at(1u, cos_theta);
    phase_per_species[2] = aerosol_phase_at(2u, cos_theta);

    let inv_n = 1.0 / f32(HILLAIRE_INSCATTER_STEPS);
    let ground_up_transmittance = transmittance_from_lut(transmittance_lut, samp, 1.0, 0.0);

    var radiance = vec4<f32>(0.0);
    var transmittance = vec4<f32>(1.0);

    for (var i: u32 = 0u; i < HILLAIRE_INSCATTER_STEPS; i = i + 1u) {
        let u_mid = (f32(i) + 0.5) * inv_n;
        let t = u_mid * u_mid * t_max_km;
        let dt = 2.0 * u_mid * inv_n * t_max_km;
        let x_t = ray_origin_km + ray_dir * t;

        let dist_to_center = length(x_t);
        let zenith_dir = x_t / dist_to_center;
        let altitude = max(dist_to_center - hp.earth_radius_km, 0.0);
        let normalized_alt = altitude / hp.atmosphere_thickness_km;
        let sample_cos_theta = dot(zenith_dir, hp.sun_dir);

        let t_to_sun = transmittance_from_lut(
            transmittance_lut,
            samp,
            sample_cos_theta,
            normalized_alt,
        );
        let ms = get_multiple_scattering(
            transmittance_lut,
            samp,
            sample_cos_theta,
            normalized_alt,
            dist_to_center,
            ground_up_transmittance,
        );

        let earth_offset = zenith_dir * 0.01;
        let shadow_origin = atm_point_from_local_pos_km(x_t - earth_offset);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, hp.sun_dir)).hits_ground,
        );
        let t_to_sun_eff = t_to_sun * earth_shadow;

        var aerosol_inscatter = vec4<f32>(0.0);
        var aerosol_ext = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
            let c = get_species_coeffs(hp, k, altitude);
            aerosol_inscatter += c.scattering * (phase_per_species[k] * t_to_sun_eff + ms);
            aerosol_ext += c.scattering + c.absorption;
        }

        let molecular_scattering = get_molecular_scattering_coefficient(hp, altitude);
        let molecular_absorption = get_molecular_absorption_coefficient(hp, altitude);
        let extinction = aerosol_ext + molecular_scattering + molecular_absorption;

        let source = hp.sun_spectral_irradiance * (
            molecular_scattering * (molecular_phase * t_to_sun_eff + ms)
            + aerosol_inscatter
        );

        let step_t = exp(-dt * extinction);
        let safe_ext = max(extinction, vec4<f32>(1.0e-7));
        let source_int = (source - source * step_t) / safe_ext;

        radiance += transmittance * source_int;
        transmittance *= step_t;
    }

    return HillaireInScatterResult(radiance, transmittance);
}
