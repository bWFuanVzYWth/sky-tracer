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

struct UnrealScatterResult {
    radiance: vec4<f32>,
    transmittance: vec4<f32>,
    multi_scat_as1: vec4<f32>,
}

fn atmosphere_ray_limit(ray_origin_km: vec3<f32>, ray_dir: vec3<f32>) -> AtmRaySegment {
    return atm_ray_segment(atm_ray_from_point(atm_point_from_local_pos_km(ray_origin_km), ray_dir));
}

fn species_phase_times_scattering(
    altitude_km: f32,
    cos_theta: f32,
    use_mie_ray_phase: bool,
) -> vec4<f32> {
    var phase_times_scattering = vec4<f32>(0.0);
    if (use_mie_ray_phase) {
        for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
            let c = get_species_coeffs(hp, k, altitude_km);
            phase_times_scattering += c.scattering * aerosol_phase_at(k, cos_theta);
        }
    } else {
        for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
            let c = get_species_coeffs(hp, k, altitude_km);
            phase_times_scattering += c.scattering * ATM_PHASE_ISOTROPIC;
        }
    }
    return phase_times_scattering;
}

fn integrate_scattered_luminance_direct(
    transmittance_lut: texture_2d<f32>,
    samp: sampler,
    ray_origin_km: vec3<f32>,
    ray_dir_in: vec3<f32>,
    sun_dir_in: vec3<f32>,
    t_max_km: f32,
    sample_count: u32,
    use_mie_ray_phase: bool,
    include_ground: bool,
    illuminance_is_one: bool,
) -> UnrealScatterResult {
    let ray_dir = normalize(ray_dir_in);
    let sun_dir = normalize(sun_dir_in);
    let cos_theta = dot(-ray_dir, sun_dir);
    let molecular_phase = select(ATM_PHASE_ISOTROPIC, molecular_phase_function(cos_theta), use_mie_ray_phase);
    let global_illum = select(hp.sun_spectral_irradiance, vec4<f32>(1.0), illuminance_is_one);

    var radiance = vec4<f32>(0.0);
    var transmittance = vec4<f32>(1.0);
    var multi_scat_as1 = vec4<f32>(0.0);

    let n = max(sample_count, 1u);
    let dt = t_max_km / f32(n);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let t = (f32(i) + 0.3) * dt;
        let x_t = ray_origin_km + ray_dir * min(t, t_max_km);

        let dist_to_center = length(x_t);
        let zenith_dir = x_t / max(dist_to_center, 1.0e-6);
        let altitude = max(dist_to_center - hp.earth_radius_km, 0.0);
        let normalized_alt = altitude / hp.atmosphere_thickness_km;
        let sample_sun_mu = dot(zenith_dir, sun_dir);

        let t_to_sun = transmittance_from_lut(transmittance_lut, samp, sample_sun_mu, normalized_alt);

        let shadow_origin = atm_point_from_local_pos_km(x_t - zenith_dir * ATM_PLANET_RADIUS_OFFSET_KM);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, sun_dir)).hits_ground,
        );
        let t_to_sun_eff = t_to_sun * earth_shadow;

        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        let aerosol_phase_sca = species_phase_times_scattering(altitude, cos_theta, use_mie_ray_phase);
        let molecular_phase_sca = coeffs.molecular_scattering * molecular_phase;
        let scattering_total = coeffs.aerosol_scattering + coeffs.molecular_scattering;
        let source = global_illum * t_to_sun_eff * (molecular_phase_sca + aerosol_phase_sca);

        let step_t = exp(-dt * coeffs.extinction);
        let safe_ext = max(coeffs.extinction, vec4<f32>(1.0e-7));
        let source_int = (source - source * step_t) / safe_ext;
        radiance += transmittance * source_int;

        let ms_int = (scattering_total - scattering_total * step_t) / safe_ext;
        multi_scat_as1 += transmittance * ms_int;

        transmittance *= step_t;
    }

    let segment = atmosphere_ray_limit(ray_origin_km, ray_dir);
    if (include_ground && segment.hits_ground && abs(segment.t_ground_km - t_max_km) < max(0.01, dt * 2.0)) {
        let ground_pos = ray_origin_km + ray_dir * segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        let sun_cos = max(dot(ground_normal, sun_dir), 0.0);
        if (sun_cos > 0.0) {
            let t_to_sun = transmittance_from_lut(transmittance_lut, samp, sun_cos, 0.0);
            radiance += global_illum
                * t_to_sun
                * transmittance
                * hp.ground_albedo_spectral
                * (sun_cos * ATM_INV_PI);
        }
    }

    return UnrealScatterResult(radiance, transmittance, multi_scat_as1);
}

fn integrate_scattered_luminance_direct_with_ground_irradiance(
    transmittance_lut: texture_2d<f32>,
    ground_irradiance_lut: texture_2d<f32>,
    samp: sampler,
    ray_origin_km: vec3<f32>,
    ray_dir_in: vec3<f32>,
    sun_dir_in: vec3<f32>,
    t_max_km: f32,
    sample_count: u32,
    use_mie_ray_phase: bool,
    include_ground: bool,
    illuminance_is_one: bool,
) -> UnrealScatterResult {
    let ray_dir = normalize(ray_dir_in);
    let sun_dir = normalize(sun_dir_in);
    var result = integrate_scattered_luminance_direct(
        transmittance_lut,
        samp,
        ray_origin_km,
        ray_dir,
        sun_dir,
        t_max_km,
        sample_count,
        use_mie_ray_phase,
        false,
        illuminance_is_one,
    );

    let n = max(sample_count, 1u);
    let dt = t_max_km / f32(n);
    let segment = atmosphere_ray_limit(ray_origin_km, ray_dir);
    if (include_ground && segment.hits_ground && abs(segment.t_ground_km - t_max_km) < max(0.01, dt * 2.0)) {
        let ground_pos = ray_origin_km + ray_dir * segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        let sun_cos_raw = dot(ground_normal, sun_dir);
        let ground_irradiance_transfer = ground_irradiance_from_lut(
            ground_irradiance_lut,
            samp,
            hp.earth_radius_km,
            sun_cos_raw,
        );
        let global_illum = select(hp.sun_spectral_irradiance, vec4<f32>(1.0), illuminance_is_one);
        result.radiance += global_illum
            * ground_irradiance_transfer
            * result.transmittance
            * hp.ground_albedo_spectral
            * ATM_INV_PI;
    }

    return result;
}

fn integrate_scattered_luminance_with_ms(
    transmittance_lut: texture_2d<f32>,
    multi_scattering_lut: texture_2d<f32>,
    samp: sampler,
    ray_origin_km: vec3<f32>,
    ray_dir_in: vec3<f32>,
    sun_dir_in: vec3<f32>,
    t_max_km: f32,
    sample_count: u32,
) -> UnrealScatterResult {
    let ray_dir = normalize(ray_dir_in);
    let sun_dir = normalize(sun_dir_in);
    let cos_theta = dot(-ray_dir, sun_dir);
    let molecular_phase = molecular_phase_function(cos_theta);

    var radiance = vec4<f32>(0.0);
    var transmittance = vec4<f32>(1.0);
    var multi_scat_as1 = vec4<f32>(0.0);

    let n = max(sample_count, 1u);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let u0 = f32(i) / f32(n);
        let u1 = f32(i + 1u) / f32(n);
        let t0 = u0 * u0 * t_max_km;
        let t1 = u1 * u1 * t_max_km;
        let t = mix(t0, t1, 0.3);
        let dt = max(t1 - t0, 0.0);
        let x_t = ray_origin_km + ray_dir * t;

        let dist_to_center = length(x_t);
        let zenith_dir = x_t / max(dist_to_center, 1.0e-6);
        let altitude = max(dist_to_center - hp.earth_radius_km, 0.0);
        let normalized_alt = altitude / hp.atmosphere_thickness_km;
        let sample_sun_mu = dot(zenith_dir, sun_dir);

        let t_to_sun = transmittance_from_lut(transmittance_lut, samp, sample_sun_mu, normalized_alt);
        let multi_transfer = multi_scattering_from_lut(
            multi_scattering_lut,
            samp,
            sample_sun_mu,
            normalized_alt,
        );

        let shadow_origin = atm_point_from_local_pos_km(x_t - zenith_dir * ATM_PLANET_RADIUS_OFFSET_KM);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, sun_dir)).hits_ground,
        );
        let t_to_sun_eff = t_to_sun * earth_shadow;

        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        let aerosol_phase_sca = species_phase_times_scattering(altitude, cos_theta, true);
        let molecular_phase_sca = coeffs.molecular_scattering * molecular_phase;
        let scattering_total = coeffs.aerosol_scattering + coeffs.molecular_scattering;
        let direct_source = t_to_sun_eff * (molecular_phase_sca + aerosol_phase_sca);
        let multi_source = multi_transfer * scattering_total;
        let source = hp.sun_spectral_irradiance * (direct_source + multi_source);

        let step_t = exp(-dt * coeffs.extinction);
        let safe_ext = max(coeffs.extinction, vec4<f32>(1.0e-7));
        let source_int = (source - source * step_t) / safe_ext;
        radiance += transmittance * source_int;

        let ms_int = (scattering_total - scattering_total * step_t) / safe_ext;
        multi_scat_as1 += transmittance * ms_int;

        transmittance *= step_t;
    }

    return UnrealScatterResult(radiance, transmittance, multi_scat_as1);
}
