@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(0) @binding(3) var scattering_lut: texture_3d<f32>;
@group(0) @binding(4) var single_rayleigh_lut: texture_3d<f32>;
@group(0) @binding(5) var lut_sampler: sampler;
@group(0) @binding(6) var aerosol_phase_lut: texture_2d_array<f32>;
@group(0) @binding(7) var sky_view_out: texture_storage_2d<rgba16float, write>;

const BRUNETON_SKY_VIEW_TRANSMITTANCE_STEPS: u32 = 32u;
const BRUNETON_SKY_VIEW_SINGLE_SCATTERING_STEPS: u32 = 64u;

fn bruneton_view_transmittance(ray_origin_km: vec3<f32>, ray_dir: vec3<f32>, t_max_km: f32) -> vec4<f32> {
    let dt = max(t_max_km, 0.0) / f32(BRUNETON_SKY_VIEW_TRANSMITTANCE_STEPS);
    var optical_depth = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < BRUNETON_SKY_VIEW_TRANSMITTANCE_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin_km + ray_dir * min(t, t_max_km);
        let altitude = max(length(pos) - hp.earth_radius_km, 0.0);
        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        optical_depth += coeffs.extinction * dt;
    }
    return exp(-optical_depth);
}

fn bruneton_runtime_single_scattering(
    ray_origin_km: vec3<f32>,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    t_max_km: f32,
    view_sun_nu: f32,
) -> vec4<f32> {
    let dt = max(t_max_km, 0.0) / f32(BRUNETON_SKY_VIEW_SINGLE_SCATTERING_STEPS);
    var transmittance_to_sample = vec4<f32>(1.0);
    var radiance = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < BRUNETON_SKY_VIEW_SINGLE_SCATTERING_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin_km + ray_dir * min(t, t_max_km);
        let r = length(pos);
        let up = pos / max(r, 1.0e-6);
        let altitude = max(r - hp.earth_radius_km, 0.0);
        let normalized_alt = clamp(altitude / hp.atmosphere_thickness_km, 0.0, 1.0);
        let mu_s = dot(up, sun_dir);

        let shadow_origin = atm_point_from_local_pos_km(pos - up * ATM_PLANET_RADIUS_OFFSET_KM);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, sun_dir)).hits_ground,
        );
        let transmittance_to_sun = transmittance_from_lut(
            transmittance_lut,
            lut_sampler,
            mu_s,
            normalized_alt,
        ) * earth_shadow;

        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        var phase_times_scattering = coeffs.molecular_scattering
            * molecular_phase_function(view_sun_nu);
        for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
            let c = get_species_coeffs(hp, k, altitude);
            phase_times_scattering += c.scattering * bruneton_aerosol_phase_at(k, view_sun_nu);
        }

        let step_t = exp(-dt * coeffs.extinction);
        let safe_ext = max(coeffs.extinction, vec4<f32>(1.0e-7));
        let segment_weight = (vec4<f32>(1.0) - step_t) / safe_ext;
        radiance += transmittance_to_sample
            * hp.sun_spectral_irradiance
            * transmittance_to_sun
            * phase_times_scattering
            * segment_weight;
        transmittance_to_sample *= step_t;
    }

    return radiance;
}

fn bruneton_sky_spectral_radiance(ray_origin_km_in: vec3<f32>, ray_dir_in: vec3<f32>) -> vec4<f32> {
    let ray_dir = normalize(ray_dir_in);
    var ray_origin_km = move_to_top_atmosphere(ray_origin_km_in, ray_dir);
    let segment = atm_ray_segment(atm_ray_from_point(atm_point_from_local_pos_km(ray_origin_km), ray_dir));
    if (segment.t_max_km <= 0.0) {
        return vec4<f32>(0.0);
    }

    let r = length(ray_origin_km);
    let up = ray_origin_km / max(r, 1.0e-6);
    let mu = dot(up, ray_dir);
    let mu_s = dot(up, hp.sun_dir);
    let nu = dot(ray_dir, hp.sun_dir);
    let accumulated_reduced = bruneton_scattering_from_lut(scattering_lut, lut_sampler, r, mu, mu_s, nu);
    let single_rayleigh_reduced = bruneton_scattering_from_lut(
        single_rayleigh_lut,
        lut_sampler,
        r,
        mu,
        mu_s,
        nu,
    );
    let multiple_scattering = max(accumulated_reduced - single_rayleigh_reduced, vec4<f32>(0.0))
        * molecular_phase_function(nu);
    let single_scattering = bruneton_runtime_single_scattering(
        ray_origin_km,
        ray_dir,
        hp.sun_dir,
        segment.t_max_km,
        nu,
    );
    var radiance = multiple_scattering + single_scattering;

    if (segment.hits_ground) {
        let ground_pos = ray_origin_km + ray_dir * segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        var ground_irradiance = bruneton_ground_irradiance_from_lut(
            irradiance_lut,
            lut_sampler,
            hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM,
            dot(ground_normal, hp.sun_dir),
        );
        let ground_sun_mu = dot(ground_normal, hp.sun_dir);
        let ground_sun_cos = sun_disc_average_cosine_factor(ground_sun_mu);
        if (ground_sun_cos > 0.0) {
            let ground_altitude = max(length(ground_pos) - hp.earth_radius_km, 0.0);
            let ground_normalized_alt = clamp(ground_altitude / hp.atmosphere_thickness_km, 0.0, 1.0);
            ground_irradiance += hp.sun_spectral_irradiance
                * transmittance_from_lut(transmittance_lut, lut_sampler, max(ground_sun_mu, 0.0), ground_normalized_alt)
                * ground_sun_cos;
        }
        let view_t = bruneton_view_transmittance(ray_origin_km, ray_dir, segment.t_ground_km);
        radiance += ground_irradiance * hp.ground_albedo_spectral * ATM_INV_PI * view_t;
    }

    return radiance;
}

fn bruneton_phase_lut_uv(cos_theta: f32) -> f32 {
    return pow(max(0.5 - 0.5 * clamp(cos_theta, -1.0, 1.0), 0.0), 1.0 / 3.0);
}

fn bruneton_aerosol_phase_at(species: u32, view_sun_nu: f32) -> vec4<f32> {
    if (hp.mie_phase_mode == ATM_MIE_PHASE_MODE_CS) {
        return vec4<f32>(cornette_shanks_phase(ATM_CS_G, view_sun_nu));
    }
    return textureSampleLevel(
        aerosol_phase_lut,
        lut_sampler,
        vec2<f32>(bruneton_phase_lut_uv(view_sun_nu), 0.5),
        i32(species),
        0.0,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(sky_view_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / dims;
    let params = sky_view_uv_to_params(uv, dims);
    let ray_dir = sky_view_dir_from_params(params);
    let origin = vec3<f32>(0.0, sky_view_height_km(), 0.0);
    let spectral = bruneton_sky_spectral_radiance(origin, ray_dir);
    let rec2020 = max(white_balanced_linear_rec2020_from_spectral(spectral), vec3<f32>(0.0));
    textureStore(sky_view_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rec2020, 1.0));
}
