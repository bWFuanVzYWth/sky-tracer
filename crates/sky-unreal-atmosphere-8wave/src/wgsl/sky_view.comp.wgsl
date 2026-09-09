@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var<uniform> hp_high: HillaireParams;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var transmittance_lut_high: texture_2d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
@group(0) @binding(5) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(6) var multi_scattering_lut_high: texture_2d<f32>;
@group(0) @binding(7) var sky_view_rec2020_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var aerosol_phase_lut: texture_2d_array<f32>;
@group(0) @binding(9) var aerosol_phase_lut_high: texture_2d_array<f32>;

const SKY_VIEW_STEPS: u32 = 128u;

struct ScatterPair {
    radiance_low: vec4<f32>,
    radiance_high: vec4<f32>,
    transmittance_low: vec4<f32>,
    transmittance_high: vec4<f32>,
}

fn integrate_scattered_luminance_pair(
    origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
    segment: AtmRaySegment,
) -> ScatterPair {
    let ray_dir = normalize(ray_dir_in);
    let sun_dir = normalize(hp.sun_dir);
    let cos_theta = dot(-ray_dir, sun_dir);
    let molecular_phase = molecular_phase_function(cos_theta);

    var radiance_low = vec4<f32>(0.0);
    var radiance_high = vec4<f32>(0.0);
    var transmittance_low = vec4<f32>(1.0);
    var transmittance_high = vec4<f32>(1.0);

    let n = max(SKY_VIEW_STEPS, 1u);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let u0 = f32(i) / f32(n);
        let u1 = f32(i + 1u) / f32(n);
        let t0 = u0 * u0 * segment.t_max_km;
        let t1 = u1 * u1 * segment.t_max_km;
        let t = mix(t0, t1, 0.3);
        let dt = max(t1 - t0, 0.0);
        let x_t = origin + ray_dir * t;

        let dist_to_center = length(x_t);
        let zenith_dir = x_t / max(dist_to_center, 1.0e-6);
        let altitude = max(dist_to_center - hp.earth_radius_km, 0.0);
        let normalized_alt = altitude / hp.atmosphere_thickness_km;
        let sample_sun_mu = dot(zenith_dir, sun_dir);
        let shadow_origin = atm_point_from_local_pos_km(x_t - zenith_dir * ATM_PLANET_RADIUS_OFFSET_KM);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, sun_dir)).hits_ground,
        );

        let coeffs_low = get_atmosphere_collision_coefficients(hp, altitude);
        let coeffs_high = get_atmosphere_collision_coefficients(hp_high, altitude);
        let aerosol_low = species_phase_times_scattering_p(
            hp,
            aerosol_phase_lut,
            lut_sampler,
            altitude,
            cos_theta,
            true,
        );
        let aerosol_high = species_phase_times_scattering_p(
            hp_high,
            aerosol_phase_lut_high,
            lut_sampler,
            altitude,
            cos_theta,
            true,
        );

        let t_to_sun_low = transmittance_from_lut(transmittance_lut, lut_sampler, sample_sun_mu, normalized_alt) * earth_shadow;
        let t_to_sun_high = transmittance_from_lut_p(
            hp_high,
            transmittance_lut_high,
            lut_sampler,
            sample_sun_mu,
            normalized_alt,
        ) * earth_shadow;
        let multi_low = multi_scattering_from_lut(multi_scattering_lut, lut_sampler, sample_sun_mu, normalized_alt);
        let multi_high = multi_scattering_from_lut_p(
            hp_high,
            multi_scattering_lut_high,
            lut_sampler,
            sample_sun_mu,
            normalized_alt,
        );

        let direct_low = t_to_sun_low * (coeffs_low.molecular_scattering * molecular_phase + aerosol_low);
        let direct_high = t_to_sun_high * (coeffs_high.molecular_scattering * molecular_phase + aerosol_high);
        let source_low = hp.sun_spectral_irradiance
            * (direct_low + multi_low * (coeffs_low.aerosol_scattering + coeffs_low.molecular_scattering));
        let source_high = hp_high.sun_spectral_irradiance
            * (direct_high + multi_high * (coeffs_high.aerosol_scattering + coeffs_high.molecular_scattering));

        let step_low = exp(-dt * coeffs_low.extinction);
        let step_high = exp(-dt * coeffs_high.extinction);
        let safe_ext_low = max(coeffs_low.extinction, vec4<f32>(1.0e-7));
        let safe_ext_high = max(coeffs_high.extinction, vec4<f32>(1.0e-7));
        radiance_low += transmittance_low * ((source_low - source_low * step_low) / safe_ext_low);
        radiance_high += transmittance_high * ((source_high - source_high * step_high) / safe_ext_high);
        transmittance_low *= step_low;
        transmittance_high *= step_high;
    }

    return ScatterPair(radiance_low, radiance_high, transmittance_low, transmittance_high);
}

fn add_ground_radiance(pair: ScatterPair, origin: vec3<f32>, ray_dir: vec3<f32>, segment: AtmRaySegment) -> ScatterPair {
    var out = pair;
    if (segment.hits_ground) {
        let ground_pos = origin + ray_dir * segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        let ground_sun_mu = dot(ground_normal, hp.sun_dir);
        let sun_cos = max(ground_sun_mu, 0.0);
        var ground_low = vec4<f32>(0.0);
        var ground_high = vec4<f32>(0.0);
        if (sun_cos > 0.0) {
            ground_low += transmittance_from_lut(transmittance_lut, lut_sampler, sun_cos, 0.0)
                * (sun_cos * ATM_INV_PI);
            ground_high += transmittance_from_lut_p(hp_high, transmittance_lut_high, lut_sampler, sun_cos, 0.0)
                * (sun_cos * ATM_INV_PI);
        }
        ground_low += multi_scattering_from_lut(multi_scattering_lut, lut_sampler, ground_sun_mu, 0.0);
        ground_high += multi_scattering_from_lut_p(
            hp_high,
            multi_scattering_lut_high,
            lut_sampler,
            ground_sun_mu,
            0.0,
        );
        out.radiance_low += hp.sun_spectral_irradiance
            * ground_low
            * hp.ground_albedo_spectral
            * pair.transmittance_low;
        out.radiance_high += hp_high.sun_spectral_irradiance
            * ground_high
            * hp_high.ground_albedo_spectral
            * pair.transmittance_high;
    }
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(sky_view_rec2020_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / dims;
    let params = sky_view_uv_to_params(uv, dims);
    let ray_dir = sky_view_dir_from_params(params);
    var origin = vec3<f32>(0.0, sky_view_height_km(), 0.0);
    origin = move_to_top_atmosphere(origin, ray_dir);
    let segment = atmosphere_ray_limit(origin, ray_dir);

    if (segment.t_max_km < 0.0) {
        textureStore(sky_view_rec2020_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0));
        return;
    }

    let scatter = add_ground_radiance(
        integrate_scattered_luminance_pair(origin, ray_dir, segment),
        origin,
        ray_dir,
        segment,
    );
    let rec2020 = max(
        white_balanced_linear_rec2020_from_spectral8(scatter.radiance_low, scatter.radiance_high),
        vec3<f32>(0.0),
    );

    textureStore(
        sky_view_rec2020_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(rec2020, 1.0),
    );
}
