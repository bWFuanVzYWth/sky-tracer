@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var scattering_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var single_mie_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var single_rayleigh_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var aerosol_phase_lut: texture_2d_array<f32>;

const BRUNETON_SINGLE_SCATTERING_STEPS: u32 = 48u;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(scattering_out);
    if (gid.x >= dims_u.x || gid.y >= dims_u.y || gid.z >= dims_u.z) {
        return;
    }
    let texel = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));

    let params = bruneton_scattering_params_from_texel(gid, dims_u);
    let origin = vec3<f32>(0.0, params.r, 0.0);
    let segment = atmosphere_ray_limit(origin, params.ray_dir);
    if (segment.t_max_km <= 0.0) {
        textureStore(scattering_out, texel, vec4<f32>(0.0));
        textureStore(single_mie_out, texel, vec4<f32>(0.0));
        textureStore(single_rayleigh_out, texel, vec4<f32>(0.0));
        return;
    }

    let dt = segment.t_max_km / f32(BRUNETON_SINGLE_SCATTERING_STEPS);
    var transmittance_to_sample = vec4<f32>(1.0);
    var rayleigh = vec4<f32>(0.0);
    var mie = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < BRUNETON_SINGLE_SCATTERING_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let pos = origin + params.ray_dir * min(t, segment.t_max_km);
        let r = length(pos);
        let up = pos / max(r, 1.0e-6);
        let altitude = max(r - hp.earth_radius_km, 0.0);
        let normalized_alt = clamp(altitude / hp.atmosphere_thickness_km, 0.0, 1.0);
        let mu_s = dot(up, params.sun_dir);

        let shadow_origin = atm_point_from_local_pos_km(pos - up * ATM_PLANET_RADIUS_OFFSET_KM);
        let earth_shadow = select(
            1.0,
            0.0,
            atm_ray_segment(atm_ray_from_point(shadow_origin, params.sun_dir)).hits_ground,
        );
        let transmittance_to_sun = transmittance_from_lut(transmittance_lut, lut_sampler, mu_s, normalized_alt) * earth_shadow;

        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        let step_t = exp(-dt * coeffs.extinction);
        let safe_ext = max(coeffs.extinction, vec4<f32>(1.0e-7));
        let segment_weight = (vec4<f32>(1.0) - step_t) / safe_ext;

        let molecular_source = hp.sun_spectral_irradiance
            * transmittance_to_sun
            * coeffs.molecular_scattering;
        let mie_source = hp.sun_spectral_irradiance
            * transmittance_to_sun
            * coeffs.aerosol_scattering;

        rayleigh += transmittance_to_sample * molecular_source * segment_weight;
        mie += transmittance_to_sample * mie_source * segment_weight;
        transmittance_to_sample *= step_t;
    }

    let rayleigh_out = vec4<f32>(max(rayleigh.rgb, vec3<f32>(0.0)), max(rayleigh.a, 0.0));
    textureStore(scattering_out, texel, rayleigh_out);
    textureStore(single_rayleigh_out, texel, rayleigh_out);
    textureStore(single_mie_out, texel, vec4<f32>(max(mie.rgb, vec3<f32>(0.0)), max(mie.a, 0.0)));
}
