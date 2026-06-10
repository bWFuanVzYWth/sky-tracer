@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var density_lut: texture_3d<f32>;
@group(0) @binding(3) var scattering_lut: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
@group(0) @binding(5) var delta_scattering_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var scattering_accum_out: texture_storage_3d<rgba16float, write>;

const BRUNETON_MULTIPLE_SCATTERING_STEPS: u32 = 48u;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(delta_scattering_out);
    if (gid.x >= dims_u.x || gid.y >= dims_u.y || gid.z >= dims_u.z) {
        return;
    }
    let texel = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));

    let params = bruneton_scattering_params_from_texel(gid, dims_u);
    let origin = vec3<f32>(0.0, params.r, 0.0);
    let segment = atm_ray_segment(atm_ray_from_point(atm_point_from_local_pos_km(origin), params.ray_dir));
    if (segment.t_max_km <= 0.0) {
        let accum = textureLoad(scattering_lut, texel, 0);
        textureStore(delta_scattering_out, texel, vec4<f32>(0.0));
        textureStore(scattering_accum_out, texel, accum);
        return;
    }

    let dt = segment.t_max_km / f32(BRUNETON_MULTIPLE_SCATTERING_STEPS);
    var transmittance_to_sample = vec4<f32>(1.0);
    var delta = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < BRUNETON_MULTIPLE_SCATTERING_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let pos = origin + params.ray_dir * min(t, segment.t_max_km);
        let r = length(pos);
        let up = pos / max(r, 1.0e-6);
        let altitude = max(r - hp.earth_radius_km, 0.0);
        let mu = dot(up, params.ray_dir);
        let mu_s = dot(up, params.sun_dir);
        let nu = dot(params.ray_dir, params.sun_dir);

        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        let step_t = exp(-dt * coeffs.extinction);
        let safe_ext = max(coeffs.extinction, vec4<f32>(1.0e-7));
        let segment_weight = (vec4<f32>(1.0) - step_t) / safe_ext;
        let source = bruneton_scattering_from_lut(density_lut, lut_sampler, r, mu, mu_s, nu);

        delta += transmittance_to_sample * source * segment_weight;
        transmittance_to_sample *= step_t;
    }

    let rayleigh_phase = max(molecular_phase_function(params.nu), 1.0e-6);
    let reduced_delta = delta / rayleigh_phase;
    let accum = textureLoad(scattering_lut, texel, 0) + reduced_delta;
    textureStore(delta_scattering_out, texel, vec4<f32>(max(reduced_delta.rgb, vec3<f32>(0.0)), max(reduced_delta.a, 0.0)));
    textureStore(scattering_accum_out, texel, vec4<f32>(max(accum.rgb, vec3<f32>(0.0)), max(accum.a, 0.0)));
}
