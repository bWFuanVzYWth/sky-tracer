@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var irradiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var delta_irradiance_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(irradiance_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / dims;
    let r_mu_s = bruneton_ground_irradiance_params_from_uv(uv, dims);
    let r = r_mu_s.x;
    let mu_s = r_mu_s.y;
    let normalized_alt = clamp((r - hp.earth_radius_km) / hp.atmosphere_thickness_km, 0.0, 1.0);

    var irradiance = vec4<f32>(0.0);
    let sun_cos = sun_disc_average_cosine_factor(mu_s);
    if (sun_cos > 0.0) {
        irradiance = hp.sun_spectral_irradiance
            * transmittance_from_lut(transmittance_lut, lut_sampler, max(mu_s, 0.0), normalized_alt)
            * sun_cos;
    }

    let delta = vec4<f32>(max(irradiance.rgb, vec3<f32>(0.0)), max(irradiance.a, 0.0));
    textureStore(delta_irradiance_out, vec2<i32>(gid.xy), delta);
    textureStore(irradiance_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
}
