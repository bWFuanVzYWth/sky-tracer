@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var irradiance_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(irradiance_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let r_mu_s = r_mu_s_from_irradiance_uv(uv);
    let r = r_mu_s.x;
    let mu_s = r_mu_s.y;
    let transmittance = get_transmittance_to_sun(transmittance_lut, lut_sampler, r, mu_s);
    let irradiance = params.sun_spectral_irradiance * transmittance * max(mu_s, 0.0);
    textureStore(irradiance_out, vec2<i32>(gid.xy), irradiance);
}
