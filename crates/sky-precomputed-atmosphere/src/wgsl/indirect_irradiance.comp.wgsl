@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var scattering_density: texture_3d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var irradiance_out: texture_storage_2d<rgba16float, write>;

fn indirect_irradiance_sample(r: f32, mu_s: f32) -> vec4<f32> {
    let sample_count: u32 = 32u;
    let omega_s = vec3<f32>(safe_sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
    var irradiance = vec4<f32>(0.0);
    for (var j: u32 = 0u; j < sample_count / 2u; j = j + 1u) {
        let theta = (f32(j) + 0.5) * (0.5 * PI) / f32(sample_count / 2u);
        let cos_theta = cos(theta);
        let sin_theta = sin(theta);
        let theta_weight = (0.5 * PI) / f32(sample_count / 2u);
        for (var i: u32 = 0u; i < sample_count; i = i + 1u) {
            let phi = (f32(i) + 0.5) * (2.0 * PI) / f32(sample_count);
            let omega_i = vec3<f32>(
                cos(phi) * sin_theta,
                sin(phi) * sin_theta,
                cos_theta,
            );
            let domega = theta_weight * (2.0 * PI / f32(sample_count)) * sin_theta;
            let nu = clamp_cosine(dot(omega_s, omega_i));
            let radiance = sample_scattering(scattering_density, lut_sampler, r, cos_theta, mu_s, nu, false);
            irradiance += radiance * cos_theta * domega;
        }
    }
    return irradiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(irradiance_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let r_mu_s = r_mu_s_from_irradiance_uv(uv);
    let irradiance = indirect_irradiance_sample(r_mu_s.x, r_mu_s.y);
    textureStore(irradiance_out, vec2<i32>(gid.xy), max(irradiance, vec4<f32>(0.0)));
}
