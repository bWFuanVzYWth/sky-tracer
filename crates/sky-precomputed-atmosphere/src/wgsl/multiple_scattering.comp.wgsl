@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var scattering_density: texture_3d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var scattering_out: texture_storage_3d<rgba16float, write>;

fn multiple_scattering_integrand(r: f32, mu: f32, mu_s: f32, nu: f32, d: f32, hits_ground: bool) -> vec4<f32> {
    let r_i = clamp_radius(safe_sqrt(d * d + 2.0 * r * mu * d + r * r));
    let mu_i = clamp_cosine((r * mu + d) / max(r_i, 1.0e-6));
    let mu_s_i = clamp_cosine((r * mu_s + d * nu) / max(r_i, 1.0e-6));
    let transmittance = get_transmittance(transmittance_lut, lut_sampler, r, mu, d, hits_ground);
    let source = sample_scattering(scattering_density, lut_sampler, r_i, mu_i, mu_s_i, nu, ray_intersects_ground(r_i, mu_i));
    return transmittance * source;
}

fn compute_multiple_scattering(r: f32, mu: f32, mu_s: f32, nu: f32, hits_ground: bool) -> vec4<f32> {
    let sample_count: u32 = 50u;
    let dx = distance_to_nearest_atmosphere_boundary(r, mu, hits_ground) / f32(sample_count);
    var radiance = vec4<f32>(0.0);
    for (var i: u32 = 0u; i <= sample_count; i = i + 1u) {
        let weight = select(1.0, 0.5, i == 0u || i == sample_count);
        radiance += multiple_scattering_integrand(r, mu, mu_s, nu, f32(i) * dx, hits_ground) * weight * dx;
    }
    return radiance;
}

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(scattering_out);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let sp = scattering_params_from_frag_coord(vec3<f32>(gid) + vec3<f32>(0.5));
    let radiance = compute_multiple_scattering(sp.r, sp.mu, sp.mu_s, sp.nu, sp.hits_ground);
    textureStore(scattering_out, vec3<i32>(gid), max(radiance, vec4<f32>(0.0)));
}
