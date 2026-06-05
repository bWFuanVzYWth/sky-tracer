@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var transmittance_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(transmittance_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let r_mu = r_mu_from_transmittance_uv(uv);
    let transmittance = compute_transmittance_to_top(r_mu.x, r_mu.y);
    textureStore(transmittance_out, vec2<i32>(gid.xy), vec4<f32>(transmittance.rgb, transmittance.a));
}
