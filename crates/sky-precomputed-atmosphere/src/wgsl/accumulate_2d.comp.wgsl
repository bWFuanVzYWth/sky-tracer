@group(0) @binding(0) var current_lut: texture_2d<f32>;
@group(0) @binding(1) var delta_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var out_lut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_lut);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let value =
        textureSampleLevel(current_lut, lut_sampler, uv, 0.0)
        + textureSampleLevel(delta_lut, lut_sampler, uv, 0.0);
    textureStore(out_lut, vec2<i32>(gid.xy), max(value, vec4<f32>(0.0)));
}
