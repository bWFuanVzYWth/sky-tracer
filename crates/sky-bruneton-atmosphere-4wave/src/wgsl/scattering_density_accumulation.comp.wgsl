struct BrunetonOrder {
    order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> bo: BrunetonOrder;
@group(0) @binding(1) var density_lut: texture_3d<f32>;
@group(0) @binding(2) var density_accum_lut: texture_3d<f32>;
@group(0) @binding(3) var density_accum_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(density_accum_out);
    if (gid.x >= dims_u.x || gid.y >= dims_u.y || gid.z >= dims_u.z) {
        return;
    }

    let texel = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    let current = textureLoad(density_lut, texel, 0);
    var previous = vec4<f32>(0.0);
    if (bo.order > 2u) {
        previous = textureLoad(density_accum_lut, texel, 0);
    }
    let accumulated = current + previous;
    textureStore(
        density_accum_out,
        texel,
        vec4<f32>(max(accumulated.rgb, vec3<f32>(0.0)), max(accumulated.a, 0.0)),
    );
}
