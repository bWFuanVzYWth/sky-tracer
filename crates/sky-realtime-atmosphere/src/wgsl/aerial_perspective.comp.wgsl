struct ApCamera {
    // AP LUT 射线方向来自相机相对反投影，避免绝对平移进入 normalize。
    relative_world_from_clip: mat4x4<f32>,
    world_position_m: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var<uniform> ap_camera: ApCamera;
@group(0) @binding(4) var ap_inscatter_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var ap_transmittance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var aerosol_phase_lut: texture_2d_array<f32>;

const AP_KM_PER_SLICE: f32 = 4.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ap_inscatter_out);
    if (id.x >= dims.x || id.y >= dims.y || id.z >= dims.z) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dims.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // clip z=0 表示远处方向；相机相对矩阵让 direction 不受世界原点大小影响。
    let relative_world_h = ap_camera.relative_world_from_clip * vec4<f32>(ndc, 0.0, 1.0);
    let world_dir = normalize(relative_world_h.xyz);

    let slice = (f32(id.z) + 0.5) / 32.0;
    let distance_km = slice * slice * 32.0 * AP_KM_PER_SLICE;
    let origin = vec3<f32>(0.0, hp.eye_distance_to_earth_center_km, 0.0);
    let result = compute_inscattering(transmittance_lut, lut_sampler, origin, world_dir, distance_km);
    let inscatter_rgb = linear_rec2020_from_spectral(result.radiance);

    textureStore(ap_inscatter_out, vec3<i32>(id), vec4<f32>(inscatter_rgb, 1.0));
    textureStore(ap_transmittance_out, vec3<i32>(id), clamp(result.transmittance, vec4<f32>(0.0), vec4<f32>(1.0)));
}
