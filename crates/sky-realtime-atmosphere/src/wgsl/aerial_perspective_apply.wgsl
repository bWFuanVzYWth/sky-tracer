struct RuntimeView {
    // AP compose 只需要从当前像素恢复相机相对位置和距离，不需要绝对世界位置。
    relative_world_from_clip: mat4x4<f32>,
    world_position: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: RuntimeView;
@group(1) @binding(0) var source_color: texture_2d<f32>;
@group(1) @binding(1) var source_depth: texture_depth_2d;
@group(1) @binding(2) var ap_inscatter_lut: texture_3d<f32>;
@group(1) @binding(3) var ap_transmittance_lut: texture_3d<f32>;
@group(1) @binding(4) var lut_sampler: sampler;

const AP_SLICE_COUNT: f32 = 32.0;
const AP_KM_PER_SLICE: f32 = 4.0;
const AP_REC2020_WHITE_FROM_FLAT_SPECTRUM: vec3<f32> =
    vec3<f32>(128.416, 108.538, 131.305);

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    let p = vec2<f32>(u * 2.0 - 1.0, 1.0 - v * 2.0);
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    let m = mat4x3<f32>(
        vec3<f32>(83.460, 1.554, -0.043),
        vec3<f32>(49.968, 86.062, -2.182),
        vec3<f32>(-11.823, 29.205, 29.153),
        vec3<f32>(6.811, -8.283, 104.377),
    );
    return m * l;
}

fn rec2020_transmittance_from_spectral(t: vec4<f32>) -> vec3<f32> {
    let rgb = linear_rec2020_from_spectral(t);
    return clamp(rgb / AP_REC2020_WHITE_FROM_FLAT_SPECTRUM, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn clean_inscatter(v: vec4<f32>) -> vec3<f32> {
    return max(vec3<f32>(
        select(0.0, v.r, v.r == v.r),
        select(0.0, v.g, v.g == v.g),
        select(0.0, v.b, v.b == v.b),
    ), vec3<f32>(0.0));
}

fn clean_transmittance(v: vec4<f32>) -> vec4<f32> {
    return clamp(vec4<f32>(
        select(1.0, v.r, v.r == v.r),
        select(1.0, v.g, v.g == v.g),
        select(1.0, v.b, v.b == v.b),
        select(1.0, v.a, v.a == v.a),
    ), vec4<f32>(0.0), vec4<f32>(1.0));
}

@fragment
fn fragment(vertex_out: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(source_color);
    let pixel = vec2<i32>(vertex_out.position.xy);
    let color = textureLoad(source_color, pixel, 0);
    let depth = textureLoad(source_depth, pixel, 0);
    if (depth <= 0.0) {
        return color;
    }

    let uv = vec2<f32>(pixel) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // 从 depth 反投影到相机相对空间，距离直接取 length，避免 world_pos - camera_pos。
    let relative_world_h = view.relative_world_from_clip * vec4<f32>(ndc, depth, 1.0);
    let safe_w = sign(relative_world_h.w) * max(abs(relative_world_h.w), 1.0e-6);
    let relative_world_pos = relative_world_h.xyz / safe_w;
    let distance_km = length(relative_world_pos) * 0.001;

    let slice_value = distance_km / AP_KM_PER_SLICE;
    var weight = 1.0;
    var s = slice_value;
    if (s < 0.5) {
        weight = clamp(s * 2.0, 0.0, 1.0);
        s = 0.5;
    }
    let w_coord = clamp(sqrt(s / AP_SLICE_COUNT), 0.0, 1.0);
    let sample_pos = vec3<f32>(uv, w_coord);
    let inscatter = clean_inscatter(textureSampleLevel(ap_inscatter_lut, lut_sampler, sample_pos, 0.0)) * weight;
    let trans_spectral = clean_transmittance(textureSampleLevel(ap_transmittance_lut, lut_sampler, sample_pos, 0.0));
    let trans_sample = rec2020_transmittance_from_spectral(trans_spectral);
    let transmittance = mix(vec3<f32>(1.0), trans_sample, weight);

    return vec4<f32>(color.rgb * transmittance + inscatter, 1.0);
}
