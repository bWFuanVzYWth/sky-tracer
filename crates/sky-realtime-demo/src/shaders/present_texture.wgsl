@group(0) @binding(0)
var source_texture: texture_2d<f32>;

struct PresentParams {
    exposure_mode_ref_diff: vec4<f32>,
    view_yaw_pitch_fov_aspect: vec4<f32>,
};

@group(0) @binding(1)
var<uniform> present: PresentParams;

@group(0) @binding(2)
var reference_texture: texture_2d<f32>;

@group(0) @binding(3)
var reference_sampler: sampler;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const DEG_TO_RAD: f32 = 0.017453292519943295;
const SQRT3: f32 = 1.7320508075688772;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let u = f32((vertex_index << 1u) & 2u);
    let v = f32(vertex_index & 2u);
    let p = vec2<f32>(u * 2.0 - 1.0, 1.0 - v * 2.0);

    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(u, v);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(source_texture);
    let pixel = clamp(
        vec2<i32>(in.position.xy),
        vec2<i32>(0, 0),
        vec2<i32>(dims) - vec2<i32>(1, 1),
    );
    let realtime_scene_rec2020 = max(textureLoad(source_texture, pixel, 0).rgb, vec3<f32>(0.0));
    let realtime = display_linear_srgb_from_scene_rec2020(realtime_scene_rec2020);
    let mode = present.exposure_mode_ref_diff.y;
    let has_reference = present.exposure_mode_ref_diff.z > 0.5;
    let diff_scale = present.exposure_mode_ref_diff.w;
    if (!has_reference || mode < 0.5) {
        return vec4<f32>(realtime, 1.0);
    }

    let framebuffer_uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let view_uv = vec2<f32>(framebuffer_uv.x, 1.0 - framebuffer_uv.y);
    let ray = view_ray_from_uv(view_uv);
    let reference_scene_srgb = max(
        textureSampleLevel(reference_texture, reference_sampler, equirect_uv(ray), 0.0).rgb,
        vec3<f32>(0.0),
    );
    let reference = display_linear_srgb_from_scene_rec2020(srgb_to_rec2020(reference_scene_srgb));

    if (mode < 1.5) {
        return vec4<f32>(reference, 1.0);
    }
    let delta = realtime - reference;
    if (mode < 2.5) {
        return vec4<f32>(min(abs(delta) * diff_scale, vec3<f32>(1.0)), 1.0);
    }
    return vec4<f32>(clamp(vec3<f32>(0.5) + delta * diff_scale * 0.5, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

fn display_linear_srgb_from_scene_rec2020(scene_rec2020: vec3<f32>) -> vec3<f32> {
    let scene = max(scene_rec2020 * present.exposure_mode_ref_diff.x, vec3<f32>(0.0));
    let display_rec2020 = compact_opendrt_tonescale(scene);
    let display_srgb = rec2020_to_srgb(display_rec2020);
    return clamp(display_srgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn compact_opendrt_tonescale(rgb: vec3<f32>) -> vec3<f32> {
    let norm = length(rgb) / SQRT3;
    if (norm <= 1.0e-8) {
        return vec3<f32>(0.0);
    }

    let tone = compact_opendrt_tonescale_scalar(norm);
    return max(rgb / norm * tone, vec3<f32>(0.0));
}

fn compact_opendrt_tonescale_scalar(x: f32) -> f32 {
    let tn_con = 1.66;
    let tn_sh = 0.5;
    let tn_toe = 0.003;
    let tn_off = 0.005;
    let tn_lp = 100.0;
    let tn_lg = 10.0;
    let tn_su = 2.0;

    let ts_x1 = pow(2.0, 6.0 * tn_sh + 4.0);
    let ts_y1 = tn_lp / 100.0;
    let ts_x0 = 0.18 + tn_off;
    let ts_y0 = tn_lg / 100.0;
    let ts_s0 = compress_toe_quadratic_inverse(ts_y0, tn_toe);
    let ts_p = tn_con / (1.0 + tn_su * 0.05);
    let ts_s10 = ts_x0 * (pow(ts_s0, -1.0 / tn_con) - 1.0);
    let ts_m1 = ts_y1 / pow(ts_x1 / (ts_x1 + ts_s10), tn_con);
    let ts_m2 = compress_toe_quadratic_inverse(ts_m1, tn_toe);
    let ts_s = ts_x0 * (pow(ts_s0 / ts_m2, -1.0 / tn_con) - 1.0);

    var y = compress_hyperbolic_power(max(x + tn_off, 0.0), ts_s, ts_p);
    y *= ts_m2;
    y = compress_toe_quadratic(y, tn_toe);
    return clamp(y, 0.0, 1.0);
}

fn compress_hyperbolic_power(x: f32, s: f32, p: f32) -> f32 {
    return pow(max(x, 0.0) / max(x + s, 1.0e-8), p);
}

fn compress_toe_quadratic(x: f32, toe: f32) -> f32 {
    if (toe <= 0.0) {
        return x;
    }
    return x * x / max(x + toe, 1.0e-8);
}

fn compress_toe_quadratic_inverse(x: f32, toe: f32) -> f32 {
    if (toe <= 0.0) {
        return x;
    }
    return 0.5 * (x + sqrt(max(x * (4.0 * toe + x), 0.0)));
}

fn srgb_to_rec2020(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.6274040 * rgb.r + 0.3292820 * rgb.g + 0.0433136 * rgb.b,
        0.0690970 * rgb.r + 0.9195400 * rgb.g + 0.0113612 * rgb.b,
        0.0163916 * rgb.r + 0.0880132 * rgb.g + 0.8955952 * rgb.b,
    );
}

fn rec2020_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        1.6604910 * rgb.r - 0.5876411 * rgb.g - 0.0728499 * rgb.b,
        -0.1245505 * rgb.r + 1.1328999 * rgb.g - 0.0083494 * rgb.b,
        -0.0181508 * rgb.r - 0.1005789 * rgb.g + 1.1187297 * rgb.b,
    );
}

fn view_ray_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let yaw = present.view_yaw_pitch_fov_aspect.x * DEG_TO_RAD;
    let pitch = present.view_yaw_pitch_fov_aspect.y * DEG_TO_RAD;
    let fov_y = present.view_yaw_pitch_fov_aspect.z * DEG_TO_RAD;
    let aspect = present.view_yaw_pitch_fov_aspect.w;

    let local_xy = (uv * 2.0 - vec2<f32>(1.0, 1.0)) * vec2<f32>(aspect, 1.0) * tan(0.5 * fov_y);
    let forward = normalize(vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)));
    let right = normalize(vec3<f32>(cos(yaw), 0.0, -sin(yaw)));
    let up = normalize(cross(forward, right));
    return normalize(forward + local_xy.x * right + local_xy.y * up);
}

fn equirect_uv(ray: vec3<f32>) -> vec2<f32> {
    let dir = normalize(ray);
    let u = fract(atan2(dir.x, dir.z) / TAU + 0.5);
    let v = clamp(0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI, 0.0, 1.0);
    return vec2<f32>(u, v);
}
