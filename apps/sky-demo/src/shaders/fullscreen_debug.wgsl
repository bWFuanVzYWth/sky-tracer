struct DebugUniform {
    viewport_spp_band_count: vec4<f32>,
    sun_observer_exposure_output: vec4<f32>,
    asset_dimensions_padding: vec4<f32>,
    view_yaw_pitch_fov_aspect: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params: DebugUniform;

const DEG_TO_RAD: f32 = 0.017453292519943295;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );

    var out: VertexOut;
    let position = positions[vertex_index];
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = position * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let uv = clamp(in.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let sun_elevation_deg = params.sun_observer_exposure_output.x;
    let sun_azimuth_deg = params.sun_observer_exposure_output.y;
    let sun_elevation = clamp(sun_elevation_deg / 90.0, -1.0, 1.0);
    let exposure = params.sun_observer_exposure_output.w;
    let band_count = params.viewport_spp_band_count.w;

    let ray = view_ray_from_uv(uv);
    let sky_weight = smoothstep(-0.012, 0.018, ray.y);
    let sky_t = clamp(ray.y, 0.0, 1.0);
    let ground_t = clamp(-ray.y, 0.0, 1.0);

    let band_debug = clamp(band_count / 16.0, 0.0, 1.0);
    let zenith = vec3<f32>(0.025, 0.075 + 0.035 * band_debug, 0.20 + 0.08 * max(sun_elevation, 0.0));
    let near_horizon = vec3<f32>(0.33 + 0.10 * max(sun_elevation, 0.0), 0.43, 0.58);
    let sky = mix(near_horizon, zenith, sqrt(sky_t));

    let ground_near = vec3<f32>(0.15, 0.13, 0.105);
    let ground_far = vec3<f32>(0.055, 0.060, 0.065);
    let ground = mix(ground_near, ground_far, sqrt(ground_t));

    let sun_dir = direction_from_azimuth_elevation(sun_azimuth_deg, sun_elevation_deg);
    let sun_cos = dot(ray, sun_dir);
    let sun_disk = smoothstep(cos(0.55 * DEG_TO_RAD), cos(0.20 * DEG_TO_RAD), sun_cos) * max(sun_elevation, 0.0);
    let sun_glow = exp((sun_cos - 1.0) * 22.0) * max(sun_elevation, 0.0);

    var color = mix(ground, sky, sky_weight);
    color += vec3<f32>(1.0, 0.83, 0.56) * sun_disk * 2.0;
    color += vec3<f32>(0.75, 0.48, 0.24) * sun_glow * 0.18;
    color *= exposure;

    return vec4<f32>(max(color, vec3<f32>(0.0, 0.0, 0.0)), 1.0);
}

fn view_ray_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let yaw = params.view_yaw_pitch_fov_aspect.x * DEG_TO_RAD;
    let pitch = params.view_yaw_pitch_fov_aspect.y * DEG_TO_RAD;
    let fov_y = params.view_yaw_pitch_fov_aspect.z * DEG_TO_RAD;
    let aspect = params.view_yaw_pitch_fov_aspect.w;

    let local_xy = (uv * 2.0 - vec2<f32>(1.0, 1.0)) * vec2<f32>(aspect, 1.0) * tan(0.5 * fov_y);
    let forward = normalize(vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)));
    let right = normalize(vec3<f32>(cos(yaw), 0.0, -sin(yaw)));
    let up = normalize(cross(forward, right));
    return normalize(forward + local_xy.x * right + local_xy.y * up);
}

fn direction_from_azimuth_elevation(azimuth_deg: f32, elevation_deg: f32) -> vec3<f32> {
    let azimuth = azimuth_deg * DEG_TO_RAD;
    let elevation = elevation_deg * DEG_TO_RAD;
    return normalize(vec3<f32>(
        sin(azimuth) * cos(elevation),
        sin(elevation),
        cos(azimuth) * cos(elevation),
    ));
}
