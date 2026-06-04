struct RuntimeView {
    // 天空只需要方向，使用相机相对反投影避免绝对 world_from_clip 的平移影响归一化。
    relative_world_from_clip: mat4x4<f32>,
    world_position: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: RuntimeView;
@group(1) @binding(0) var<uniform> sky_params: CaSkyViewParams;
@group(1) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(2) var lut_sampler: sampler;
@group(1) @binding(3) var sky_view_lut: texture_2d<f32>;
@group(1) @binding(4) var<uniform> sun: CaSun;
@group(1) @binding(5) var<uniform> atmosphere: VoxelAtmosphereLighting;

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

fn sky_view_radiance(dir: vec3<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(sky_view_lut));
    let uv = ca_sky_view_uv_from_dir(sky_params, dir, dims);
    return max(textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb, vec3<f32>(0.0));
}

fn sky_ray_above_ground(dir: vec3<f32>) -> bool {
    let origin = vec3<f32>(0.0, ca_sky_view_height_km(sky_params), 0.0);
    return ca_sky_ray_sphere_intersection(origin, normalize(dir), sky_params.earth_radius_km) < 0.0;
}

fn sun_transmittance_at_view(direction: vec3<f32>) -> vec3<f32> {
    let sun_mu = clamp(normalize(direction).y, -1.0, 1.0);
    if (sun_mu <= 0.0) {
        return vec3<f32>(0.0);
    }
    let normalized_altitude = clamp(
        sky_params.eye_altitude_km / sky_params.atmosphere_thickness_km,
        0.0,
        1.0,
    );
    return ca_atmosphere_sun_transmittance_rec2020(
        transmittance_lut,
        lut_sampler,
        atmosphere,
        sun_mu,
        normalized_altitude,
    );
}

@fragment
fn fragment(vertex_out: VertexOutput) -> @location(0) vec4<f32> {
    // clip z=0 是 reversed-Z 的远处方向；相机相对矩阵让 xyz 直接作为方向使用。
    let relative_world_h = view.relative_world_from_clip * vec4<f32>(vertex_out.ndc, 0.0, 1.0);
    let world_dir = normalize(relative_world_h.xyz);
    var rgb = sky_view_radiance(world_dir);

    if (sky_ray_above_ground(world_dir)) {
        let transmittance = sun_transmittance_at_view(world_dir);
        rgb += ca_sun_disk_eval(sun, world_dir, transmittance);
    }

    return vec4<f32>(rgb, 1.0);
}
