struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var<uniform> view: RuntimeView;
@group(0) @binding(2) var<uniform> sun: CaSun;
@group(0) @binding(3) var transmittance_lut_low: texture_2d<f32>;
@group(0) @binding(4) var transmittance_lut_high: texture_2d<f32>;
@group(0) @binding(5) var lut_sampler: sampler;
@group(0) @binding(6) var sky_view_lut_low: texture_2d<f32>;
@group(0) @binding(7) var sky_view_lut_high: texture_2d<f32>;

const ATM_SUN_SPECTRAL_IRRADIANCE_LOW: vec4<f32> =
    vec4<f32>(1.74773457, 1.76290144, 2.05664327, 1.81461108);
const ATM_SUN_SPECTRAL_IRRADIANCE_HIGH: vec4<f32> =
    vec4<f32>(1.88470162, 1.85678685, 1.72942683, 1.53650429);

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
    let dims = vec2<f32>(textureDimensions(sky_view_lut_low));
    let uv = sky_view_uv_from_dir(dir, dims);
    let lo = max(textureSampleLevel(sky_view_lut_low, lut_sampler, uv, 0.0), vec4<f32>(0.0));
    let hi = max(textureSampleLevel(sky_view_lut_high, lut_sampler, uv, 0.0), vec4<f32>(0.0));
    return max(white_balanced_linear_rec2020_from_spectral8(lo, hi), vec3<f32>(0.0));
}

fn sky_ray_above_ground(dir: vec3<f32>) -> bool {
    let origin = vec3<f32>(0.0, hp.eye_distance_to_earth_center_km, 0.0);
    return ray_sphere_intersection(origin, normalize(dir), hp.earth_radius_km) < 0.0;
}

fn spectral_sun_transmittance_to_rec2020(transmittance_low: vec4<f32>, transmittance_high: vec4<f32>) -> vec3<f32> {
    let clear_sun = max(
        white_balanced_linear_rec2020_from_spectral8(
            ATM_SUN_SPECTRAL_IRRADIANCE_LOW,
            ATM_SUN_SPECTRAL_IRRADIANCE_HIGH,
        ),
        vec3<f32>(1.0e-6),
    );
    let attenuated_sun = max(
        white_balanced_linear_rec2020_from_spectral8(
            ATM_SUN_SPECTRAL_IRRADIANCE_LOW * transmittance_low,
            ATM_SUN_SPECTRAL_IRRADIANCE_HIGH * transmittance_high,
        ),
        vec3<f32>(0.0),
    );
    return clamp(attenuated_sun / clear_sun, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sun_transmittance_at_view(direction: vec3<f32>) -> vec3<f32> {
    let dir = normalize(direction);
    if (!sky_ray_above_ground(dir)) {
        return vec3<f32>(0.0);
    }
    let sun_mu = clamp(dir.y, -1.0, 1.0);
    let normalized_altitude = clamp(hp.eye_altitude_km / hp.atmosphere_thickness_km, 0.0, 1.0);
    let spectral_t_low = transmittance_from_lut(
        transmittance_lut_low,
        lut_sampler,
        sun_mu,
        normalized_altitude,
    );
    let spectral_t_high = transmittance_from_lut(
        transmittance_lut_high,
        lut_sampler,
        sun_mu,
        normalized_altitude,
    );
    return spectral_sun_transmittance_to_rec2020(spectral_t_low, spectral_t_high);
}

@fragment
fn fragment(vertex_out: VertexOutput) -> @location(0) vec4<f32> {
    let relative_world_h = view.relative_world_from_clip * vec4<f32>(vertex_out.ndc, 0.0, 1.0);
    let world_dir = normalize(relative_world_h.xyz);
    var rgb = sky_view_radiance(world_dir);

    if (sky_ray_above_ground(world_dir)) {
        let transmittance = sun_transmittance_at_view(world_dir);
        rgb += ca_sun_disk_eval(sun, world_dir, transmittance);
    }

    return vec4<f32>(rgb, 1.0);
}
