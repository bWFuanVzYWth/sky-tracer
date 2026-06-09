struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var<uniform> view: RuntimeView;
@group(0) @binding(2) var<uniform> sun: CaSun;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
@group(0) @binding(5) var sky_view_lut: texture_2d<f32>;

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
    let uv = sky_view_uv_from_dir(dir, dims);
    return max(textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb, vec3<f32>(0.0));
}

fn sky_ray_above_ground(dir: vec3<f32>) -> bool {
    let origin = vec3<f32>(0.0, sky_view_height_km(), 0.0);
    return ray_sphere_intersection(origin, normalize(dir), hp.earth_radius_km) < 0.0;
}

fn spectral_sun_transmittance_to_rec2020(transmittance: vec4<f32>) -> vec3<f32> {
    let clear_sun = max(white_balanced_linear_rec2020_from_spectral(hp.sun_spectral_irradiance), vec3<f32>(1.0e-6));
    let attenuated_sun = max(white_balanced_linear_rec2020_from_spectral(hp.sun_spectral_irradiance * transmittance), vec3<f32>(0.0));
    return clamp(attenuated_sun / clear_sun, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sun_transmittance_at_view(direction: vec3<f32>) -> vec3<f32> {
    let dir = normalize(direction);
    let sun_mu = clamp(dir.y, -1.0, 1.0);
    if (sun_mu <= 0.0) {
        return vec3<f32>(0.0);
    }
    let normalized_altitude = clamp(hp.eye_altitude_km / hp.atmosphere_thickness_km, 0.0, 1.0);
    let spectral_t = transmittance_from_lut(transmittance_lut, lut_sampler, sun_mu, normalized_altitude);
    return spectral_sun_transmittance_to_rec2020(spectral_t);
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
