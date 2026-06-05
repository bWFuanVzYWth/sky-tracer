@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var<uniform> view: RuntimeView;
@group(0) @binding(2) var<uniform> sun: CaSun;
@group(0) @binding(3) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(4) var scattering_lut: texture_3d<f32>;
@group(0) @binding(5) var irradiance_lut: texture_2d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var phase_lut: texture_2d_array<f32>;
@group(0) @binding(8) var single_molecular: texture_3d<f32>;
@group(0) @binding(9) var single_aerosol0: texture_3d<f32>;
@group(0) @binding(10) var single_aerosol1: texture_3d<f32>;
@group(0) @binding(11) var single_aerosol2: texture_3d<f32>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_xy: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let xy = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    var out: VertexOut;
    out.position = vec4<f32>(xy * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.clip_xy = out.position.xy;
    return out;
}

fn view_ray_world(clip_xy: vec2<f32>) -> vec3<f32> {
    let p = view.relative_world_from_clip * vec4<f32>(clip_xy, 0.0, 1.0);
    return normalize(p.xyz);
}

fn ground_radiance(r: f32, mu: f32, mu_s: f32, ray_dir: vec3<f32>) -> vec4<f32> {
    let d_ground = distance_to_bottom_atmosphere_boundary(r, mu);
    let normal = normalize(vec3<f32>(
        d_ground * ray_dir.x,
        r + d_ground * ray_dir.y,
        d_ground * ray_dir.z,
    ));
    let ground_mu_s = clamp_cosine(dot(normal, params.sun_dir));

    let view_transmittance = get_transmittance(transmittance_lut, lut_sampler, r, mu, d_ground, true);
    let sun_irradiance =
        params.sun_spectral_irradiance
        * get_transmittance_to_sun(transmittance_lut, lut_sampler, params.earth_radius_km, ground_mu_s)
        * max(ground_mu_s, 0.0);
    let sky_irradiance = sample_irradiance(irradiance_lut, lut_sampler, params.earth_radius_km, ground_mu_s);
    return view_transmittance * params.ground_albedo_spectral * (sun_irradiance + sky_irradiance) * INV_PI;
}

@fragment
fn fragment(input: VertexOut) -> @location(0) vec4<f32> {
    let ray_dir = view_ray_world(input.clip_xy);
    let camera = vec3<f32>(0.0, params.eye_distance_to_earth_center_km, 0.0);
    let r = length(camera);
    let mu = clamp_cosine(dot(camera, ray_dir) / max(r, 1.0e-6));
    let mu_s = clamp_cosine(dot(camera, params.sun_dir) / max(r, 1.0e-6));
    let nu = clamp_cosine(dot(ray_dir, params.sun_dir));
    let hits_ground = ray_intersects_ground(r, mu);

    var spectral =
        sample_single_scattering(
            phase_lut,
            lut_sampler,
            single_molecular,
            single_aerosol0,
            single_aerosol1,
            single_aerosol2,
            r,
            mu,
            mu_s,
            nu,
            hits_ground,
        )
        + sample_scattering(scattering_lut, lut_sampler, r, mu, mu_s, nu, hits_ground);
    if (hits_ground) {
        spectral += ground_radiance(r, mu, mu_s, ray_dir);
    }

    var rgb = white_balanced_linear_rec2020_from_spectral(max(spectral, vec4<f32>(0.0)));

    if (!hits_ground) {
        let sun_transmittance = get_transmittance_to_sun(transmittance_lut, lut_sampler, r, mu_s);
        let sun_through_rgb = white_balanced_linear_rec2020_from_spectral(params.sun_spectral_irradiance * sun_transmittance);
        let sun_outer_rgb = max(
            white_balanced_linear_rec2020_from_spectral(params.sun_spectral_irradiance),
            vec3<f32>(1.0e-6),
        );
        let transmittance_rgb = clamp(sun_through_rgb / sun_outer_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb += ca_sun_disk_eval(sun, ray_dir, transmittance_rgb);
    }

    return vec4<f32>(max(rgb, vec3<f32>(0.0)), 1.0);
}
