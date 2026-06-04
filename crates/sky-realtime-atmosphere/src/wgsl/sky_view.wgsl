// Shared SkyView sampling helpers.
//
// The texture is a single-view biased sky approximation. Providers are
// responsible for publishing params matching the generated LUT.

struct CaSkyViewParams {
    earth_radius_km: f32,
    atmosphere_thickness_km: f32,
    eye_distance_to_earth_center_km: f32,
    eye_altitude_km: f32,
    sun_dir: vec3<f32>,
    sky_view_height_km: f32,
}

const CA_SKY_PI: f32 = 3.141592653589793;

fn ca_sky_from_unit_to_sub_uvs(u: f32, resolution: f32) -> f32 {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

fn ca_sky_ray_sphere_intersection(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    if (c > 0.0 && b > 0.0) {
        return -1.0;
    }
    let d = b * b - c;
    if (d < 0.0) {
        return -1.0;
    }
    if (d > b * b) {
        return -b + sqrt(d);
    }
    return -b - sqrt(d);
}

fn ca_sky_top_radius_km(p: CaSkyViewParams) -> f32 {
    return p.earth_radius_km + p.atmosphere_thickness_km;
}

fn ca_sky_view_height_km(p: CaSkyViewParams) -> f32 {
    return clamp(
        p.sky_view_height_km,
        p.earth_radius_km + 1.0e-3,
        ca_sky_top_radius_km(p) - 1.0e-3,
    );
}

fn ca_sky_view_horizon_angles(p: CaSkyViewParams, view_height: f32) -> vec2<f32> {
    let bottom = p.earth_radius_km;
    let v_horizon = sqrt(max(view_height * view_height - bottom * bottom, 0.0));
    let cos_beta = clamp(v_horizon / max(view_height, 1.0e-6), 0.0, 1.0);
    let beta = acos(cos_beta);
    let zenith_horizon_angle = CA_SKY_PI - beta;
    return vec2<f32>(zenith_horizon_angle, beta);
}

fn ca_sky_view_params_to_uv(
    p: CaSkyViewParams,
    view_zenith_cos_angle: f32,
    light_view_cos_angle: f32,
    intersect_ground: bool,
    dims: vec2<f32>,
) -> vec2<f32> {
    let angles = ca_sky_view_horizon_angles(p, ca_sky_view_height_km(p));
    let zenith_horizon_angle = angles.x;
    let beta = max(angles.y, 1.0e-6);
    let view_angle = acos(clamp(view_zenith_cos_angle, -1.0, 1.0));

    var v: f32;
    if (!intersect_ground) {
        var coord = clamp(view_angle / max(zenith_horizon_angle, 1.0e-6), 0.0, 1.0);
        coord = 1.0 - coord;
        coord = sqrt(max(coord, 0.0));
        coord = 1.0 - coord;
        v = coord * 0.5;
    } else {
        var coord = clamp((view_angle - zenith_horizon_angle) / beta, 0.0, 1.0);
        coord = sqrt(max(coord, 0.0));
        v = coord * 0.5 + 0.5;
    }

    var u = clamp(-light_view_cos_angle * 0.5 + 0.5, 0.0, 1.0);
    u = sqrt(u);

    return vec2<f32>(
        ca_sky_from_unit_to_sub_uvs(u, dims.x),
        ca_sky_from_unit_to_sub_uvs(v, dims.y),
    );
}

fn ca_sky_view_uv_from_dir(p: CaSkyViewParams, dir_in: vec3<f32>, dims: vec2<f32>) -> vec2<f32> {
    let dir = normalize(dir_in);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_zenith_cos_angle = dot(dir, up);

    let view_h = dir - up * view_zenith_cos_angle;
    let sun_h = p.sun_dir - up * dot(p.sun_dir, up);
    let view_h_len = length(view_h);
    let sun_h_len = length(sun_h);
    let view_h_dir = view_h / max(view_h_len, 1.0e-5);
    let sun_h_dir = sun_h / max(sun_h_len, 1.0e-5);
    let light_view_cos_angle = select(
        1.0,
        dot(view_h_dir, sun_h_dir),
        view_h_len > 1.0e-5 && sun_h_len > 1.0e-5,
    );

    let origin = vec3<f32>(0.0, ca_sky_view_height_km(p), 0.0);
    let intersect_ground =
        ca_sky_ray_sphere_intersection(origin, dir, p.earth_radius_km) >= 0.0;
    return ca_sky_view_params_to_uv(
        p,
        view_zenith_cos_angle,
        light_view_cos_angle,
        intersect_ground,
        dims,
    );
}
