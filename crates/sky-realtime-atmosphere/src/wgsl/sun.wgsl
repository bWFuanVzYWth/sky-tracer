// SkyView 不包含太阳圆盘。后端需要分别采样天空与太阳，避免环境采样和显式
// 太阳采样共用错误 PDF。

struct CaSun {
    sun_to_scene: vec3<f32>,
    angular_radius_rad: f32,
    irradiance_rec2020_w_m2: vec3<f32>,
    cos_angular_radius: f32,
}

const CA_SUN_PI: f32 = 3.141592653589793;
const CA_SUN_DIRECTIONAL_COS_RADIUS: f32 = 0.999999;

fn ca_sun_to_sun(sun: CaSun) -> vec3<f32> {
    return normalize(-sun.sun_to_scene);
}

fn ca_sun_cos_radius(sun: CaSun) -> f32 {
    return clamp(sun.cos_angular_radius, -1.0, 1.0);
}

fn ca_sun_solid_angle_from_cos_radius(cos_radius: f32) -> f32 {
    return 2.0 * CA_SUN_PI * max(1.0 - cos_radius, 0.0);
}

fn ca_sun_solid_angle(sun: CaSun) -> f32 {
    return ca_sun_solid_angle_from_cos_radius(ca_sun_cos_radius(sun));
}

fn ca_sun_disk_is_finite(sun: CaSun) -> bool {
    return ca_sun_cos_radius(sun) < CA_SUN_DIRECTIONAL_COS_RADIUS
        && ca_sun_solid_angle(sun) > 0.0;
}

fn ca_sun_disk_pdf(sun: CaSun) -> f32 {
    let solid_angle = ca_sun_solid_angle(sun);
    if !ca_sun_disk_is_finite(sun) {
        return 0.0;
    }
    return 1.0 / solid_angle;
}

fn ca_sun_disk_contains_dir(sun: CaSun, direction: vec3<f32>) -> bool {
    if !ca_sun_disk_is_finite(sun) {
        return false;
    }
    return dot(normalize(direction), ca_sun_to_sun(sun)) >= ca_sun_cos_radius(sun);
}

fn ca_sun_disk_radiance(sun: CaSun, transmittance: vec3<f32>) -> vec3<f32> {
    let solid_angle = ca_sun_solid_angle(sun);
    if !ca_sun_disk_is_finite(sun) {
        return vec3<f32>(0.0);
    }
    return sun.irradiance_rec2020_w_m2 * transmittance / solid_angle;
}

fn ca_sun_disk_eval(sun: CaSun, direction: vec3<f32>, transmittance: vec3<f32>) -> vec3<f32> {
    if !ca_sun_disk_contains_dir(sun, direction) {
        return vec3<f32>(0.0);
    }
    return ca_sun_disk_radiance(sun, transmittance);
}

fn ca_sun_build_tangent(axis: vec3<f32>) -> vec3<f32> {
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(axis.y) > 0.999,
    );
    return normalize(cross(reference, axis));
}

fn ca_sun_sample_uniform_cone(sun: CaSun, sample: vec2<f32>) -> vec3<f32> {
    let axis = ca_sun_to_sun(sun);
    let cos_radius = ca_sun_cos_radius(sun);
    let cos_theta = mix(cos_radius, 1.0, sample.x);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * CA_SUN_PI * sample.y;
    let tangent = ca_sun_build_tangent(axis);
    let bitangent = cross(axis, tangent);
    return normalize(
        tangent * (cos(phi) * sin_theta)
            + bitangent * (sin(phi) * sin_theta)
            + axis * cos_theta
    );
}
