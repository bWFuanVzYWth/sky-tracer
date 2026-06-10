@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_out: texture_storage_2d<rgba16float, write>;

const TRANSMITTANCE_STEPS: u32 = 96u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(transmittance_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let bottom = hp.earth_radius_km;
    let top = bottom + hp.atmosphere_thickness_km;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = h * uv.y;
    let radius_km = sqrt(rho * rho + bottom * bottom);
    let d_min = top - radius_km;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);

    var mu: f32;
    if (d == 0.0) {
        mu = 1.0;
    } else {
        mu = (h * h - rho * rho - d * d) / (2.0 * radius_km * d);
    }
    mu = clamp(mu, -1.0, 1.0);

    let sin_theta = sqrt(max(1.0 - mu * mu, 0.0));
    let ray_origin = vec3<f32>(0.0, radius_km, 0.0);
    let ray_dir = vec3<f32>(sin_theta, mu, 0.0);
    let t_max = ray_sphere_intersection(ray_origin, ray_dir, top);
    let dt = max(t_max, 0.0) / f32(TRANSMITTANCE_STEPS);

    var optical_depth = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < TRANSMITTANCE_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let pos = ray_origin + ray_dir * t;
        let altitude = length(pos) - hp.earth_radius_km;
        let coeffs = get_atmosphere_collision_coefficients(hp, altitude);
        optical_depth += coeffs.extinction * dt;
    }

    textureStore(transmittance_out, vec2<i32>(gid.xy), exp(-optical_depth));
}
