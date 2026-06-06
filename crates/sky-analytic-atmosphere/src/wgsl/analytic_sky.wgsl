const PI: f32 = 3.141592653589793;
const INV_4PI: f32 = 0.07957747154594767;
const RAYLEIGH_PHASE_SCALE: f32 = 0.05968310365946075;
const RAYLEIGH_SCALE_HEIGHT_KM: f32 = 8.0;
const MIE_SCALE_HEIGHT_KM: f32 = 1.2;
const MIE_G: f32 = 0.8;
const SUN_DIRECTIONAL_COS_RADIUS: f32 = 0.999999;

struct AnalyticParams {
    planet: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_spectral_irradiance: vec4<f32>,
    rayleigh_scattering_base: vec4<f32>,
    mie_scattering_base: vec4<f32>,
    mie_extinction_base: vec4<f32>,
}

struct AnalyticView {
    relative_world_from_clip: mat4x4<f32>,
    world_position: vec4<f32>,
}

struct AnalyticSun {
    sun_to_scene: vec3<f32>,
    angular_radius_rad: f32,
    irradiance_rec2020_w_m2: vec3<f32>,
    cos_angular_radius: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@group(0) @binding(0) var<uniform> ap: AnalyticParams;
@group(0) @binding(1) var<uniform> view: AnalyticView;
@group(0) @binding(2) var<uniform> sun: AnalyticSun;

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

@fragment
fn fragment(vertex_out: VertexOutput) -> @location(0) vec4<f32> {
    let relative_world_h = view.relative_world_from_clip * vec4<f32>(vertex_out.ndc, 0.0, 1.0);
    let world_dir = normalize(relative_world_h.xyz);
    let radiance = analytic_sky_radiance(world_dir);
    return vec4<f32>(radiance, 1.0);
}

fn bottom_radius_km() -> f32 {
    return ap.planet.x;
}

fn top_radius_km() -> f32 {
    return ap.planet.y;
}

fn eye_radius_km() -> f32 {
    return clamp(ap.planet.z, bottom_radius_km() + 1.0e-3, top_radius_km() - 1.0e-3);
}

fn to_sun_dir() -> vec3<f32> {
    return normalize(ap.sun_dir.xyz);
}

fn ray_sphere_intersection(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
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

fn rayleigh_phase(mu: f32) -> f32 {
    return RAYLEIGH_PHASE_SCALE * (1.0 + mu * mu);
}

fn henyey_greenstein_phase(mu: f32, g: f32) -> f32 {
    let denom = pow(max(1.0 + g * g - 2.0 * g * mu, 1.0e-5), 1.5);
    return INV_4PI * (1.0 - g * g) / denom;
}

fn chapman_approx(radius_km: f32, mu: f32, scale_height_km: f32) -> f32 {
    let x = max(radius_km / max(scale_height_km, 1.0e-4), 1.0);
    let horizon = sqrt(0.5 * PI * x);
    let inv_horizon = 1.0 / max(horizon, 1.0e-4);
    if (mu >= 0.0) {
        return 1.0 / max(mu + inv_horizon, 1.0e-4);
    }
    let reflected = 1.0 / max(-mu + inv_horizon, 1.0e-4);
    return max(horizon, 2.0 * horizon - reflected);
}

fn column_to_space(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let radius = max(length(pos_km), bottom_radius_km() + 1.0e-3);
    let altitude = max(radius - bottom_radius_km(), 0.0);
    let up = pos_km / radius;
    let mu = dot(normalize(dir), up);
    return scale_height_km * exp(-altitude / scale_height_km)
        * chapman_approx(radius, mu, scale_height_km);
}

fn column_to_top(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let d = normalize(dir);
    if (ray_sphere_intersection(pos_km, d, bottom_radius_km()) >= 0.0) {
        return 1.0e8;
    }
    let t_top = ray_sphere_intersection(pos_km, d, top_radius_km());
    if (t_top < 0.0) {
        return 0.0;
    }
    let p_top = pos_km + d * t_top;
    return max(column_to_space(pos_km, d, scale_height_km) - column_to_space(p_top, d, scale_height_km), 0.0);
}

fn average_transmittance(tau0: vec4<f32>, tau1: vec4<f32>) -> vec4<f32> {
    let d = tau1 - tau0;
    let small = abs(d) < vec4<f32>(1.0e-3);
    let safe_d = select(d, vec4<f32>(1.0), small);
    let exact = (exp(-tau0) - exp(-tau1)) / safe_d;
    let approx = exp(-0.5 * (tau0 + tau1));
    return select(exact, approx, small);
}

fn spectral_sun_transmittance_to_rec2020(transmittance: vec4<f32>) -> vec3<f32> {
    let clear_sun = max(white_balanced_linear_rec2020_from_spectral(ap.sun_spectral_irradiance), vec3<f32>(1.0e-6));
    let attenuated_sun = max(white_balanced_linear_rec2020_from_spectral(ap.sun_spectral_irradiance * transmittance), vec3<f32>(0.0));
    return clamp(attenuated_sun / clear_sun, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sun_disk_eval(direction: vec3<f32>, transmittance: vec3<f32>) -> vec3<f32> {
    let cos_radius = clamp(sun.cos_angular_radius, -1.0, 1.0);
    if (cos_radius >= SUN_DIRECTIONAL_COS_RADIUS) {
        return vec3<f32>(0.0);
    }
    if (dot(normalize(direction), to_sun_dir()) < cos_radius) {
        return vec3<f32>(0.0);
    }
    let solid_angle = 2.0 * PI * max(1.0 - cos_radius, 0.0);
    if (solid_angle <= 0.0) {
        return vec3<f32>(0.0);
    }
    return sun.irradiance_rec2020_w_m2 * transmittance / solid_angle;
}

fn analytic_sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    let origin = vec3<f32>(0.0, eye_radius_km(), 0.0);
    let dir = normalize(direction);
    if (ray_sphere_intersection(origin, dir, bottom_radius_km()) >= 0.0) {
        return vec3<f32>(0.0);
    }

    let t_top = ray_sphere_intersection(origin, dir, top_radius_km());
    if (t_top < 0.0) {
        return vec3<f32>(0.0);
    }

    let sun_dir = to_sun_dir();
    let p_end = origin + dir * t_top;
    let view_col_r = column_to_top(origin, dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let view_col_m = column_to_top(origin, dir, MIE_SCALE_HEIGHT_KM);
    let sun_col0_r = column_to_top(origin, sun_dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let sun_col0_m = column_to_top(origin, sun_dir, MIE_SCALE_HEIGHT_KM);
    let sun_col1_r = column_to_top(p_end, sun_dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let sun_col1_m = column_to_top(p_end, sun_dir, MIE_SCALE_HEIGHT_KM);

    let tau_s0 = ap.rayleigh_scattering_base * sun_col0_r + ap.mie_extinction_base * sun_col0_m;
    let tau_s1_v = ap.rayleigh_scattering_base * (sun_col1_r + view_col_r)
        + ap.mie_extinction_base * (sun_col1_m + view_col_m);
    let avg_t = average_transmittance(tau_s0, tau_s1_v);
    let mu = dot(sun_dir, dir);
    let scatter = ap.rayleigh_scattering_base * view_col_r * rayleigh_phase(mu)
        + ap.mie_scattering_base * view_col_m * henyey_greenstein_phase(mu, MIE_G);
    let sky_spectral = ap.sun_spectral_irradiance * scatter * avg_t;
    var rgb = max(white_balanced_linear_rec2020_from_spectral(sky_spectral), vec3<f32>(0.0));

    let sun_tau = ap.rayleigh_scattering_base * sun_col0_r + ap.mie_extinction_base * sun_col0_m;
    let sun_t = spectral_sun_transmittance_to_rec2020(exp(-sun_tau));
    rgb += sun_disk_eval(dir, sun_t);
    return rgb;
}

fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    let m = mat4x3<f32>(
        vec3<f32>(86.3182148, -0.122697755, 0.547224869),
        vec3<f32>(30.0452569, 92.3535448, -8.36373448),
        vec3<f32>(-1.57281544, 29.5419052, 48.5065647),
        vec3<f32>(3.57535605, -9.78845357, 70.7444659),
    );
    return m * l;
}

fn white_balance_rec2020(rgb: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(1.01363293, 0.00103366792, 0.00115468962),
        vec3<f32>(0.019007348, 0.974260442, -0.00255465921),
        vec3<f32>(0.00260596377, -0.00288158643, 1.19816913),
    );
    return m * rgb;
}

fn white_balanced_linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    return white_balance_rec2020(linear_rec2020_from_spectral(l));
}
