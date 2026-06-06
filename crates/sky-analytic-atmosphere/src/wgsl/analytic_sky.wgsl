const PI: f32 = 3.141592653589793;
const SQRT_PI: f32 = 1.772453850905516;
const TWO_INV_SQRT_PI: f32 = 1.1283791670955126;
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

// Source: Vasylyev et al. 2021, "Approximate Chapman function for high zenith angles",
// Earth, Planets and Space, Eq. 37. The erfcx rational form evaluates
// exp(x * x) * erfc(x) using the Smith and Smith approximation cited there.
// https://doi.org/10.1186/s40623-021-01435-y
fn erfcx_vasylyev(x_in: f32) -> f32 {
    let x = clamp(x_in, 0.0, 100.0);
    if (x <= 8.0) {
        return (1.0606963 + 0.55643831 * x) / (1.0619896 + 1.7245609 * x + x * x);
    }
    return 0.56498823 / (0.06651874 + x);
}

fn chapman_vasylyev_upward(radius_km: f32, mu_in: f32, scale_height_km: f32) -> f32 {
    let x = max(radius_km / max(scale_height_km, 1.0e-4), 1.0);
    let mu = clamp(mu_in, 0.0, 1.0);
    if (mu > 0.9999) {
        return 1.0 / max(mu, 1.0e-4);
    }
    let horizon = sqrt(0.5 * PI * x) * (1.0 + 0.375 / x);
    // Eq. 37 contains strong cancellation as mu approaches 0. In f32 this
    // creates visible horizon spikes/bands, so keep the same asymptote and
    // use a stable rational form before blending back to the Vasylyev fit.
    let stable_horizon = horizon / (1.0 + 0.8 * horizon * mu);
    if (mu < 8.0e-3) {
        return stable_horizon;
    }

    let sin_z = sqrt(max(1.0 - mu * mu, 0.0));
    let x_sin = max(x * sin_z, 1.0e-4);
    let erfcx_arg = sqrt(max(x * (1.0 - sin_z), 0.0));
    let first = sqrt(0.5 * PI)
        * sqrt(x_sin)
        * (1.0 + 0.375 / x_sin)
        * erfcx_vasylyev(erfcx_arg);

    let cos2 = max(1.0 - sin_z * sin_z, 1.0e-6);
    let t = sqrt(max(0.5 * sin_z * (1.0 + sin_z), 0.0));
    let bracket = sin_z * sin_z * sin_z - t * t * t + 0.375 * t * cos2;
    let second = (1.0 - t - bracket / max(x_sin * cos2 * cos2, 1.0e-6)) / sqrt(cos2);
    let vasylyev = max(first + second, 0.0);
    let horizon_blend_t = clamp((mu - 8.0e-3) / (2.0e-2 - 8.0e-3), 0.0, 1.0);
    let horizon_blend = horizon_blend_t * horizon_blend_t * (3.0 - 2.0 * horizon_blend_t);
    return stable_horizon + (vasylyev - stable_horizon) * horizon_blend;
}

fn column_to_space(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let radius = max(length(pos_km), bottom_radius_km() + 1.0e-3);
    let altitude = max(radius - bottom_radius_km(), 0.0);
    let up = pos_km / radius;
    let mu = dot(normalize(dir), up);
    return scale_height_km * exp(-altitude / scale_height_km)
        * chapman_vasylyev_upward(radius, max(mu, 0.0), scale_height_km);
}

fn column_to_top_unblocked(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let d = normalize(dir);
    let t_top = ray_sphere_intersection(pos_km, d, top_radius_km());
    if (t_top < 0.0) {
        return 0.0;
    }
    let p_top = pos_km + d * t_top;
    return max(column_to_space(pos_km, d, scale_height_km) - column_to_space(p_top, d, scale_height_km), 0.0);
}

fn column_to_top(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let d = normalize(dir);
    if (ray_sphere_intersection(pos_km, d, bottom_radius_km()) >= 0.0) {
        return 1.0e8;
    }
    let radius = max(length(pos_km), bottom_radius_km() + 1.0e-3);
    let mu = dot(d, pos_km / radius);
    if (mu < 0.0) {
        let near_side = column_to_top_unblocked(pos_km, -d, scale_height_km);
        let tangent_radius = radius * sqrt(max(1.0 - mu * mu, 0.0));
        let tangent_altitude = max(tangent_radius - bottom_radius_km(), 0.0);
        let tangent_to_space = scale_height_km * exp(-tangent_altitude / scale_height_km)
            * chapman_vasylyev_upward(max(tangent_radius, bottom_radius_km()), 0.0, scale_height_km);
        return max(2.0 * tangent_to_space - near_side, 0.0);
    }
    return column_to_top_unblocked(pos_km, d, scale_height_km);
}

fn erf_approx(x_in: f32) -> f32 {
    let s = select(-1.0, 1.0, x_in >= 0.0);
    let x = abs(x_in);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * exp(-x * x));
}

fn erfi_approx(x_in: f32) -> f32 {
    let s = select(-1.0, 1.0, x_in >= 0.0);
    let x = abs(x_in);
    if (x < 2.5) {
        let xx = x * x;
        var term = x;
        var sum = x;
        for (var i = 1u; i <= 12u; i = i + 1u) {
            let n = f32(i);
            term *= xx * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
            sum += term;
        }
        return s * TWO_INV_SQRT_PI * sum;
    }

    let inv_x2 = 1.0 / max(x * x, 1.0e-6);
    let series = 1.0
        + 0.5 * inv_x2
        + 0.75 * inv_x2 * inv_x2
        + 1.875 * inv_x2 * inv_x2 * inv_x2
        + 6.5625 * inv_x2 * inv_x2 * inv_x2 * inv_x2;
    return s * exp(min(x * x, 80.0)) * series / (SQRT_PI * max(x, 1.0e-4));
}

fn average_transmittance_linear_scalar(tau0: f32, tau1: f32) -> f32 {
    let d = tau1 - tau0;
    if (abs(d) < 1.0e-3) {
        return exp(-0.5 * (tau0 + tau1));
    }
    return (exp(-tau0) - exp(-tau1)) / d;
}

fn average_transmittance_quadratic_scalar(tau0: f32, tau_mid: f32, tau1: f32, u_mid_in: f32) -> f32 {
    if (u_mid_in <= 0.05 || u_mid_in >= 0.95) {
        return average_transmittance_linear_scalar(tau0, tau1);
    }

    let u_mid = clamp(u_mid_in, 0.05, 0.95);
    let d = tau1 - tau0;
    // tau(u) = tau0 + d * u + q * u * (1 - u). q is estimated from one
    // closed-form midpoint. Positive q means the optical depth arches above
    // the endpoint line, which is the problematic low-sun horizon case.
    let q = (tau_mid - (tau0 + d * u_mid)) / max(u_mid * (1.0 - u_mid), 1.0e-4);
    if (abs(q) < 1.0e-3) {
        return average_transmittance_linear_scalar(tau0, tau1);
    }

    let b = d + q;
    if (q > 0.0) {
        let sqrt_q = sqrt(q);
        let x0 = -b / (2.0 * sqrt_q);
        let x1 = sqrt_q + x0;
        let scale = exp(clamp(-tau0 - b * b / (4.0 * q), -80.0, 80.0));
        let integral = scale * SQRT_PI * (erfi_approx(x1) - erfi_approx(x0)) / (2.0 * sqrt_q);
        return max(integral, 0.0);
    }

    let p = -q;
    let sqrt_p = sqrt(p);
    let x0 = b / (2.0 * sqrt_p);
    let x1 = sqrt_p + x0;
    let scale = exp(clamp(-tau0 + b * b / (4.0 * p), -80.0, 80.0));
    let integral = scale * SQRT_PI * (erf_approx(x1) - erf_approx(x0)) / (2.0 * sqrt_p);
    return max(integral, 0.0);
}

fn average_transmittance_quadratic(
    tau0: vec4<f32>,
    tau_mid: vec4<f32>,
    tau1: vec4<f32>,
    u_mid: f32,
) -> vec4<f32> {
    return vec4<f32>(
        average_transmittance_quadratic_scalar(tau0.x, tau_mid.x, tau1.x, u_mid),
        average_transmittance_quadratic_scalar(tau0.y, tau_mid.y, tau1.y, u_mid),
        average_transmittance_quadratic_scalar(tau0.z, tau_mid.z, tau1.z, u_mid),
        average_transmittance_quadratic_scalar(tau0.w, tau_mid.w, tau1.w, u_mid),
    );
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
    let p_mid = origin + dir * (0.5 * t_top);
    let view_col_r = column_to_top(origin, dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let view_col_m = column_to_top(origin, dir, MIE_SCALE_HEIGHT_KM);
    let view_col_mid_r = clamp(view_col_r - column_to_top(p_mid, dir, RAYLEIGH_SCALE_HEIGHT_KM), 0.0, view_col_r);
    let view_col_mid_m = clamp(view_col_m - column_to_top(p_mid, dir, MIE_SCALE_HEIGHT_KM), 0.0, view_col_m);
    let sun_col0_r = column_to_top(origin, sun_dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let sun_col0_m = column_to_top(origin, sun_dir, MIE_SCALE_HEIGHT_KM);
    let sun_col_mid_r = column_to_top(p_mid, sun_dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let sun_col_mid_m = column_to_top(p_mid, sun_dir, MIE_SCALE_HEIGHT_KM);
    let sun_col1_r = column_to_top(p_end, sun_dir, RAYLEIGH_SCALE_HEIGHT_KM);
    let sun_col1_m = column_to_top(p_end, sun_dir, MIE_SCALE_HEIGHT_KM);

    let tau_s0 = ap.rayleigh_scattering_base * sun_col0_r + ap.mie_extinction_base * sun_col0_m;
    let tau_mid = ap.rayleigh_scattering_base * (sun_col_mid_r + view_col_mid_r)
        + ap.mie_extinction_base * (sun_col_mid_m + view_col_mid_m);
    let tau_s1_v = ap.rayleigh_scattering_base * (sun_col1_r + view_col_r)
        + ap.mie_extinction_base * (sun_col1_m + view_col_m);
    let avg_t_r = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_r / max(view_col_r, 1.0e-6),
    );
    let avg_t_m = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_m / max(view_col_m, 1.0e-6),
    );
    let mu = dot(sun_dir, dir);
    let rayleigh_scatter = ap.rayleigh_scattering_base * view_col_r * rayleigh_phase(mu) * avg_t_r;
    let mie_scatter = ap.mie_scattering_base * view_col_m * henyey_greenstein_phase(mu, MIE_G) * avg_t_m;
    let sky_spectral = ap.sun_spectral_irradiance * (rayleigh_scatter + mie_scatter);
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
