const PI: f32 = 3.141592653589793;
const SQRT_PI: f32 = 1.772453850905516;
const TWO_INV_SQRT_PI: f32 = 1.1283791670955126;
const INV_PI: f32 = 0.31830988618379067;
const INV_4PI: f32 = 0.07957747154594767;
const RAYLEIGH_PHASE_SCALE: f32 = 0.05968310365946075;

// Physical model knobs for this analytic approximation. Rayleigh and Mie use
// a radial quadratic exponential density:
//   rho(r) = exp(-(r^2 - Rg^2) / (2 * Rg * H)).
// H is the near-ground equivalent scale height, fitted from the offline data
// as vertical column / surface density rather than the usual 8 km / 1.2 km
// textbook exponential model.
const RAYLEIGH_EQUIVALENT_HEIGHT_KM: f32 = 8.4675;
const MIE_EQUIVALENT_HEIGHT_KM: f32 = 1.6670;
const MIE_HACK_PHASE_E: f32 = 3500.0;
const OZONE_CENTER_ALTITUDE_KM: f32 = 22.0;
const OZONE_HALF_WIDTH_KM: f32 = 22.0;
const GROUND_ALBEDO: f32 = 0.18;

// Numerical and quality controls.
const SEGMENT_MIDPOINT_U: f32 = 0.5;
const PLANET_RADIUS_EPS_KM: f32 = 1.0e-3;
const MIN_SCALE_HEIGHT_KM: f32 = 1.0e-4;
const DISTANCE_EPS_KM: f32 = 1.0e-6;
const GEOMETRY_EPS: f32 = 1.0e-8;
const LOG_ARG_EPS: f32 = 1.0e-20;
const NO_INTERSECTION: f32 = -1.0;
const OPAQUE_COLUMN: f32 = 1.0e8;
const SUN_DIRECTIONAL_COS_RADIUS: f32 = 0.999999;
const ERFCX_MAX_X: f32 = 100.0;
const ERFCX_RATIONAL_SWITCH_X: f32 = 8.0;
const ERFI_SERIES_SWITCH_X: f32 = 2.5;
const ERFI_SERIES_TERMS: u32 = 12u;
const OPTICAL_DEPTH_CLAMP: f32 = 80.0;
const TRANSMITTANCE_LINEAR_TAU_EPS: f32 = 1.0e-3;
const TRANSMITTANCE_QUADRATIC_Q_EPS: f32 = 1.0e-3;
const TRANSMITTANCE_QUADRATIC_U_MIN: f32 = 0.05;
const TRANSMITTANCE_QUADRATIC_U_MAX: f32 = 0.95;
const COLUMN_RATIO_EPS: f32 = 1.0e-6;
const GROUND_SKY_RAYLEIGH_PHASE_MU_SCALE: f32 = 0.5;

struct AnalyticParams {
    planet: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_spectral_irradiance: vec4<f32>,
    rayleigh_scattering_base: vec4<f32>,
    mie_scattering_base: vec4<f32>,
    mie_extinction_base: vec4<f32>,
    ozone_absorption_base: vec4<f32>,
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

// Fullscreen entry points.
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

// Uniform-derived scene parameters.
fn bottom_radius_km() -> f32 {
    return ap.planet.x;
}

fn top_radius_km() -> f32 {
    return ap.planet.y;
}

fn eye_radius_km() -> f32 {
    return clamp(ap.planet.z, bottom_radius_km() + PLANET_RADIUS_EPS_KM, top_radius_km() - PLANET_RADIUS_EPS_KM);
}

fn to_sun_dir() -> vec3<f32> {
    return normalize(ap.sun_dir.xyz);
}

// Geometry helpers.
fn ray_sphere_intersection(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    if (c > 0.0 && b > 0.0) {
        return NO_INTERSECTION;
    }
    let d = b * b - c;
    if (d < 0.0) {
        return NO_INTERSECTION;
    }
    if (d > b * b) {
        return -b + sqrt(d);
    }
    return -b - sqrt(d);
}

fn eye_ground_intersection(dir: vec3<f32>) -> f32 {
    let r = eye_radius_km();
    let ground = bottom_radius_km();
    let horizon_mu2 = max(1.0 - (ground * ground) / max(r * r, GEOMETRY_EPS), 0.0);
    let mu = normalize(dir).y;
    if (mu >= 0.0 || mu * mu < horizon_mu2) {
        return NO_INTERSECTION;
    }
    return -r * mu - r * sqrt(max(mu * mu - horizon_mu2, 0.0));
}

// Phase functions.
fn rayleigh_phase(mu: f32) -> f32 {
    return RAYLEIGH_PHASE_SCALE * (1.0 + mu * mu);
}

fn opac_like_mie_phase_hack(mu: f32) -> f32 {
    // Empirical Alpha-Piscium/Jessie Klein-Nishina-style phase. This is not
    // a physically correct atmospheric Mie phase, but it is cheap, normalized,
    // and visually closer to OPAC's strong forward lobe than Henyey-Greenstein.
    let e = MIE_HACK_PHASE_E;
    return e / (2.0 * PI * (e * (1.0 - clamp(mu, -1.0, 1.0)) + 1.0) * log(2.0 * e + 1.0));
}

// Source: Vasylyev et al. 2021, "Approximate Chapman function for high zenith angles",
// Earth, Planets and Space, Eq. 37. The erfcx rational form evaluates
// exp(x * x) * erfc(x) using the Smith and Smith approximation cited there.
// We reuse only the stable erfcx approximation here; the density model below
// is radial quadratic exponential rather than a Chapman standard exponential.
// https://doi.org/10.1186/s40623-021-01435-y
fn erfcx_vasylyev(x_in: f32) -> f32 {
    let x = clamp(x_in, 0.0, ERFCX_MAX_X);
    if (x <= ERFCX_RATIONAL_SWITCH_X) {
        return (1.0606963 + 0.55643831 * x) / (1.0619896 + 1.7245609 * x + x * x);
    }
    return 0.56498823 / (0.06651874 + x);
}

// Analytic column-density approximation for radial quadratic exponential
// density. This avoids the unstable "column to space A - column to space B"
// difference and has fixed cost for every view direction.
fn ray_sphere_exit_distance(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let d = b * b - c;
    if (d < 0.0) {
        return NO_INTERSECTION;
    }
    let root = sqrt(d);
    let t0 = -b - root;
    let t1 = -b + root;
    if (t1 >= 0.0) {
        return t1;
    }
    if (t0 >= 0.0) {
        return t0;
    }
    return NO_INTERSECTION;
}

fn radial_quadratic_column_positive_side(c: f32, b: f32, distance: f32, a: f32) -> f32 {
    let sqrt_a = sqrt(a);
    let x0 = max(b, 0.0) * sqrt_a;
    let x1 = x0 + distance * sqrt_a;
    let attenuation = exp(clamp(-(x1 * x1 - x0 * x0), -OPTICAL_DEPTH_CLAMP, 0.0));
    let erfcx_delta = max(erfcx_vasylyev(x0) - attenuation * erfcx_vasylyev(x1), 0.0);
    let density_base = exp(clamp(-a * max(c, 0.0), -OPTICAL_DEPTH_CLAMP, 0.0));
    return density_base * SQRT_PI * erfcx_delta / (2.0 * sqrt_a);
}

fn radial_quadratic_density_column_segment(
    pos_km: vec3<f32>,
    dir: vec3<f32>,
    distance_km: f32,
    scale_height_km: f32,
) -> f32 {
    let distance = max(distance_km, 0.0);
    if (distance <= DISTANCE_EPS_KM) {
        return 0.0;
    }

    let d = normalize(dir);
    let ground_radius = bottom_radius_km();
    let h = max(scale_height_km, MIN_SCALE_HEIGHT_KM);
    let a = 1.0 / (2.0 * ground_radius * h);
    let c = max(dot(pos_km, pos_km) - ground_radius * ground_radius, 0.0);
    let b = dot(pos_km, d);

    if (b >= 0.0) {
        return radial_quadratic_column_positive_side(c, b, distance, a);
    }

    let b_end = b + distance;
    if (b_end <= 0.0) {
        let end_pos = pos_km + d * distance;
        let end_c = max(dot(end_pos, end_pos) - ground_radius * ground_radius, 0.0);
        return radial_quadratic_column_positive_side(end_c, -b_end, distance, a);
    }

    let sqrt_a = sqrt(a);
    let x0 = b * sqrt_a;
    let x1 = b_end * sqrt_a;
    let tangent_excess_radius2 = max(c - b * b, 0.0);
    let tangent_density = exp(clamp(-a * tangent_excess_radius2, -OPTICAL_DEPTH_CLAMP, 0.0));
    return max(tangent_density * SQRT_PI * (erf_approx(x1) - erf_approx(x0)) / (2.0 * sqrt_a), 0.0);
}

fn column_to_top(pos_km: vec3<f32>, dir: vec3<f32>, scale_height_km: f32) -> f32 {
    let d = normalize(dir);
    if (ray_sphere_intersection(pos_km, d, bottom_radius_km()) >= 0.0) {
        return OPAQUE_COLUMN;
    }
    let t_top = ray_sphere_exit_distance(pos_km, d, top_radius_km());
    if (t_top < 0.0) {
        return 0.0;
    }
    return radial_quadratic_density_column_segment(pos_km, d, t_top, scale_height_km);
}

// Closed-form triangular ozone layer.
fn radial_ramp_primitive(pos_km: vec3<f32>, dir: vec3<f32>, s: f32, threshold_radius_km: f32) -> f32 {
    let b = dot(pos_km, dir);
    let p2 = max(dot(pos_km, pos_km) - b * b, GEOMETRY_EPS);
    let p = sqrt(p2);
    let x = s + b;
    let radius = sqrt(x * x + p2);
    let sqrt_integral = 0.5 * (x * radius + p2 * log(max((x + radius) / p, LOG_ARG_EPS)));
    return sqrt_integral - threshold_radius_km * x;
}

fn radial_ramp_interval(
    pos_km: vec3<f32>,
    dir: vec3<f32>,
    lo: f32,
    hi: f32,
    threshold_radius_km: f32,
) -> f32 {
    if (hi <= lo) {
        return 0.0;
    }

    let b = dot(pos_km, dir);
    let p2 = max(dot(pos_km, pos_km) - b * b, 0.0);
    let threshold2 = threshold_radius_km * threshold_radius_km;
    var mass = 0.0;
    if (p2 >= threshold2) {
        mass = radial_ramp_primitive(pos_km, dir, hi, threshold_radius_km)
            - radial_ramp_primitive(pos_km, dir, lo, threshold_radius_km);
    } else {
        let root = sqrt(max(threshold2 - p2, 0.0));
        let enter = -b - root;
        let exit = -b + root;
        let hi0 = min(hi, enter);
        if (hi0 > lo) {
            mass += radial_ramp_primitive(pos_km, dir, hi0, threshold_radius_km)
                - radial_ramp_primitive(pos_km, dir, lo, threshold_radius_km);
        }
        let lo1 = max(lo, exit);
        if (hi > lo1) {
            mass += radial_ramp_primitive(pos_km, dir, hi, threshold_radius_km)
                - radial_ramp_primitive(pos_km, dir, lo1, threshold_radius_km);
        }
    }
    return max(mass, 0.0);
}

fn ozone_triangle_column_segment(pos_km: vec3<f32>, dir: vec3<f32>, t_max_km: f32) -> f32 {
    if (t_max_km <= 0.0) {
        return 0.0;
    }

    let d = normalize(dir);
    let bottom = bottom_radius_km();
    let r0 = bottom + max(OZONE_CENTER_ALTITUDE_KM - OZONE_HALF_WIDTH_KM, 0.0);
    let r1 = bottom + OZONE_CENTER_ALTITUDE_KM;
    let r2 = bottom + OZONE_CENTER_ALTITUDE_KM + OZONE_HALF_WIDTH_KM;

    let b = dot(pos_km, d);
    let p2 = max(dot(pos_km, pos_km) - b * b, 0.0);
    let disc = r2 * r2 - p2;
    if (disc <= 0.0) {
        return 0.0;
    }

    let root = sqrt(disc);
    let lo = max(0.0, -b - root);
    let hi = min(t_max_km, -b + root);
    if (hi <= lo) {
        return 0.0;
    }

    let mass = radial_ramp_interval(pos_km, d, lo, hi, r0)
        - 2.0 * radial_ramp_interval(pos_km, d, lo, hi, r1)
        + radial_ramp_interval(pos_km, d, lo, hi, r2);
    return max(mass / max(OZONE_HALF_WIDTH_KM, MIN_SCALE_HEIGHT_KM), 0.0);
}

fn ozone_column_to_top(pos_km: vec3<f32>, dir: vec3<f32>) -> f32 {
    let d = normalize(dir);
    if (ray_sphere_intersection(pos_km, d, bottom_radius_km()) >= 0.0) {
        return OPAQUE_COLUMN;
    }
    let t_top = ray_sphere_exit_distance(pos_km, d, top_radius_km());
    if (t_top < 0.0) {
        return 0.0;
    }
    return ozone_triangle_column_segment(pos_km, d, t_top);
}

fn optical_depth_from_columns(rayleigh_col: f32, mie_col: f32, ozone_col: f32) -> vec4<f32> {
    return ap.rayleigh_scattering_base * rayleigh_col
        + ap.mie_extinction_base * mie_col
        + ap.ozone_absorption_base * ozone_col;
}

fn optical_depth_segment(pos_km: vec3<f32>, dir: vec3<f32>, distance_km: f32) -> vec4<f32> {
    let d = normalize(dir);
    let distance = max(distance_km, 0.0);
    let rayleigh_col = radial_quadratic_density_column_segment(pos_km, d, distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let mie_col = radial_quadratic_density_column_segment(pos_km, d, distance, MIE_EQUIVALENT_HEIGHT_KM);
    let ozone_col = ozone_triangle_column_segment(pos_km, d, distance);
    return optical_depth_from_columns(rayleigh_col, mie_col, ozone_col);
}

// Ground terms.
fn ground_bounce_transfer(pos_km: vec3<f32>, sun_dir: vec3<f32>) -> vec4<f32> {
    let radius = max(length(pos_km), bottom_radius_km() + PLANET_RADIUS_EPS_KM);
    let up = pos_km / radius;
    let sun_cos = max(dot(up, sun_dir), 0.0);
    if (sun_cos <= 0.0) {
        return vec4<f32>(0.0);
    }

    let bottom = bottom_radius_km();
    let ground_pos = up * (bottom + PLANET_RADIUS_EPS_KM);
    let ground_to_sample_distance = max(radius - (bottom + PLANET_RADIUS_EPS_KM), 0.0);
    let horizon = sqrt(max(radius * radius - bottom * bottom, 0.0));
    let planet_solid_angle = 2.0 * PI * (1.0 - horizon / radius);

    let sun_ground_tau = optical_depth_from_columns(
        column_to_top(ground_pos, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM),
        column_to_top(ground_pos, sun_dir, MIE_EQUIVALENT_HEIGHT_KM),
        ozone_column_to_top(ground_pos, sun_dir),
    );
    let ground_to_sample_r = radial_quadratic_density_column_segment(
        ground_pos,
        up,
        ground_to_sample_distance,
        RAYLEIGH_EQUIVALENT_HEIGHT_KM,
    );
    let ground_to_sample_m = radial_quadratic_density_column_segment(
        ground_pos,
        up,
        ground_to_sample_distance,
        MIE_EQUIVALENT_HEIGHT_KM,
    );
    let ground_to_sample_o = ozone_triangle_column_segment(ground_pos, up, ground_to_sample_distance);
    let ground_to_sample_tau = optical_depth_from_columns(
        ground_to_sample_r,
        ground_to_sample_m,
        ground_to_sample_o,
    );

    return vec4<f32>(INV_4PI * GROUND_ALBEDO * INV_PI * planet_solid_angle * sun_cos)
        * exp(-(sun_ground_tau + ground_to_sample_tau));
}

fn ground_direct_irradiance_transfer(ground_pos_km: vec3<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> vec4<f32> {
    let sun_cos = max(dot(normal, sun_dir), 0.0);
    if (sun_cos <= 0.0) {
        return vec4<f32>(0.0);
    }
    let sun_tau = optical_depth_from_columns(
        column_to_top(ground_pos_km, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM),
        column_to_top(ground_pos_km, sun_dir, MIE_EQUIVALENT_HEIGHT_KM),
        ozone_column_to_top(ground_pos_km, sun_dir),
    );
    return vec4<f32>(sun_cos) * exp(-sun_tau);
}

fn ground_sky_irradiance_transfer_approx(ground_pos_km: vec3<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> vec4<f32> {
    let sun_cos = max(dot(normal, sun_dir), 0.0);
    if (sun_cos <= 0.0) {
        return vec4<f32>(0.0);
    }

    let t_top = ray_sphere_intersection(ground_pos_km, normal, top_radius_km());
    if (t_top <= 0.0) {
        return vec4<f32>(0.0);
    }

    let p_mid = ground_pos_km + normal * (SEGMENT_MIDPOINT_U * t_top);
    let p_end = ground_pos_km + normal * t_top;
    let view_col_r = column_to_top(ground_pos_km, normal, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let view_col_m = column_to_top(ground_pos_km, normal, MIE_EQUIVALENT_HEIGHT_KM);
    let view_col_mid_r = radial_quadratic_density_column_segment(
        ground_pos_km,
        normal,
        SEGMENT_MIDPOINT_U * t_top,
        RAYLEIGH_EQUIVALENT_HEIGHT_KM,
    );
    let view_col_mid_m = radial_quadratic_density_column_segment(
        ground_pos_km,
        normal,
        SEGMENT_MIDPOINT_U * t_top,
        MIE_EQUIVALENT_HEIGHT_KM,
    );
    let view_col_o = ozone_triangle_column_segment(ground_pos_km, normal, t_top);
    let view_col_mid_o = ozone_triangle_column_segment(ground_pos_km, normal, SEGMENT_MIDPOINT_U * t_top);

    let sun_col0_r = column_to_top(ground_pos_km, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col0_m = column_to_top(ground_pos_km, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col0_o = ozone_column_to_top(ground_pos_km, sun_dir);
    let sun_col_mid_r = column_to_top(p_mid, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_m = column_to_top(p_mid, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_o = ozone_column_to_top(p_mid, sun_dir);
    let sun_col1_r = column_to_top(p_end, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col1_m = column_to_top(p_end, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col1_o = ozone_column_to_top(p_end, sun_dir);

    let tau_s0 = optical_depth_from_columns(sun_col0_r, sun_col0_m, sun_col0_o);
    let tau_mid = optical_depth_from_columns(sun_col_mid_r, sun_col_mid_m, sun_col_mid_o)
        + optical_depth_from_columns(view_col_mid_r, view_col_mid_m, view_col_mid_o);
    let tau_s1_v = optical_depth_from_columns(sun_col1_r, sun_col1_m, sun_col1_o)
        + optical_depth_from_columns(view_col_r, view_col_m, view_col_o);
    let avg_t_r = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_r / max(view_col_r, COLUMN_RATIO_EPS),
    );
    let avg_t_m = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_m / max(view_col_m, COLUMN_RATIO_EPS),
    );

    // Low-order diffuse skylight estimate for visible Lambert ground. The
    // aerosol term intentionally uses an isotropic phase instead of the OPAC
    // forward-lobe hack, because this approximates hemispherical irradiance,
    // not a single view ray through the sun aureole.
    let rayleigh = ap.rayleigh_scattering_base * view_col_r * rayleigh_phase(GROUND_SKY_RAYLEIGH_PHASE_MU_SCALE * sun_cos) * avg_t_r;
    let mie = ap.mie_scattering_base * view_col_m * INV_4PI * avg_t_m;
    return PI * (rayleigh + mie);
}

fn view_segment_scatter_to_ground(
    origin: vec3<f32>,
    ground_pos_km: vec3<f32>,
    sun_dir: vec3<f32>,
) -> vec4<f32> {
    let ray_dir = normalize(ground_pos_km - origin);
    let distance = length(ground_pos_km - origin);
    if (distance <= DISTANCE_EPS_KM) {
        return vec4<f32>(0.0);
    }

    let p_mid = origin + ray_dir * (SEGMENT_MIDPOINT_U * distance);
    let view_col_r = radial_quadratic_density_column_segment(origin, ray_dir, distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let view_col_m = radial_quadratic_density_column_segment(origin, ray_dir, distance, MIE_EQUIVALENT_HEIGHT_KM);
    let view_col_mid_r = radial_quadratic_density_column_segment(origin, ray_dir, SEGMENT_MIDPOINT_U * distance, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let view_col_mid_m = radial_quadratic_density_column_segment(origin, ray_dir, SEGMENT_MIDPOINT_U * distance, MIE_EQUIVALENT_HEIGHT_KM);
    let view_col_o = ozone_triangle_column_segment(origin, ray_dir, distance);
    let view_col_mid_o = ozone_triangle_column_segment(origin, ray_dir, SEGMENT_MIDPOINT_U * distance);

    let sun_col0_r = column_to_top(origin, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col0_m = column_to_top(origin, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col0_o = ozone_column_to_top(origin, sun_dir);
    let sun_col_mid_r = column_to_top(p_mid, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_m = column_to_top(p_mid, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_o = ozone_column_to_top(p_mid, sun_dir);
    let sun_col1_r = column_to_top(ground_pos_km, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col1_m = column_to_top(ground_pos_km, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col1_o = ozone_column_to_top(ground_pos_km, sun_dir);

    let tau_s0 = optical_depth_from_columns(sun_col0_r, sun_col0_m, sun_col0_o);
    let tau_view_mid = optical_depth_from_columns(view_col_mid_r, view_col_mid_m, view_col_mid_o);
    let tau_view1 = optical_depth_from_columns(view_col_r, view_col_m, view_col_o);
    let tau_mid = optical_depth_from_columns(sun_col_mid_r, sun_col_mid_m, sun_col_mid_o) + tau_view_mid;
    let tau_s1_v = optical_depth_from_columns(sun_col1_r, sun_col1_m, sun_col1_o) + tau_view1;
    let avg_t_r = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_r / max(view_col_r, COLUMN_RATIO_EPS),
    );
    let avg_t_m = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_m / max(view_col_m, COLUMN_RATIO_EPS),
    );

    let mu = dot(sun_dir, ray_dir);
    let rayleigh_scatter = ap.rayleigh_scattering_base * view_col_r * rayleigh_phase(mu) * avg_t_r;
    let mie_scatter = ap.mie_scattering_base * view_col_m * opac_like_mie_phase_hack(mu) * avg_t_m;
    return ap.sun_spectral_irradiance * (rayleigh_scatter + mie_scatter);
}

fn analytic_ground_radiance(origin: vec3<f32>, dir: vec3<f32>, t_ground: f32, sun_dir: vec3<f32>) -> vec3<f32> {
    let ground_hit = origin + dir * t_ground;
    let normal = normalize(ground_hit);
    let ground_pos = normal * (bottom_radius_km() + PLANET_RADIUS_EPS_KM);
    let view_to_eye = normalize(origin - ground_pos);
    let view_distance = length(origin - ground_pos);

    let direct_transfer = ground_direct_irradiance_transfer(ground_pos, normal, sun_dir);
    let sky_transfer = ground_sky_irradiance_transfer_approx(ground_pos, normal, sun_dir);
    let view_transmittance = exp(-optical_depth_segment(ground_pos, view_to_eye, view_distance));
    let ground_spectral = ap.sun_spectral_irradiance
        * (direct_transfer + sky_transfer)
        * vec4<f32>(GROUND_ALBEDO * INV_PI)
        * view_transmittance;
    let view_scatter_spectral = view_segment_scatter_to_ground(origin, ground_pos, sun_dir);
    return max(white_balanced_linear_rec2020_from_spectral(ground_spectral + view_scatter_spectral), vec3<f32>(0.0));
}

// Error functions and closed-form transmittance averages.
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
    if (x < ERFI_SERIES_SWITCH_X) {
        let xx = x * x;
        var term = x;
        var sum = x;
        for (var i = 1u; i <= ERFI_SERIES_TERMS; i = i + 1u) {
            let n = f32(i);
            term *= xx * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
            sum += term;
        }
        return s * TWO_INV_SQRT_PI * sum;
    }

    let inv_x2 = 1.0 / max(x * x, COLUMN_RATIO_EPS);
    let series = 1.0
        + 0.5 * inv_x2
        + 0.75 * inv_x2 * inv_x2
        + 1.875 * inv_x2 * inv_x2 * inv_x2
        + 6.5625 * inv_x2 * inv_x2 * inv_x2 * inv_x2;
    return s * exp(min(x * x, OPTICAL_DEPTH_CLAMP)) * series / (SQRT_PI * max(x, MIN_SCALE_HEIGHT_KM));
}

fn average_transmittance_linear_scalar(tau0: f32, tau1: f32) -> f32 {
    let d = tau1 - tau0;
    if (abs(d) < TRANSMITTANCE_LINEAR_TAU_EPS) {
        return exp(-0.5 * (tau0 + tau1));
    }
    return (exp(-tau0) - exp(-tau1)) / d;
}

fn average_transmittance_quadratic_scalar(tau0: f32, tau_mid: f32, tau1: f32, u_mid_in: f32) -> f32 {
    if (tau_mid > OPTICAL_DEPTH_CLAMP) {
        return 0.0;
    }
    if (tau0 > OPTICAL_DEPTH_CLAMP || tau1 > OPTICAL_DEPTH_CLAMP) {
        return average_transmittance_linear_scalar(tau0, tau1);
    }
    if (u_mid_in <= TRANSMITTANCE_QUADRATIC_U_MIN || u_mid_in >= TRANSMITTANCE_QUADRATIC_U_MAX) {
        return average_transmittance_linear_scalar(tau0, tau1);
    }

    let u_mid = clamp(u_mid_in, TRANSMITTANCE_QUADRATIC_U_MIN, TRANSMITTANCE_QUADRATIC_U_MAX);
    let d = tau1 - tau0;
    // tau(u) = tau0 + d * u + q * u * (1 - u). q is estimated from one
    // closed-form midpoint. Positive q means the optical depth arches above
    // the endpoint line, which is the problematic low-sun horizon case.
    let q = (tau_mid - (tau0 + d * u_mid)) / max(u_mid * (1.0 - u_mid), MIN_SCALE_HEIGHT_KM);
    if (abs(q) < TRANSMITTANCE_QUADRATIC_Q_EPS) {
        return average_transmittance_linear_scalar(tau0, tau1);
    }

    let b = d + q;
    if (q > 0.0) {
        let sqrt_q = sqrt(q);
        let x0 = -b / (2.0 * sqrt_q);
        let x1 = sqrt_q + x0;
        let scale = exp(clamp(-tau0 - b * b / (4.0 * q), -OPTICAL_DEPTH_CLAMP, OPTICAL_DEPTH_CLAMP));
        let integral = scale * SQRT_PI * (erfi_approx(x1) - erfi_approx(x0)) / (2.0 * sqrt_q);
        return max(integral, 0.0);
    }

    let p = -q;
    let sqrt_p = sqrt(p);
    let x0 = b / (2.0 * sqrt_p);
    let x1 = sqrt_p + x0;
    let scale = exp(clamp(-tau0 + b * b / (4.0 * p), -OPTICAL_DEPTH_CLAMP, OPTICAL_DEPTH_CLAMP));
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

// Sun disk and spectral display conversion.
fn spectral_sun_transmittance_to_rec2020(transmittance: vec4<f32>) -> vec3<f32> {
    let clear_sun = max(white_balanced_linear_rec2020_from_spectral(ap.sun_spectral_irradiance), vec3<f32>(COLUMN_RATIO_EPS));
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

// Main analytic sky/ground evaluation.
fn analytic_sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    let origin = vec3<f32>(0.0, eye_radius_km(), 0.0);
    let dir = normalize(direction);
    let sun_dir = to_sun_dir();
    let t_ground = eye_ground_intersection(dir);
    if (t_ground >= 0.0) {
        return analytic_ground_radiance(origin, dir, t_ground, sun_dir);
    }

    let t_top = ray_sphere_intersection(origin, dir, top_radius_km());
    if (t_top < 0.0) {
        return vec3<f32>(0.0);
    }

    let p_end = origin + dir * t_top;
    let p_mid = origin + dir * (SEGMENT_MIDPOINT_U * t_top);
    let view_col_r = column_to_top(origin, dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let view_col_m = column_to_top(origin, dir, MIE_EQUIVALENT_HEIGHT_KM);
    let view_col_mid_r = radial_quadratic_density_column_segment(
        origin,
        dir,
        SEGMENT_MIDPOINT_U * t_top,
        RAYLEIGH_EQUIVALENT_HEIGHT_KM,
    );
    let view_col_mid_m = radial_quadratic_density_column_segment(
        origin,
        dir,
        SEGMENT_MIDPOINT_U * t_top,
        MIE_EQUIVALENT_HEIGHT_KM,
    );
    let sun_col0_r = column_to_top(origin, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col0_m = column_to_top(origin, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col0_o = ozone_column_to_top(origin, sun_dir);
    let sun_col_mid_r = column_to_top(p_mid, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_m = column_to_top(p_mid, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col_mid_o = ozone_column_to_top(p_mid, sun_dir);
    let sun_col1_r = column_to_top(p_end, sun_dir, RAYLEIGH_EQUIVALENT_HEIGHT_KM);
    let sun_col1_m = column_to_top(p_end, sun_dir, MIE_EQUIVALENT_HEIGHT_KM);
    let sun_col1_o = ozone_column_to_top(p_end, sun_dir);
    let view_col_o = ozone_triangle_column_segment(origin, dir, t_top);
    let view_col_mid_o = ozone_triangle_column_segment(origin, dir, SEGMENT_MIDPOINT_U * t_top);

    let tau_s0 = optical_depth_from_columns(sun_col0_r, sun_col0_m, sun_col0_o);
    let tau_view_mid = optical_depth_from_columns(view_col_mid_r, view_col_mid_m, view_col_mid_o);
    let tau_view1 = optical_depth_from_columns(view_col_r, view_col_m, view_col_o);
    let tau_mid = optical_depth_from_columns(sun_col_mid_r, sun_col_mid_m, sun_col_mid_o) + tau_view_mid;
    let tau_s1_v = optical_depth_from_columns(sun_col1_r, sun_col1_m, sun_col1_o) + tau_view1;
    let avg_t_r = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_r / max(view_col_r, COLUMN_RATIO_EPS),
    );
    let avg_t_m = average_transmittance_quadratic(
        tau_s0,
        tau_mid,
        tau_s1_v,
        view_col_mid_m / max(view_col_m, COLUMN_RATIO_EPS),
    );
    let avg_view_t_r = average_transmittance_quadratic(
        vec4<f32>(0.0),
        tau_view_mid,
        tau_view1,
        view_col_mid_r / max(view_col_r, COLUMN_RATIO_EPS),
    );
    let avg_view_t_m = average_transmittance_quadratic(
        vec4<f32>(0.0),
        tau_view_mid,
        tau_view1,
        view_col_mid_m / max(view_col_m, COLUMN_RATIO_EPS),
    );
    let mu = dot(sun_dir, dir);
    let rayleigh_scatter = ap.rayleigh_scattering_base * view_col_r * rayleigh_phase(mu) * avg_t_r;
    let mie_scatter = ap.mie_scattering_base * view_col_m * opac_like_mie_phase_hack(mu) * avg_t_m;
    let ground_transfer = ground_bounce_transfer(p_mid, sun_dir);
    let ground_scatter = ground_transfer
        * (ap.rayleigh_scattering_base * view_col_r * avg_view_t_r
            + ap.mie_scattering_base * view_col_m * avg_view_t_m);
    let sky_spectral = ap.sun_spectral_irradiance * (rayleigh_scatter + mie_scatter + ground_scatter);
    var rgb = max(white_balanced_linear_rec2020_from_spectral(sky_spectral), vec3<f32>(0.0));

    let sun_t = spectral_sun_transmittance_to_rec2020(exp(-tau_s0));
    rgb += sun_disk_eval(dir, sun_t);
    return rgb;
}

// Spectral-to-Rec.2020 conversion and white balance.
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
