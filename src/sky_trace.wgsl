const BAND_COUNT: u32 = 15u;
const SPECIES_COUNT: u32 = 4u;
const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const TAU: f32 = 6.28318530717958647692;
const RAY_EPSILON_KM: f32 = 0.0001;
const GROUND_ALBEDO: f32 = 0.3;
const ISOTROPIC_PDF: f32 = 0.07957747154594766788;

struct Constants {
    width: u32,
    height: u32,
    spp: u32,
    direct_light_samples: u32,
    sample_offset: u32,
    samples_this_dispatch: u32,
    tile_y: u32,
    tile_height: u32,
    atmosphere_len: u32,
    aerosol_len: u32,
    majorant_layers: u32,
    phase_bins: u32,
    seed_lo: u32,
    seed_hi: u32,
    watchdog_limit: u32,
    _pad0: u32,
    _pad1: u32,
    observer_altitude_km: f32,
    ground_radius_km: f32,
    atmosphere_radius_km: f32,
    top_altitude_km: f32,
    sun_x: f32,
    sun_y: f32,
    sun_z: f32,
    sun_angular_radius_rad: f32,
    sun_solid_angle_sr: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

struct Band {
    center_nm: f32,
    lower_nm: f32,
    upper_nm: f32,
    ozone_cross_section_cm2: f32,
    solar_radiance_w_m2_sr: f32,
    rayleigh_cross_section_m2: f32,
}

struct AtmospherePoint {
    altitude_km: f32,
    temperature_k: f32,
    air_cm3: f32,
    ozone_cm3: f32,
}

struct AerosolPoint {
    altitude_km: f32,
    mass_0: f32,
    mass_1: f32,
    mass_2: f32,
    mass_3: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct AerosolOptics {
    scattering_km_inv_per_g_m3: f32,
    absorption_km_inv_per_g_m3: f32,
}

struct Diagnostics {
    watchdog: atomic<u32>,
}

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

struct BoundaryHit {
    hit: bool,
    t_km: f32,
    kind: u32,
}

struct SphereHit {
    hit: bool,
    t0: f32,
    t1: f32,
}

struct MediumCoefficients {
    rayleigh_scattering_km_inv: f32,
    aerosol_scattering_km_inv: vec4f,
    aerosol_absorption_km_inv: vec4f,
    ozone_absorption_km_inv: f32,
}

struct CollisionSample {
    hit: bool,
    t_km: f32,
    coeffs: MediumCoefficients,
}

struct AtmosphereSample {
    temperature_k: f32,
    air_cm3: f32,
    ozone_cm3: f32,
}

struct AerosolSample {
    mass_g_m3: vec4f,
}

struct DirectionPdf {
    dir: vec3f,
    pdf: f32,
}

struct Basis {
    tangent: vec3f,
    bitangent: vec3f,
}

struct Rng {
    state: u32,
}

struct TraceResult {
    radiance: f32,
    watchdog: bool,
}

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read> bands: array<Band>;
@group(0) @binding(2) var<storage, read> atmosphere_profile: array<AtmospherePoint>;
@group(0) @binding(3) var<storage, read> aerosol_profile: array<AerosolPoint>;
@group(0) @binding(4) var<storage, read> aerosol_optics: array<AerosolOptics>;
@group(0) @binding(5) var<storage, read> majorants_km_inv: array<f32>;
@group(0) @binding(6) var<storage, read> phase_values: array<f32>;
@group(0) @binding(7) var<storage, read_write> film: array<f32>;
@group(0) @binding(8) var<storage, read_write> diagnostics: Diagnostics;

fn hash32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn make_rng(pixel: u32, sample: u32, band: u32) -> Rng {
    let seed = constants.seed_lo
        ^ hash32(constants.seed_hi)
        ^ hash32(pixel * 0x9e3779b9u)
        ^ hash32(sample * 0x85ebca6bu)
        ^ hash32(band * 0xc2b2ae35u);
    return Rng(hash32(seed));
}

fn rng_next_u32(rng: ptr<function, Rng>) -> u32 {
    (*rng).state = (*rng).state * 1664525u + 1013904223u;
    return hash32((*rng).state);
}

fn rng_f32(rng: ptr<function, Rng>) -> f32 {
    let value = rng_next_u32(rng) >> 8u;
    return f32(value) * 0.000000059604644775390625;
}

fn normalize_or_y(v: vec3f) -> vec3f {
    let len = length(v);
    if len > 0.0 {
        return v / len;
    }
    return vec3f(0.0, 1.0, 0.0);
}

fn ray_at(ray: Ray, t_km: f32) -> vec3f {
    return ray.origin + ray.dir * t_km;
}

fn altitude_km(position: vec3f) -> f32 {
    return length(position) - constants.ground_radius_km;
}

fn sun_dir() -> vec3f {
    return normalize_or_y(vec3f(constants.sun_x, constants.sun_y, constants.sun_z));
}

fn intersect_sphere(ray: Ray, radius: f32) -> SphereHit {
    let b = dot(ray.origin, ray.dir);
    let c = dot(ray.origin, ray.origin) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return SphereHit(false, 0.0, 0.0);
    }
    let s = sqrt(discriminant);
    return SphereHit(true, -b - s, -b + s);
}

fn positive_root(t0: f32, t1: f32) -> f32 {
    if t0 > RAY_EPSILON_KM {
        return t0;
    }
    if t1 > RAY_EPSILON_KM {
        return t1;
    }
    return -1.0;
}

fn next_boundary(ray: Ray) -> BoundaryHit {
    let atmosphere_hit = intersect_sphere(ray, constants.atmosphere_radius_km);
    if !atmosphere_hit.hit {
        return BoundaryHit(false, 0.0, 0u);
    }

    let ground_hit = intersect_sphere(ray, constants.ground_radius_km);
    var t_ground = -1.0;
    if ground_hit.hit {
        t_ground = positive_root(ground_hit.t0, ground_hit.t1);
    }

    let t_exit = max(atmosphere_hit.t1, RAY_EPSILON_KM);
    if t_ground > RAY_EPSILON_KM && t_ground < t_exit {
        return BoundaryHit(true, t_ground, 0u);
    }
    return BoundaryHit(true, t_exit, 1u);
}

fn layer_for_altitude(altitude: f32) -> u32 {
    let top = max(constants.top_altitude_km, 0.000001);
    let clamped_altitude = clamp(altitude, 0.0, top - 0.000001);
    let layer = u32(floor(clamped_altitude / top * f32(constants.majorant_layers)));
    return min(layer, constants.majorant_layers - 1u);
}

fn majorant_for_layer(band: u32, layer: u32) -> f32 {
    return max(majorants_km_inv[band * constants.majorant_layers + layer], 0.00000001);
}

fn boundary_candidate(ray: Ray, t: f32, current_next: f32, altitude_boundary: f32) -> f32 {
    if altitude_boundary <= 0.0 || altitude_boundary >= constants.top_altitude_km {
        return current_next;
    }

    let radius = constants.ground_radius_km + altitude_boundary;
    let hit = intersect_sphere(ray, radius);
    if !hit.hit {
        return current_next;
    }

    var next_t = current_next;
    if hit.t0 > t + RAY_EPSILON_KM && hit.t0 < next_t {
        next_t = hit.t0;
    }
    if hit.t1 > t + RAY_EPSILON_KM && hit.t1 < next_t {
        next_t = hit.t1;
    }
    return next_t;
}

fn next_majorant_segment_end(ray: Ray, t: f32, t_max: f32) -> f32 {
    let probe_t = min(t + RAY_EPSILON_KM, t_max);
    let layer = layer_for_altitude(altitude_km(ray_at(ray, probe_t)));
    let dz = constants.top_altitude_km / f32(constants.majorant_layers);
    let lo = f32(layer) * dz;
    let hi = f32(layer + 1u) * dz;
    var next_t = t_max;
    next_t = boundary_candidate(ray, t, next_t, lo);
    next_t = boundary_candidate(ray, t, next_t, hi);
    return next_t;
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

fn interpolate_atmosphere(altitude: f32) -> AtmosphereSample {
    var hi = 0u;
    loop {
        if hi >= constants.atmosphere_len || atmosphere_profile[hi].altitude_km >= altitude {
            break;
        }
        hi = hi + 1u;
    }

    if hi == 0u {
        let p = atmosphere_profile[0u];
        return AtmosphereSample(p.temperature_k, p.air_cm3, p.ozone_cm3);
    }
    if hi >= constants.atmosphere_len {
        let p = atmosphere_profile[constants.atmosphere_len - 1u];
        return AtmosphereSample(p.temperature_k, p.air_cm3, p.ozone_cm3);
    }

    let lo = atmosphere_profile[hi - 1u];
    let high = atmosphere_profile[hi];
    let denom = max(high.altitude_km - lo.altitude_km, 0.000001);
    let t = clamp((altitude - lo.altitude_km) / denom, 0.0, 1.0);
    return AtmosphereSample(
        lerp(lo.temperature_k, high.temperature_k, t),
        lerp(lo.air_cm3, high.air_cm3, t),
        lerp(lo.ozone_cm3, high.ozone_cm3, t),
    );
}

fn aerosol_mass(point: AerosolPoint) -> vec4f {
    return vec4f(point.mass_0, point.mass_1, point.mass_2, point.mass_3);
}

fn interpolate_aerosol(altitude: f32) -> AerosolSample {
    var hi = 0u;
    loop {
        if hi >= constants.aerosol_len || aerosol_profile[hi].altitude_km >= altitude {
            break;
        }
        hi = hi + 1u;
    }

    if hi == 0u {
        return AerosolSample(aerosol_mass(aerosol_profile[0u]));
    }
    if hi >= constants.aerosol_len {
        return AerosolSample(aerosol_mass(aerosol_profile[constants.aerosol_len - 1u]));
    }

    let lo = aerosol_profile[hi - 1u];
    let high = aerosol_profile[hi];
    let denom = max(high.altitude_km - lo.altitude_km, 0.000001);
    let t = clamp((altitude - lo.altitude_km) / denom, 0.0, 1.0);
    return AerosolSample(max(aerosol_mass(lo) * (1.0 - t) + aerosol_mass(high) * t, vec4f(0.0)));
}

fn coefficients_at(position: vec3f, band: u32) -> MediumCoefficients {
    let altitude = altitude_km(position);
    if altitude < 0.0 || altitude > constants.top_altitude_km {
        return MediumCoefficients(0.0, vec4f(0.0), vec4f(0.0), 0.0);
    }

    let atm = interpolate_atmosphere(altitude);
    let aero = interpolate_aerosol(altitude);
    let band_data = bands[band];

    let opt0 = aerosol_optics[band * SPECIES_COUNT + 0u];
    let opt1 = aerosol_optics[band * SPECIES_COUNT + 1u];
    let opt2 = aerosol_optics[band * SPECIES_COUNT + 2u];
    let opt3 = aerosol_optics[band * SPECIES_COUNT + 3u];
    let aerosol_sca = vec4f(
        aero.mass_g_m3.x * opt0.scattering_km_inv_per_g_m3,
        aero.mass_g_m3.y * opt1.scattering_km_inv_per_g_m3,
        aero.mass_g_m3.z * opt2.scattering_km_inv_per_g_m3,
        aero.mass_g_m3.w * opt3.scattering_km_inv_per_g_m3,
    );
    let aerosol_abs = vec4f(
        aero.mass_g_m3.x * opt0.absorption_km_inv_per_g_m3,
        aero.mass_g_m3.y * opt1.absorption_km_inv_per_g_m3,
        aero.mass_g_m3.z * opt2.absorption_km_inv_per_g_m3,
        aero.mass_g_m3.w * opt3.absorption_km_inv_per_g_m3,
    );

    let rayleigh = max(atm.air_cm3 * band_data.rayleigh_cross_section_m2 * 1000000000.0, 0.0);
    let ozone = max(atm.ozone_cm3 * band_data.ozone_cross_section_cm2 * 100000.0, 0.0);
    return MediumCoefficients(rayleigh, max(aerosol_sca, vec4f(0.0)), max(aerosol_abs, vec4f(0.0)), ozone);
}

fn scattering_total(coeffs: MediumCoefficients) -> f32 {
    return coeffs.rayleigh_scattering_km_inv
        + coeffs.aerosol_scattering_km_inv.x
        + coeffs.aerosol_scattering_km_inv.y
        + coeffs.aerosol_scattering_km_inv.z
        + coeffs.aerosol_scattering_km_inv.w;
}

fn extinction_total(coeffs: MediumCoefficients) -> f32 {
    return scattering_total(coeffs)
        + coeffs.ozone_absorption_km_inv
        + coeffs.aerosol_absorption_km_inv.x
        + coeffs.aerosol_absorption_km_inv.y
        + coeffs.aerosol_absorption_km_inv.z
        + coeffs.aerosol_absorption_km_inv.w;
}

fn sample_real_collision(ray: Ray, t_max: f32, band: u32, rng: ptr<function, Rng>) -> CollisionSample {
    var t = 0.0;
    var guard = 0u;
    loop {
        if t >= t_max {
            return CollisionSample(false, 0.0, MediumCoefficients(0.0, vec4f(0.0), vec4f(0.0), 0.0));
        }
        if guard > 100000u {
            atomicStore(&diagnostics.watchdog, 1u);
            return CollisionSample(false, 0.0, MediumCoefficients(0.0, vec4f(0.0), vec4f(0.0), 0.0));
        }
        guard = guard + 1u;

        let segment_end = next_majorant_segment_end(ray, t, t_max);
        let layer = layer_for_altitude(altitude_km(ray_at(ray, min(t + RAY_EPSILON_KM, segment_end))));
        let majorant = majorant_for_layer(band, layer);
        loop {
            if t >= segment_end {
                break;
            }
            let u = max(1.0 - rng_f32(rng), 0.0000001);
            t = t - log(u) / majorant;
            if t >= segment_end {
                break;
            }
            let coeffs = coefficients_at(ray_at(ray, t), band);
            let accept = clamp(extinction_total(coeffs) / majorant, 0.0, 1.0);
            if rng_f32(rng) < accept {
                return CollisionSample(true, t, coeffs);
            }
        }
        t = segment_end;
    }
    return CollisionSample(false, 0.0, MediumCoefficients(0.0, vec4f(0.0), vec4f(0.0), 0.0));
}

fn delta_transmittance(ray: Ray, t_max: f32, band: u32, rng: ptr<function, Rng>) -> f32 {
    var t = 0.0;
    var guard = 0u;
    loop {
        if t >= t_max {
            return 1.0;
        }
        if guard > 100000u {
            atomicStore(&diagnostics.watchdog, 1u);
            return 0.0;
        }
        guard = guard + 1u;

        let segment_end = next_majorant_segment_end(ray, t, t_max);
        let layer = layer_for_altitude(altitude_km(ray_at(ray, min(t + RAY_EPSILON_KM, segment_end))));
        let majorant = majorant_for_layer(band, layer);
        loop {
            if t >= segment_end {
                break;
            }
            let u = max(1.0 - rng_f32(rng), 0.0000001);
            t = t - log(u) / majorant;
            if t >= segment_end {
                break;
            }
            let coeffs = coefficients_at(ray_at(ray, t), band);
            let accept = clamp(extinction_total(coeffs) / majorant, 0.0, 1.0);
            if rng_f32(rng) < accept {
                return 0.0;
            }
        }
        t = segment_end;
    }
    return 0.0;
}

fn rayleigh_phase(mu: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

fn mie_phase(species: u32, band: u32, mu: f32) -> f32 {
    let u = pow((1.0 - clamp(mu, -1.0, 1.0)) * 0.5, 0.3333333333333333);
    let f = u * f32(constants.phase_bins) - 0.5;
    let i0 = min(u32(clamp(floor(f), 0.0, f32(constants.phase_bins - 1u))), constants.phase_bins - 1u);
    let i1 = min(i0 + 1u, constants.phase_bins - 1u);
    let t = clamp(f - f32(i0), 0.0, 1.0);
    let base = (species * BAND_COUNT + band) * constants.phase_bins;
    let p0 = phase_values[base + i0];
    let p1 = phase_values[base + i1];
    return max(lerp(p0, p1, t), 0.0);
}

fn mixed_phase_value(coeffs: MediumCoefficients, band: u32, mu: f32) -> f32 {
    let scattering = scattering_total(coeffs);
    if scattering <= 0.0 {
        return 0.0;
    }

    var phase = 0.0;
    let rayleigh_weight = coeffs.rayleigh_scattering_km_inv / scattering;
    phase = phase + rayleigh_weight * rayleigh_phase(mu);
    phase = phase + coeffs.aerosol_scattering_km_inv.x / scattering * mie_phase(0u, band, mu);
    phase = phase + coeffs.aerosol_scattering_km_inv.y / scattering * mie_phase(1u, band, mu);
    phase = phase + coeffs.aerosol_scattering_km_inv.z / scattering * mie_phase(2u, band, mu);
    phase = phase + coeffs.aerosol_scattering_km_inv.w / scattering * mie_phase(3u, band, mu);
    return phase;
}

fn orthonormal_basis(n_in: vec3f) -> Basis {
    let n = normalize_or_y(n_in);
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = normalize_or_y(vec3f(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x));
    let bitangent = normalize_or_y(vec3f(b, sign + n.y * n.y * a, -n.y));
    return Basis(tangent, bitangent);
}

fn direction_from_axis_mu(axis: vec3f, mu_in: f32, xi_phi: f32) -> vec3f {
    let mu = clamp(mu_in, -1.0, 1.0);
    let sin_theta = sqrt(max(1.0 - mu * mu, 0.0));
    let phi = TAU * xi_phi;
    let basis = orthonormal_basis(axis);
    return normalize_or_y(
        basis.tangent * (sin_theta * cos(phi))
        + basis.bitangent * (sin_theta * sin(phi))
        + normalize_or_y(axis) * mu,
    );
}

fn sample_isotropic(rng: ptr<function, Rng>) -> DirectionPdf {
    let z = 1.0 - 2.0 * rng_f32(rng);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = TAU * rng_f32(rng);
    return DirectionPdf(vec3f(r * cos(phi), z, r * sin(phi)), ISOTROPIC_PDF);
}

fn sample_uniform_cone(axis: vec3f, angular_radius: f32, rng: ptr<function, Rng>) -> DirectionPdf {
    let cos_max = cos(angular_radius);
    let cos_theta = 1.0 - clamp(rng_f32(rng), 0.0, 0.99999994) * (1.0 - cos_max);
    let pdf = 1.0 / (TAU * (1.0 - cos_max));
    return DirectionPdf(direction_from_axis_mu(axis, cos_theta, rng_f32(rng)), pdf);
}

fn sample_cosine_hemisphere(axis: vec3f, rng: ptr<function, Rng>) -> DirectionPdf {
    let u = rng_f32(rng);
    let r = sqrt(u);
    let phi = TAU * rng_f32(rng);
    let cos_theta = sqrt(max(1.0 - u, 0.0));
    let basis = orthonormal_basis(axis);
    let dir = normalize_or_y(
        basis.tangent * (r * cos(phi))
        + basis.bitangent * (r * sin(phi))
        + normalize_or_y(axis) * cos_theta,
    );
    return DirectionPdf(dir, max(dot(normalize_or_y(axis), dir), 0.0) * INV_PI);
}

fn direction_in_cone(dir: vec3f, axis: vec3f, angular_radius: f32) -> bool {
    return dot(normalize_or_y(dir), normalize_or_y(axis)) >= cos(angular_radius);
}

fn direct_sun_at_scatter(pos: vec3f, view_dir: vec3f, band: u32, coeffs: MediumCoefficients, rng: ptr<function, Rng>) -> f32 {
    var sum = 0.0;
    var sample = 0u;
    loop {
        if sample >= constants.direct_light_samples {
            break;
        }
        let light = sample_uniform_cone(sun_dir(), constants.sun_angular_radius_rad, rng);
        let shadow_ray = Ray(pos + light.dir * RAY_EPSILON_KM, light.dir);
        let boundary = next_boundary(shadow_ray);
        if boundary.hit && boundary.kind == 1u {
            let trans = delta_transmittance(shadow_ray, boundary.t_km, band, rng);
            let mu = dot(view_dir, light.dir);
            let phase = mixed_phase_value(coeffs, band, mu);
            sum = sum + bands[band].solar_radiance_w_m2_sr * trans * phase / light.pdf;
        }
        sample = sample + 1u;
    }
    return sum / f32(constants.direct_light_samples);
}

fn ground_radiance(pos: vec3f, band: u32, rng: ptr<function, Rng>) -> f32 {
    let normal = normalize_or_y(pos);
    var sum = 0.0;
    var sample = 0u;
    loop {
        if sample >= constants.direct_light_samples {
            break;
        }
        let light = sample_uniform_cone(sun_dir(), constants.sun_angular_radius_rad, rng);
        let cos_sun = max(dot(normal, light.dir), 0.0);
        if cos_sun > 0.0 {
            let shadow_ray = Ray(pos + light.dir * RAY_EPSILON_KM, light.dir);
            let boundary = next_boundary(shadow_ray);
            if boundary.hit && boundary.kind == 1u {
                let trans = delta_transmittance(shadow_ray, boundary.t_km, band, rng);
                sum = sum + GROUND_ALBEDO * INV_PI * bands[band].solar_radiance_w_m2_sr * trans * cos_sun / light.pdf;
            }
        }
        sample = sample + 1u;
    }
    return sum / f32(constants.direct_light_samples);
}

fn camera_ray(u: f32, v: f32) -> Ray {
    let azimuth = (u - 0.5) * TAU;
    let elevation = (0.5 - v) * PI;
    let cos_e = cos(elevation);
    let dir = normalize_or_y(vec3f(
        cos_e * sin(azimuth),
        sin(elevation),
        cos_e * cos(azimuth),
    ));
    let origin = vec3f(0.0, constants.ground_radius_km + max(constants.observer_altitude_km, 0.0), 0.0);
    return Ray(origin, dir);
}

fn russian_roulette(throughput: ptr<function, f32>, depth: u32, rng: ptr<function, Rng>) -> bool {
    if depth < 4u {
        return true;
    }
    let q = clamp((*throughput), 0.05, 0.95);
    if rng_f32(rng) > q {
        return false;
    }
    (*throughput) = (*throughput) / q;
    return true;
}

fn trace_path(initial_ray: Ray, band: u32, rng: ptr<function, Rng>) -> TraceResult {
    var ray = initial_ray;
    var throughput = 1.0;
    var radiance = 0.0;
    var depth = 0u;

    loop {
        if depth >= constants.watchdog_limit {
            return TraceResult(radiance, true);
        }

        let boundary = next_boundary(ray);
        if !boundary.hit {
            break;
        }

        let collision = sample_real_collision(ray, boundary.t_km, band, rng);
        if collision.hit {
            let pos = ray_at(ray, collision.t_km);
            let scattering = scattering_total(collision.coeffs);
            let extinction = extinction_total(collision.coeffs);
            if extinction <= 0.0 || scattering <= 0.0 {
                break;
            }

            throughput = throughput * clamp(scattering / extinction, 0.0, 1.0);
            if throughput <= 0.0 {
                break;
            }

            radiance = radiance + throughput * direct_sun_at_scatter(pos, ray.dir, band, collision.coeffs, rng);

            let phase_sample = sample_isotropic(rng);
            let mu = dot(ray.dir, phase_sample.dir);
            let phase = mixed_phase_value(collision.coeffs, band, mu);
            throughput = throughput * phase / phase_sample.pdf;

            if !russian_roulette(&throughput, depth, rng) {
                break;
            }

            ray = Ray(pos + phase_sample.dir * RAY_EPSILON_KM, phase_sample.dir);
        } else if boundary.kind == 1u {
            if depth == 0u && direction_in_cone(ray.dir, sun_dir(), constants.sun_angular_radius_rad) {
                radiance = radiance + throughput * bands[band].solar_radiance_w_m2_sr;
            }
            break;
        } else {
            let pos = ray_at(ray, boundary.t_km);
            radiance = radiance + throughput * ground_radiance(pos, band, rng);

            let normal = normalize_or_y(pos);
            let bounce = sample_cosine_hemisphere(normal, rng);
            let cos_out = max(dot(normal, bounce.dir), 0.0);
            if bounce.pdf <= 0.0 || cos_out <= 0.0 {
                break;
            }
            throughput = throughput * GROUND_ALBEDO * INV_PI * cos_out / bounce.pdf;

            if !russian_roulette(&throughput, depth, rng) {
                break;
            }

            ray = Ray(pos + bounce.dir * RAY_EPSILON_KM, bounce.dir);
        }

        if radiance < 0.0 || radiance != radiance || throughput < 0.0 || throughput != throughput {
            return TraceResult(0.0, false);
        }
        depth = depth + 1u;
    }

    return TraceResult(max(radiance, 0.0), false);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y + constants.tile_y;
    let band = id.z;
    if x >= constants.width || id.y >= constants.tile_height || y >= constants.height || band >= BAND_COUNT {
        return;
    }

    let pixel = y * constants.width + x;
    let output_index = pixel * BAND_COUNT + band;
    if constants.sample_offset == 0u {
        film[output_index] = 0.0;
    }

    var sum = 0.0;
    var sample = 0u;
    loop {
        if sample >= constants.samples_this_dispatch {
            break;
        }
        let sample_index = constants.sample_offset + sample;
        var rng = make_rng(pixel, sample_index, band);
        let u = (f32(x) + rng_f32(&rng)) / f32(constants.width);
        let v = (f32(y) + rng_f32(&rng)) / f32(constants.height);
        let traced = trace_path(camera_ray(u, v), band, &rng);
        if traced.watchdog {
            atomicStore(&diagnostics.watchdog, 1u);
        }
        sum = sum + traced.radiance;
        sample = sample + 1u;
    }

    film[output_index] = film[output_index] + sum;
}
