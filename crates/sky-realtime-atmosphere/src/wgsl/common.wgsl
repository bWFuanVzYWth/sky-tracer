// Hillaire spectral atmosphere shared helpers.
//
// Length is in km. The aerosol model keeps only the three dominant species
// used by this project: waso, inso and soot.

const ATM_PI: f32 = 3.141592653589793;
const ATM_INV_PI: f32 = 0.31830988618379067154;
const ATM_INV_4PI: f32 = 0.07957747154594767;
const ATM_PHASE_ISOTROPIC: f32 = 0.07957747154594767;
const ATM_RAYLEIGH_PHASE_SCALE: f32 = 0.05968310365946075;
const ATM_CS_G: f32 = 0.8;
const ATM_MIE_PHASE_MODE_CS: u32 = 1u;
const ATM_NUM_AEROSOL_SPECIES: u32 = 3u;
const ATM_SUN_COS_THETA_MAX: f32 = 0.999989;

struct HillaireSpecies {
    sigma_sca: vec4<f32>,
    sigma_abs: vec4<f32>,
    base_density: f32,
    bg_density: f32,
    height_scale: f32,
    _pad: f32,
}

struct HillaireParams {
    earth_radius_km: f32,
    atmosphere_thickness_km: f32,
    eye_distance_to_earth_center_km: f32,
    eye_altitude_km: f32,
    sun_dir: vec3<f32>,
    sky_view_height_km: f32,
    sun_spectral_irradiance: vec4<f32>,
    molecular_scattering_base: vec4<f32>,
    ozone_absorption_cross_section: vec4<f32>,
    species: array<HillaireSpecies, 3>,
    turbidity: f32,
    ozone_mean_dobson: f32,
    mie_phase_mode: u32,
    _pad_misc1: u32,
    ground_albedo_spectral: vec4<f32>,
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

struct AtmPoint {
    radius_km: f32,
    altitude_km: f32,
    local_pos_km: vec3<f32>,
    up: vec3<f32>,
}

struct AtmRay {
    origin: AtmPoint,
    dir: vec3<f32>,
}

struct AtmRaySegment {
    t_top_km: f32,
    t_ground_km: f32,
    t_max_km: f32,
    hits_ground: bool,
    exits_atmosphere: bool,
}

fn atm_top_radius_km() -> f32 {
    return hp.earth_radius_km + hp.atmosphere_thickness_km;
}

fn atm_clamp_radius_km(radius_km: f32) -> f32 {
    return clamp(radius_km, hp.earth_radius_km + 1.0e-3, atm_top_radius_km() - 1.0e-3);
}

fn atm_point_from_local_pos_km(local_pos_km: vec3<f32>) -> AtmPoint {
    let radius_km = max(length(local_pos_km), 1.0e-6);
    let up = local_pos_km / radius_km;
    return AtmPoint(
        radius_km,
        radius_km - hp.earth_radius_km,
        local_pos_km,
        up,
    );
}

fn atm_point_from_radius_km(radius_km: f32) -> AtmPoint {
    let r = atm_clamp_radius_km(radius_km);
    return atm_point_from_local_pos_km(vec3<f32>(0.0, r, 0.0));
}

fn atm_ray_from_point(origin: AtmPoint, dir: vec3<f32>) -> AtmRay {
    return AtmRay(origin, normalize(dir));
}

fn atm_ray_segment(ray: AtmRay) -> AtmRaySegment {
    let t_top = ray_sphere_intersection(ray.origin.local_pos_km, ray.dir, atm_top_radius_km());
    let t_ground = ray_sphere_intersection(ray.origin.local_pos_km, ray.dir, hp.earth_radius_km);
    let hits_ground = t_ground >= 0.0;
    let t_max = select(t_top, t_ground, hits_ground);
    return AtmRaySegment(
        t_top,
        t_ground,
        t_max,
        hits_ground,
        !hits_ground && t_top >= 0.0,
    );
}

fn atm_ray_above_horizon(ray: AtmRay) -> bool {
    return !atm_ray_segment(ray).hits_ground;
}

fn molecular_phase_function(cos_theta: f32) -> f32 {
    return ATM_RAYLEIGH_PHASE_SCALE * (1.0 + cos_theta * cos_theta);
}

fn cornette_shanks_phase(g: f32, cos_theta: f32) -> f32 {
    let k = 3.0 / (8.0 * ATM_PI) * (1.0 - g * g) / (2.0 + g * g);
    let denom = pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1.0e-6), 1.5);
    return k * (1.0 + cos_theta * cos_theta) / denom;
}

fn get_molecular_scattering_coefficient(p: HillaireParams, h: f32) -> vec4<f32> {
    return p.molecular_scattering_base * exp(-0.07771971 * pow(h, 1.16364243));
}

fn get_molecular_absorption_coefficient(p: HillaireParams, h_in: f32) -> vec4<f32> {
    let h = h_in + 1.0e-4;
    let t = log(h) - 3.22261;
    let density = 3.78547397e20 * (1.0 / h) * exp(-t * t * 5.55555555);
    return p.ozone_absorption_cross_section * p.ozone_mean_dobson * density;
}

fn get_species_density(species: HillaireSpecies, h: f32) -> f32 {
    if (species.base_density <= 0.0 && species.bg_density <= 0.0) {
        return 0.0;
    }
    let scale = max(species.height_scale, 1.0e-3);
    let bl = species.base_density * exp(-h / scale);
    let ft = species.bg_density * exp(-h / scale);
    let t = smoothstep(1.0, 2.0, h);
    return mix(bl, ft, t);
}

struct AerosolCoeffs {
    scattering: vec4<f32>,
    absorption: vec4<f32>,
    density: f32,
}

fn get_species_coeffs(p: HillaireParams, species_idx: u32, h: f32) -> AerosolCoeffs {
    let species = p.species[species_idx];
    let n = get_species_density(species, h);
    let weight = n * p.turbidity;
    return AerosolCoeffs(
        species.sigma_sca * weight,
        species.sigma_abs * weight,
        n,
    );
}

struct AtmCoeffs {
    aerosol_scattering: vec4<f32>,
    aerosol_absorption: vec4<f32>,
    molecular_scattering: vec4<f32>,
    molecular_absorption: vec4<f32>,
    extinction: vec4<f32>,
}

fn get_atmosphere_collision_coefficients(p: HillaireParams, h_in: f32) -> AtmCoeffs {
    let h = max(h_in, 0.0);
    var aerosol_sca = vec4<f32>(0.0);
    var aerosol_abs = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
        let c = get_species_coeffs(p, k, h);
        aerosol_sca += c.scattering;
        aerosol_abs += c.absorption;
    }
    let mol_sca = get_molecular_scattering_coefficient(p, h);
    let mol_abs = get_molecular_absorption_coefficient(p, h);
    return AtmCoeffs(
        aerosol_sca,
        aerosol_abs,
        mol_sca,
        mol_abs,
        aerosol_sca + aerosol_abs + mol_sca + mol_abs,
    );
}

fn transmittance_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    cos_theta: f32,
    normalized_altitude: f32,
) -> vec4<f32> {
    let r_km = hp.earth_radius_km
        + clamp(normalized_altitude, 0.0, 1.0) * hp.atmosphere_thickness_km;
    let mu = clamp(cos_theta, -1.0, 1.0);
    let bottom = hp.earth_radius_km;
    let top = bottom + hp.atmosphere_thickness_km;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = sqrt(max(r_km * r_km - bottom * bottom, 0.0));
    let discriminant = r_km * r_km * (mu * mu - 1.0) + top * top;
    let d = max(0.0, -r_km * mu + sqrt(max(discriminant, 0.0)));
    let d_min = top - r_km;
    let d_max = rho + h;
    let x_mu = clamp((d - d_min) / max(d_max - d_min, 1.0e-6), 0.0, 1.0);
    let x_r = clamp(rho / max(h, 1.0e-6), 0.0, 1.0);
    return textureSampleLevel(lut, samp, vec2<f32>(x_mu, x_r), 0.0);
}

fn get_multiple_scattering_analytical(
    transmittance_lut: texture_2d<f32>,
    samp: sampler,
    cos_theta: f32,
    normalized_height: f32,
    d: f32,
    ground_up_transmittance: vec4<f32>,
) -> vec4<f32> {
    let earth_radius = hp.earth_radius_km;
    let omega = 2.0 * ATM_PI
        * (1.0 - sqrt(max(d * d - earth_radius * earth_radius, 0.0)) / max(d, 1.0e-6));

    let t_to_ground = transmittance_from_lut(transmittance_lut, samp, cos_theta, 0.0);
    let t_ground_to_sample = ground_up_transmittance
        / max(transmittance_from_lut(transmittance_lut, samp, 1.0, normalized_height), vec4<f32>(1.0e-6));

    let l_ground = ATM_PHASE_ISOTROPIC
        * omega
        * (hp.ground_albedo_spectral * ATM_INV_PI)
        * t_to_ground
        * t_ground_to_sample
        * cos_theta;
    let l_ms = 0.02 * vec4<f32>(0.217, 0.347, 0.594, 1.0)
        * (1.0 / (1.0 + 5.0 * exp(-17.92 * cos_theta)));

    return l_ms + l_ground;
}

fn get_multiple_scattering(
    transmittance_lut: texture_2d<f32>,
    samp: sampler,
    cos_theta: f32,
    normalized_height: f32,
    d: f32,
    ground_up_transmittance: vec4<f32>,
) -> vec4<f32> {
    return get_multiple_scattering_analytical(
        transmittance_lut,
        samp,
        cos_theta,
        normalized_height,
        d,
        ground_up_transmittance,
    );
}

fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    let m = mat4x3<f32>(
        vec3<f32>(83.460, 1.554, -0.043),
        vec3<f32>(49.968, 86.062, -2.182),
        vec3<f32>(-11.823, 29.205, 29.153),
        vec3<f32>(6.811, -8.283, 104.377),
    );
    return (m * l) * vec3<f32>(0.9441, 0.9888, 1.0761);
}

fn atm_from_unit_to_sub_uvs(u: f32, resolution: f32) -> f32 {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

fn atm_from_sub_uvs_to_unit(u: f32, resolution: f32) -> f32 {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

fn sky_view_height_km() -> f32 {
    return atm_clamp_radius_km(hp.sky_view_height_km);
}

fn sky_view_horizon_angles(view_height: f32) -> vec2<f32> {
    let bottom = hp.earth_radius_km;
    let v_horizon = sqrt(max(view_height * view_height - bottom * bottom, 0.0));
    let cos_beta = clamp(v_horizon / max(view_height, 1.0e-6), 0.0, 1.0);
    let beta = acos(cos_beta);
    let zenith_horizon_angle = ATM_PI - beta;
    return vec2<f32>(zenith_horizon_angle, beta);
}

fn sky_view_params_to_uv(
    view_zenith_cos_angle: f32,
    light_view_cos_angle: f32,
    intersect_ground: bool,
    dims: vec2<f32>,
) -> vec2<f32> {
    let angles = sky_view_horizon_angles(sky_view_height_km());
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
        atm_from_unit_to_sub_uvs(u, dims.x),
        atm_from_unit_to_sub_uvs(v, dims.y),
    );
}

fn sky_view_uv_to_params(uv_in: vec2<f32>, dims: vec2<f32>) -> vec2<f32> {
    let uv = vec2<f32>(
        clamp(atm_from_sub_uvs_to_unit(uv_in.x, dims.x), 0.0, 1.0),
        clamp(atm_from_sub_uvs_to_unit(uv_in.y, dims.y), 0.0, 1.0),
    );
    let angles = sky_view_horizon_angles(sky_view_height_km());
    let zenith_horizon_angle = angles.x;
    let beta = angles.y;

    var view_zenith_cos_angle: f32;
    if (uv.y < 0.5) {
        var coord = 2.0 * uv.y;
        coord = 1.0 - coord;
        coord = coord * coord;
        coord = 1.0 - coord;
        view_zenith_cos_angle = cos(zenith_horizon_angle * coord);
    } else {
        var coord = uv.y * 2.0 - 1.0;
        coord = coord * coord;
        view_zenith_cos_angle = cos(zenith_horizon_angle + beta * coord);
    }

    var coord = uv.x;
    coord = coord * coord;
    let light_view_cos_angle = -(coord * 2.0 - 1.0);
    return vec2<f32>(view_zenith_cos_angle, light_view_cos_angle);
}

fn sky_view_dir_from_params(params: vec2<f32>) -> vec3<f32> {
    let view_zenith_cos_angle = clamp(params.x, -1.0, 1.0);
    let light_view_cos_angle = clamp(params.y, -1.0, 1.0);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sun_h = hp.sun_dir - up * dot(hp.sun_dir, up);
    let sun_h_len = length(sun_h);
    let forward = sun_h / max(sun_h_len, 1.0e-5);
    let forward_safe = select(vec3<f32>(1.0, 0.0, 0.0), forward, sun_h_len > 1.0e-5);
    let side = normalize(cross(up, forward_safe));

    let sin_view = sqrt(max(1.0 - view_zenith_cos_angle * view_zenith_cos_angle, 0.0));
    let side_scale = sqrt(max(1.0 - light_view_cos_angle * light_view_cos_angle, 0.0));
    return normalize(
        up * view_zenith_cos_angle
            + sin_view * (forward_safe * light_view_cos_angle + side * side_scale)
    );
}

fn sky_view_uv_from_dir(dir_in: vec3<f32>, dims: vec2<f32>) -> vec2<f32> {
    let dir = normalize(dir_in);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_zenith_cos_angle = dot(dir, up);

    let view_h = dir - up * view_zenith_cos_angle;
    let sun_h = hp.sun_dir - up * dot(hp.sun_dir, up);
    let view_h_len = length(view_h);
    let sun_h_len = length(sun_h);
    let view_h_dir = view_h / max(view_h_len, 1.0e-5);
    let sun_h_dir = sun_h / max(sun_h_len, 1.0e-5);
    let light_view_cos_angle = select(
        1.0,
        dot(view_h_dir, sun_h_dir),
        view_h_len > 1.0e-5 && sun_h_len > 1.0e-5,
    );

    let origin = atm_point_from_radius_km(sky_view_height_km());
    let intersect_ground = atm_ray_segment(atm_ray_from_point(origin, dir)).hits_ground;
    return sky_view_params_to_uv(
        view_zenith_cos_angle,
        light_view_cos_angle,
        intersect_ground,
        dims,
    );
}
