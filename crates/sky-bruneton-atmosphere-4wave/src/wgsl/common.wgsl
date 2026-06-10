const ATM_PI: f32 = 3.141592653589793;
const ATM_INV_PI: f32 = 0.31830988618379067154;
const ATM_INV_4PI: f32 = 0.07957747154594767;
const ATM_PHASE_ISOTROPIC: f32 = 0.07957747154594767;
const ATM_RAYLEIGH_PHASE_SCALE: f32 = 0.05968310365946075;
const ATM_CS_G: f32 = 0.8;
const ATM_MIE_PHASE_MODE_CS: u32 = 1u;
const ATM_NUM_AEROSOL_SPECIES: u32 = 3u;
const ATM_SUN_COS_THETA_MAX: f32 = 0.999989;
const ATM_SUN_ANGULAR_RADIUS_RAD: f32 = 0.00471;
const ATM_PLANET_RADIUS_OFFSET_KM: f32 = 0.01;
const ATM_SKY_VIEW_SKY_FRACTION: f32 = 0.75;
const ATM_SKY_VIEW_GROUND_FRACTION: f32 = 0.25;

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

struct RuntimeView {
    relative_world_from_clip: mat4x4<f32>,
    world_position: vec4<f32>,
}

fn atm_from_unit_to_sub_uvs(u: f32, resolution: f32) -> f32 {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

fn atm_from_sub_uvs_to_unit(u: f32, resolution: f32) -> f32 {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
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
    return AtmPoint(radius_km, radius_km - hp.earth_radius_km, local_pos_km, up);
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
    return AtmRaySegment(t_top, t_ground, t_max, hits_ground, !hits_ground && t_top >= 0.0);
}

fn move_to_top_atmosphere(pos_in: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let view_height = length(pos_in);
    if (view_height <= atm_top_radius_km()) {
        return pos_in;
    }
    let t_top = ray_sphere_intersection(pos_in, dir, atm_top_radius_km());
    if (t_top < 0.0) {
        return pos_in;
    }
    let up = pos_in / max(view_height, 1.0e-6);
    return pos_in + dir * t_top - up * ATM_PLANET_RADIUS_OFFSET_KM;
}

fn molecular_phase_function(cos_theta: f32) -> f32 {
    return ATM_RAYLEIGH_PHASE_SCALE * (1.0 + cos_theta * cos_theta);
}

fn cornette_shanks_phase(g: f32, cos_theta: f32) -> f32 {
    let k = 3.0 / (8.0 * ATM_PI) * (1.0 - g * g) / (2.0 + g * g);
    let denom = pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1.0e-6), 1.5);
    return k * (1.0 + cos_theta * cos_theta) / denom;
}

fn sun_disc_average_cosine_factor(mu_s: f32) -> f32 {
    let alpha = ATM_SUN_ANGULAR_RADIUS_RAD;
    if (mu_s <= -alpha) {
        return 0.0;
    }
    if (mu_s >= alpha) {
        return mu_s;
    }
    let visible = mu_s + alpha;
    return visible * visible / (4.0 * alpha);
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
    return AerosolCoeffs(species.sigma_sca * weight, species.sigma_abs * weight, n);
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
    return AtmCoeffs(aerosol_sca, aerosol_abs, mol_sca, mol_abs, aerosol_sca + aerosol_abs + mol_sca + mol_abs);
}

fn transmittance_uv_from_params(cos_theta: f32, normalized_altitude: f32) -> vec2<f32> {
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
    return vec2<f32>(x_mu, x_r);
}

fn transmittance_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    cos_theta: f32,
    normalized_altitude: f32,
) -> vec4<f32> {
    return textureSampleLevel(lut, samp, transmittance_uv_from_params(cos_theta, normalized_altitude), 0.0);
}

fn multi_scattering_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    sun_zenith_cos_angle: f32,
    normalized_altitude: f32,
) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(lut));
    let uv_unit = vec2<f32>(
        clamp(sun_zenith_cos_angle * 0.5 + 0.5, 0.0, 1.0),
        clamp(normalized_altitude, 0.0, 1.0),
    );
    let uv = vec2<f32>(
        atm_from_unit_to_sub_uvs(uv_unit.x, dims.x),
        atm_from_unit_to_sub_uvs(uv_unit.y, dims.y),
    );
    return textureSampleLevel(lut, samp, uv, 0.0);
}

fn ground_irradiance_uv_from_r_mu_s(r: f32, mu_s: f32, dims: vec2<f32>) -> vec2<f32> {
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let x_mu_s = clamp(mu_s * 0.5 + 0.5, 0.0, 1.0);
    let x_r = clamp((r - bottom) / max(top - bottom, 1.0e-6), 0.0, 1.0);
    return vec2<f32>(
        atm_from_unit_to_sub_uvs(x_mu_s, dims.x),
        atm_from_unit_to_sub_uvs(x_r, dims.y),
    );
}

fn ground_irradiance_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    r: f32,
    mu_s: f32,
) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(lut));
    let uv = ground_irradiance_uv_from_r_mu_s(r, mu_s, dims);
    return textureSampleLevel(lut, samp, uv, 0.0);
}

fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    // 410/480/560/630 nm -> scene-linear Rec.2020. The columns are direct CIE 1931
    // 2 degree CMF samples transformed to Rec.2020 and multiplied by the
    // cmf-mass fixed-sum quadrature weights:
    // 176.941/85.8237/76.2946/70.9408 nm.
    let m = mat4x3<f32>(
        vec3<f32>(3.841910401, -4.207847549, 34.699506104),
        vec3<f32>(-7.830465365, 14.914489331, 65.365373942),
        vec3<f32>(50.786945459, 92.47799295, -2.16643693),
        vec3<f32>(71.544621425, 0.006405143, 0.003173681),
    );
    return m * l;
}

fn white_balance_rec2020(rgb: vec3<f32>) -> vec3<f32> {
    // Bradford 41-band solar-white-to-D65 adaptation expressed in Rec.2020 RGB.
    // This matches the offline reference colorimetry instead of neutralizing
    // the sparse 410/480/560/630 nm solar samples.
    let m = mat3x3<f32>(
        vec3<f32>(0.973450179, -0.00110537346, 0.000549268697),
        vec3<f32>(-0.0199533690, 1.01471724, -0.000413338668),
        vec3<f32>(0.000904216012, -0.000617879953, 1.06404848),
    );
    return m * rgb;
}

fn white_balanced_linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    return white_balance_rec2020(linear_rec2020_from_spectral(l));
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
        v = coord * ATM_SKY_VIEW_SKY_FRACTION;
    } else {
        var coord = clamp((view_angle - zenith_horizon_angle) / beta, 0.0, 1.0);
        coord = sqrt(max(coord, 0.0));
        v = coord * ATM_SKY_VIEW_GROUND_FRACTION + ATM_SKY_VIEW_SKY_FRACTION;
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
    if (uv.y < ATM_SKY_VIEW_SKY_FRACTION) {
        var coord = uv.y / ATM_SKY_VIEW_SKY_FRACTION;
        coord = 1.0 - coord;
        coord = coord * coord;
        coord = 1.0 - coord;
        view_zenith_cos_angle = cos(zenith_horizon_angle * coord);
    } else {
        var coord = (uv.y - ATM_SKY_VIEW_SKY_FRACTION) / ATM_SKY_VIEW_GROUND_FRACTION;
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
    return sky_view_params_to_uv(view_zenith_cos_angle, light_view_cos_angle, intersect_ground, dims);
}

const ATM_SCATTERING_MU_S_SIZE: u32 = 256u;
const ATM_SCATTERING_NU_SIZE: u32 = 32u;
const ATM_SCATTERING_SKY_FRACTION: f32 = 0.75;
const ATM_SCATTERING_GROUND_FRACTION: f32 = 0.25;
const ATM_SCATTERING_NU_PACK_POWER: f32 = 3.0;

struct BrunetonScatteringParams {
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
}

struct BrunetonDirPair {
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
}

fn bruneton_index_to_unit(index: u32, resolution: u32) -> f32 {
    if (resolution <= 1u) {
        return 0.0;
    }
    return f32(index) / f32(resolution - 1u);
}

fn bruneton_radius_from_unit(u: f32) -> f32 {
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let bottom2 = bottom * bottom;
    let top2 = top * top;
    return sqrt(mix(bottom2, top2, clamp(u, 0.0, 1.0)));
}

fn bruneton_unit_from_radius(r_in: f32) -> f32 {
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let r = clamp(r_in, bottom, top);
    return clamp((r * r - bottom * bottom) / max(top * top - bottom * bottom, 1.0e-6), 0.0, 1.0);
}

fn bruneton_horizon_angles(radius_km: f32) -> vec2<f32> {
    let bottom = hp.earth_radius_km;
    let v_horizon = sqrt(max(radius_km * radius_km - bottom * bottom, 0.0));
    let cos_beta = clamp(v_horizon / max(radius_km, 1.0e-6), 0.0, 1.0);
    let beta = acos(cos_beta);
    let zenith_horizon_angle = ATM_PI - beta;
    return vec2<f32>(zenith_horizon_angle, beta);
}

fn bruneton_angle_unit_from_mu(radius_km: f32, mu: f32) -> f32 {
    let angles = bruneton_horizon_angles(radius_km);
    let zenith_horizon_angle = max(angles.x, 1.0e-6);
    let beta = max(angles.y, 1.0e-6);
    let view_angle = acos(clamp(mu, -1.0, 1.0));

    if (view_angle <= zenith_horizon_angle) {
        var coord = clamp(view_angle / zenith_horizon_angle, 0.0, 1.0);
        coord = 1.0 - coord;
        coord = sqrt(max(coord, 0.0));
        coord = 1.0 - coord;
        return coord * ATM_SCATTERING_SKY_FRACTION;
    }

    var coord = clamp((view_angle - zenith_horizon_angle) / beta, 0.0, 1.0);
    coord = sqrt(max(coord, 0.0));
    return coord * ATM_SCATTERING_GROUND_FRACTION + ATM_SCATTERING_SKY_FRACTION;
}

fn bruneton_mu_from_angle_unit(radius_km: f32, unit_in: f32) -> f32 {
    let unit = clamp(unit_in, 0.0, 1.0);
    let angles = bruneton_horizon_angles(radius_km);
    let zenith_horizon_angle = angles.x;
    let beta = angles.y;

    if (unit < ATM_SCATTERING_SKY_FRACTION) {
        var coord = unit / ATM_SCATTERING_SKY_FRACTION;
        coord = 1.0 - coord;
        coord = coord * coord;
        coord = 1.0 - coord;
        return cos(zenith_horizon_angle * coord);
    }

    var coord = (unit - ATM_SCATTERING_SKY_FRACTION) / ATM_SCATTERING_GROUND_FRACTION;
    coord = coord * coord;
    return cos(zenith_horizon_angle + beta * coord);
}

fn bruneton_nu_range_from_mu_mu_s(mu: f32, mu_s: f32) -> vec2<f32> {
    let mu_c = clamp(mu, -1.0, 1.0);
    let mu_s_c = clamp(mu_s, -1.0, 1.0);
    let sin_mu = sqrt(max(1.0 - mu_c * mu_c, 0.0));
    let sin_mu_s = sqrt(max(1.0 - mu_s_c * mu_s_c, 0.0));
    let center = mu_c * mu_s_c;
    let radius = sin_mu * sin_mu_s;
    return vec2<f32>(
        clamp(center - radius, -1.0, 1.0),
        clamp(center + radius, -1.0, 1.0),
    );
}

fn bruneton_unit_from_valid_nu(nu: f32, mu: f32, mu_s: f32) -> f32 {
    let range = bruneton_nu_range_from_mu_mu_s(mu, mu_s);
    let width = max(range.y - range.x, 1.0e-6);
    let high_to_low = clamp((range.y - clamp(nu, range.x, range.y)) / width, 0.0, 1.0);
    return pow(high_to_low, 1.0 / ATM_SCATTERING_NU_PACK_POWER);
}

fn bruneton_valid_nu_from_unit(unit: f32, mu: f32, mu_s: f32) -> f32 {
    let range = bruneton_nu_range_from_mu_mu_s(mu, mu_s);
    let u = clamp(unit, 0.0, 1.0);
    return range.y - (range.y - range.x) * u * u * u;
}

fn bruneton_dirs_from_params(mu: f32, mu_s: f32, nu: f32) -> BrunetonDirPair {
    let mu_c = clamp(mu, -1.0, 1.0);
    let mu_s_c = clamp(mu_s, -1.0, 1.0);
    let sin_mu = sqrt(max(1.0 - mu_c * mu_c, 0.0));
    let sin_mu_s = sqrt(max(1.0 - mu_s_c * mu_s_c, 0.0));
    let sun_dir = normalize(vec3<f32>(sin_mu_s, mu_s_c, 0.0));

    var cos_phi = 1.0;
    if (sin_mu * sin_mu_s > 1.0e-5) {
        cos_phi = clamp((clamp(nu, -1.0, 1.0) - mu_c * mu_s_c) / (sin_mu * sin_mu_s), -1.0, 1.0);
    }
    let sin_phi = sqrt(max(1.0 - cos_phi * cos_phi, 0.0));
    let ray_dir = normalize(vec3<f32>(sin_mu * cos_phi, mu_c, sin_mu * sin_phi));
    return BrunetonDirPair(ray_dir, sun_dir);
}

fn bruneton_scattering_params_from_texel(gid: vec3<u32>, dims_u: vec3<u32>) -> BrunetonScatteringParams {
    let mu_s_size = ATM_SCATTERING_MU_S_SIZE;
    let nu_size = ATM_SCATTERING_NU_SIZE;
    let mu_s_index = gid.x % mu_s_size;
    let nu_index = gid.x / mu_s_size;

    let r = bruneton_radius_from_unit(bruneton_index_to_unit(gid.z, dims_u.z));
    let mu = bruneton_mu_from_angle_unit(r, bruneton_index_to_unit(gid.y, dims_u.y));
    let mu_s = bruneton_mu_from_angle_unit(r, bruneton_index_to_unit(mu_s_index, mu_s_size));
    let nu = bruneton_valid_nu_from_unit(bruneton_index_to_unit(nu_index, nu_size), mu, mu_s);
    let dirs = bruneton_dirs_from_params(mu, mu_s, nu);
    let valid_nu = dot(dirs.ray_dir, dirs.sun_dir);
    return BrunetonScatteringParams(r, mu, mu_s, valid_nu, dirs.ray_dir, dirs.sun_dir);
}

fn bruneton_scattering_uv_for_nu_slice(
    r: f32,
    mu: f32,
    mu_s: f32,
    nu_slice: u32,
    dims_u: vec3<u32>,
) -> vec3<f32> {
    let mu_s_size = ATM_SCATTERING_MU_S_SIZE;
    let width = f32(dims_u.x);
    let height = f32(dims_u.y);
    let depth = f32(dims_u.z);
    let mu_unit = bruneton_angle_unit_from_mu(r, mu);
    let mu_s_unit = bruneton_angle_unit_from_mu(r, mu_s);
    let r_unit = bruneton_unit_from_radius(r);
    let x_texel = f32(nu_slice * mu_s_size) + 0.5 + mu_s_unit * f32(mu_s_size - 1u);
    let y_texel = 0.5 + mu_unit * f32(dims_u.y - 1u);
    let z_texel = 0.5 + r_unit * f32(dims_u.z - 1u);
    return vec3<f32>(x_texel / width, y_texel / height, z_texel / depth);
}

fn bruneton_scattering_from_lut(
    lut: texture_3d<f32>,
    samp: sampler,
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
) -> vec4<f32> {
    let dims_u = textureDimensions(lut);
    let nu_unit = bruneton_unit_from_valid_nu(nu, mu, mu_s);
    let nu_f = clamp(nu_unit * f32(ATM_SCATTERING_NU_SIZE - 1u), 0.0, f32(ATM_SCATTERING_NU_SIZE - 1u));
    let nu_slice = u32(min(floor(nu_f + 0.5), f32(ATM_SCATTERING_NU_SIZE - 1u)));
    let uv = bruneton_scattering_uv_for_nu_slice(r, mu, mu_s, nu_slice, dims_u);
    return textureSampleLevel(lut, samp, uv, 0.0);
}

fn bruneton_ground_irradiance_params_from_uv(uv_in: vec2<f32>, dims: vec2<f32>) -> vec2<f32> {
    let uv = vec2<f32>(
        clamp(atm_from_sub_uvs_to_unit(uv_in.x, dims.x), 0.0, 1.0),
        clamp(atm_from_sub_uvs_to_unit(uv_in.y, dims.y), 0.0, 1.0),
    );
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let r = mix(bottom, top, uv.y);
    let mu_s = bruneton_mu_from_angle_unit(r, uv.x);
    return vec2<f32>(r, mu_s);
}

fn bruneton_ground_irradiance_uv_from_r_mu_s(r: f32, mu_s: f32, dims: vec2<f32>) -> vec2<f32> {
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let r_unit = clamp((r - bottom) / max(top - bottom, 1.0e-6), 0.0, 1.0);
    let mu_s_unit = bruneton_angle_unit_from_mu(r, mu_s);
    return vec2<f32>(
        atm_from_unit_to_sub_uvs(mu_s_unit, dims.x),
        atm_from_unit_to_sub_uvs(r_unit, dims.y),
    );
}

fn bruneton_ground_irradiance_from_lut(
    lut: texture_2d<f32>,
    samp: sampler,
    r: f32,
    mu_s: f32,
) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(lut));
    let uv = bruneton_ground_irradiance_uv_from_r_mu_s(r, mu_s, dims);
    return textureSampleLevel(lut, samp, uv, 0.0);
}
