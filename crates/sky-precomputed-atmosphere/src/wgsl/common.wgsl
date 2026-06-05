// Bruneton/E.B. precomputed atmosphere helpers adapted to four spectral lanes.
// The LUT parameterization and integration structure follow the reference
// implementation; atmosphere, aerosol and phase data come from this project.

const PI: f32 = 3.141592653589793;
const INV_PI: f32 = 0.31830988618379067154;
const TRANSMITTANCE_TEXTURE_WIDTH: f32 = 256.0;
const TRANSMITTANCE_TEXTURE_HEIGHT: f32 = 64.0;
const SCATTERING_TEXTURE_R_SIZE: f32 = 32.0;
const SCATTERING_TEXTURE_MU_SIZE: f32 = 128.0;
const SCATTERING_TEXTURE_MU_S_SIZE: f32 = 32.0;
const SCATTERING_TEXTURE_NU_SIZE: f32 = 8.0;
const SCATTERING_TEXTURE_WIDTH: f32 = 256.0;
const SCATTERING_TEXTURE_HEIGHT: f32 = 128.0;
const SCATTERING_TEXTURE_DEPTH: f32 = 32.0;
const IRRADIANCE_TEXTURE_WIDTH: f32 = 64.0;
const IRRADIANCE_TEXTURE_HEIGHT: f32 = 16.0;
const SPECIES_COUNT: u32 = 3u;

struct Species {
    sigma_sca: vec4<f32>,
    sigma_abs: vec4<f32>,
    base_density: f32,
    bg_density: f32,
    height_scale: f32,
    _pad: f32,
}

struct PrecomputedParams {
    earth_radius_km: f32,
    atmosphere_thickness_km: f32,
    eye_distance_to_earth_center_km: f32,
    eye_altitude_km: f32,
    sun_dir: vec3<f32>,
    sun_angular_radius_rad: f32,
    sun_spectral_irradiance: vec4<f32>,
    molecular_scattering_base: vec4<f32>,
    ozone_absorption_cross_section: vec4<f32>,
    species: array<Species, 3>,
    turbidity: f32,
    ozone_mean_dobson: f32,
    mu_s_min: f32,
    mie_phase_mode: u32,
    ground_albedo_spectral: vec4<f32>,
}

struct OrderParams {
    scattering_order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct RuntimeView {
    relative_world_from_clip: mat4x4<f32>,
    world_position: vec4<f32>,
}

fn top_radius_km() -> f32 {
    return params.earth_radius_km + params.atmosphere_thickness_km;
}

fn clamp_cosine(mu: f32) -> f32 {
    return clamp(mu, -1.0, 1.0);
}

fn clamp_radius(r: f32) -> f32 {
    return clamp(r, params.earth_radius_km, top_radius_km());
}

fn safe_sqrt(a: f32) -> f32 {
    return sqrt(max(a, 0.0));
}

fn texture_coord_from_unit_range(x: f32, size: f32) -> f32 {
    return 0.5 / size + x * (1.0 - 1.0 / size);
}

fn unit_range_from_texture_coord(u: f32, size: f32) -> f32 {
    return clamp((u - 0.5 / size) / (1.0 - 1.0 / size), 0.0, 1.0);
}

fn distance_to_top_atmosphere_boundary(r: f32, mu: f32) -> f32 {
    let top = top_radius_km();
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    return max(-r * mu + safe_sqrt(discriminant), 0.0);
}

fn distance_to_bottom_atmosphere_boundary(r: f32, mu: f32) -> f32 {
    let bottom = params.earth_radius_km;
    let discriminant = r * r * (mu * mu - 1.0) + bottom * bottom;
    return max(-r * mu - safe_sqrt(discriminant), 0.0);
}

fn ray_intersects_ground(r: f32, mu: f32) -> bool {
    let bottom = params.earth_radius_km;
    return mu < 0.0 && r * r * (mu * mu - 1.0) + bottom * bottom >= 0.0;
}

fn distance_to_nearest_atmosphere_boundary(r: f32, mu: f32, hits_ground: bool) -> f32 {
    if (hits_ground) {
        return distance_to_bottom_atmosphere_boundary(r, mu);
    }
    return distance_to_top_atmosphere_boundary(r, mu);
}

fn molecular_density(h: f32) -> f32 {
    return exp(-0.07771971 * pow(max(h, 0.0), 1.16364243));
}

fn molecular_scattering(h: f32) -> vec4<f32> {
    return params.molecular_scattering_base * molecular_density(h);
}

fn molecular_absorption(h_in: f32) -> vec4<f32> {
    let h = max(h_in, 0.0) + 1.0e-4;
    let t = log(h) - 3.22261;
    let density = 3.78547397e20 * (1.0 / h) * exp(-t * t * 5.55555555);
    return params.ozone_absorption_cross_section * params.ozone_mean_dobson * density;
}

fn species_density(species: Species, h: f32) -> f32 {
    let scale = max(species.height_scale, 1.0e-3);
    let base = species.base_density * exp(-h / scale);
    let bg = species.bg_density * exp(-h / scale);
    let t = smoothstep(1.0, 2.0, h);
    return max(mix(base, bg, t), 0.0);
}

fn species_scattering(species_index: u32, h: f32) -> vec4<f32> {
    let species = params.species[species_index];
    return species.sigma_sca * species_density(species, h) * params.turbidity;
}

fn species_absorption(species_index: u32, h: f32) -> vec4<f32> {
    let species = params.species[species_index];
    return species.sigma_abs * species_density(species, h) * params.turbidity;
}

fn aerosol_scattering_total(h: f32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < SPECIES_COUNT; k = k + 1u) {
        sum += species_scattering(k, h);
    }
    return sum;
}

fn aerosol_absorption_total(h: f32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < SPECIES_COUNT; k = k + 1u) {
        sum += species_absorption(k, h);
    }
    return sum;
}

fn extinction(h: f32) -> vec4<f32> {
    return molecular_scattering(h) + molecular_absorption(h)
        + aerosol_scattering_total(h) + aerosol_absorption_total(h);
}

fn rayleigh_phase(nu: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + nu * nu);
}

fn phase_lut_uv(mu: f32) -> f32 {
    return pow(max(0.5 - 0.5 * clamp(mu, -1.0, 1.0), 0.0), 1.0 / 3.0);
}

fn aerosol_phase_at(
    phase_lut: texture_2d_array<f32>,
    samp: sampler,
    species: u32,
    forward_cos: f32,
) -> vec4<f32> {
    return textureSampleLevel(
        phase_lut,
        samp,
        vec2<f32>(phase_lut_uv(forward_cos), 0.5),
        i32(species),
        0.0,
    );
}

fn aerosol_phase_weighted_scattering(
    phase_lut: texture_2d_array<f32>,
    samp: sampler,
    h: f32,
    nu: f32,
) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < SPECIES_COUNT; k = k + 1u) {
        sum += species_scattering(k, h) * aerosol_phase_at(phase_lut, samp, k, nu);
    }
    return sum;
}

fn transmittance_uv_from_r_mu(r: f32, mu: f32) -> vec2<f32> {
    let bottom = params.earth_radius_km;
    let top = top_radius_km();
    let h = safe_sqrt(top * top - bottom * bottom);
    let rho = safe_sqrt(r * r - bottom * bottom);
    let d = distance_to_top_atmosphere_boundary(r, mu);
    let d_min = top - r;
    let d_max = rho + h;
    let x_mu = (d - d_min) / max(d_max - d_min, 1.0e-6);
    let x_r = rho / max(h, 1.0e-6);
    return vec2<f32>(
        texture_coord_from_unit_range(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
        texture_coord_from_unit_range(x_r, TRANSMITTANCE_TEXTURE_HEIGHT),
    );
}

fn r_mu_from_transmittance_uv(uv: vec2<f32>) -> vec2<f32> {
    let x_mu = unit_range_from_texture_coord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    let x_r = unit_range_from_texture_coord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    let bottom = params.earth_radius_km;
    let top = top_radius_km();
    let h = safe_sqrt(top * top - bottom * bottom);
    let rho = h * x_r;
    let r = safe_sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + x_mu * (d_max - d_min);
    let mu = select((h * h - rho * rho - d * d) / max(2.0 * r * d, 1.0e-6), 1.0, d == 0.0);
    return vec2<f32>(r, clamp_cosine(mu));
}

fn compute_transmittance_to_top(r: f32, mu: f32) -> vec4<f32> {
    let sample_count: u32 = 500u;
    let dx = distance_to_top_atmosphere_boundary(r, mu) / f32(sample_count);
    var optical_depth = vec4<f32>(0.0);
    for (var i: u32 = 0u; i <= sample_count; i = i + 1u) {
        let d_i = f32(i) * dx;
        let r_i = safe_sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        let h_i = r_i - params.earth_radius_km;
        let weight = select(1.0, 0.5, i == 0u || i == sample_count);
        optical_depth += extinction(h_i) * weight * dx;
    }
    return exp(-optical_depth);
}

fn sample_transmittance_to_top(
    tex: texture_2d<f32>,
    samp: sampler,
    r: f32,
    mu: f32,
) -> vec4<f32> {
    return textureSampleLevel(tex, samp, transmittance_uv_from_r_mu(r, mu), 0.0);
}

fn get_transmittance(
    tex: texture_2d<f32>,
    samp: sampler,
    r: f32,
    mu: f32,
    d: f32,
    hits_ground: bool,
) -> vec4<f32> {
    let r_d = clamp_radius(safe_sqrt(d * d + 2.0 * r * mu * d + r * r));
    let mu_d = clamp_cosine((r * mu + d) / max(r_d, 1.0e-6));
    if (hits_ground) {
        return min(
            sample_transmittance_to_top(tex, samp, r_d, -mu_d)
                / max(sample_transmittance_to_top(tex, samp, r, -mu), vec4<f32>(1.0e-6)),
            vec4<f32>(1.0),
        );
    }
    return min(
        sample_transmittance_to_top(tex, samp, r, mu)
            / max(sample_transmittance_to_top(tex, samp, r_d, mu_d), vec4<f32>(1.0e-6)),
        vec4<f32>(1.0),
    );
}

fn get_transmittance_to_sun(
    tex: texture_2d<f32>,
    samp: sampler,
    r: f32,
    mu_s: f32,
) -> vec4<f32> {
    let sin_theta_h = params.earth_radius_km / max(r, 1.0e-6);
    let cos_theta_h = -safe_sqrt(1.0 - sin_theta_h * sin_theta_h);
    let horizon_blend = smoothstep(
        -sin_theta_h * params.sun_angular_radius_rad,
        sin_theta_h * params.sun_angular_radius_rad,
        mu_s - cos_theta_h,
    );
    return sample_transmittance_to_top(tex, samp, r, mu_s) * horizon_blend;
}

fn scattering_uvwz_from_r_mu_mu_s_nu(
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    hits_ground: bool,
) -> vec4<f32> {
    let bottom = params.earth_radius_km;
    let top = top_radius_km();
    let h = safe_sqrt(top * top - bottom * bottom);
    let rho = safe_sqrt(r * r - bottom * bottom);
    let u_r = texture_coord_from_unit_range(rho / max(h, 1.0e-6), SCATTERING_TEXTURE_R_SIZE);

    let r_mu = r * mu;
    let discriminant = r_mu * r_mu - r * r + bottom * bottom;
    var u_mu: f32;
    if (hits_ground) {
        let d = -r_mu - safe_sqrt(discriminant);
        let d_min = r - bottom;
        let d_max = rho;
        let x = select((d - d_min) / max(d_max - d_min, 1.0e-6), 0.0, d_max == d_min);
        u_mu = 0.5 - 0.5 * texture_coord_from_unit_range(x, SCATTERING_TEXTURE_MU_SIZE * 0.5);
    } else {
        let d = -r_mu + safe_sqrt(discriminant + h * h);
        let d_min = top - r;
        let d_max = rho + h;
        u_mu = 0.5 + 0.5 * texture_coord_from_unit_range(
            (d - d_min) / max(d_max - d_min, 1.0e-6),
            SCATTERING_TEXTURE_MU_SIZE * 0.5,
        );
    }

    let d_s = distance_to_top_atmosphere_boundary(bottom, mu_s);
    let d_min_s = top - bottom;
    let d_max_s = h;
    let a = (d_s - d_min_s) / max(d_max_s - d_min_s, 1.0e-6);
    let d_min_limit = distance_to_top_atmosphere_boundary(bottom, params.mu_s_min);
    let a_limit = (d_min_limit - d_min_s) / max(d_max_s - d_min_s, 1.0e-6);
    let u_mu_s = texture_coord_from_unit_range(
        max(1.0 - a / max(a_limit, 1.0e-6), 0.0) / (1.0 + a),
        SCATTERING_TEXTURE_MU_S_SIZE,
    );
    let u_nu = clamp(nu * 0.5 + 0.5, 0.0, 1.0);
    return vec4<f32>(u_nu, u_mu_s, u_mu, u_r);
}

struct ScatteringParams {
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    hits_ground: bool,
}

fn scattering_params_from_frag_coord(frag_coord: vec3<f32>) -> ScatteringParams {
    let size = vec4<f32>(
        SCATTERING_TEXTURE_NU_SIZE - 1.0,
        SCATTERING_TEXTURE_MU_S_SIZE,
        SCATTERING_TEXTURE_MU_SIZE,
        SCATTERING_TEXTURE_R_SIZE,
    );
    let frag_coord_nu = floor(frag_coord.x / SCATTERING_TEXTURE_MU_S_SIZE);
    let frag_coord_mu_s = frag_coord.x - frag_coord_nu * SCATTERING_TEXTURE_MU_S_SIZE;
    let uvwz = vec4<f32>(
        frag_coord_nu,
        frag_coord_mu_s,
        frag_coord.y,
        frag_coord.z,
    ) / size;

    let bottom = params.earth_radius_km;
    let top = top_radius_km();
    let h = safe_sqrt(top * top - bottom * bottom);
    let rho = h * unit_range_from_texture_coord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
    let r = safe_sqrt(rho * rho + bottom * bottom);

    var mu: f32;
    var hits_ground: bool;
    if (uvwz.z < 0.5) {
        let d_min = r - bottom;
        let d_max = rho;
        let d = d_min + (d_max - d_min) * unit_range_from_texture_coord(
            1.0 - 2.0 * uvwz.z,
            SCATTERING_TEXTURE_MU_SIZE * 0.5,
        );
        mu = select(-(rho * rho + d * d) / max(2.0 * r * d, 1.0e-6), -1.0, d == 0.0);
        hits_ground = true;
    } else {
        let d_min = top - r;
        let d_max = rho + h;
        let d = d_min + (d_max - d_min) * unit_range_from_texture_coord(
            2.0 * uvwz.z - 1.0,
            SCATTERING_TEXTURE_MU_SIZE * 0.5,
        );
        mu = select((h * h - rho * rho - d * d) / max(2.0 * r * d, 1.0e-6), 1.0, d == 0.0);
        hits_ground = false;
    }
    mu = clamp_cosine(mu);

    let x_mu_s = unit_range_from_texture_coord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
    let d_min_s = top - bottom;
    let d_max_s = h;
    let d_limit = distance_to_top_atmosphere_boundary(bottom, params.mu_s_min);
    let a_limit = (d_limit - d_min_s) / max(d_max_s - d_min_s, 1.0e-6);
    let a = (a_limit - x_mu_s * a_limit) / max(1.0 + x_mu_s * a_limit, 1.0e-6);
    let d_s = d_min_s + min(a, a_limit) * (d_max_s - d_min_s);
    var mu_s = select((h * h - d_s * d_s) / max(2.0 * bottom * d_s, 1.0e-6), 1.0, d_s == 0.0);
    mu_s = clamp_cosine(mu_s);

    var nu = clamp_cosine(uvwz.x * 2.0 - 1.0);
    let nu_min = mu * mu_s - safe_sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s));
    let nu_max = mu * mu_s + safe_sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s));
    nu = clamp(nu, nu_min, nu_max);

    return ScatteringParams(r, mu, mu_s, nu, hits_ground);
}

fn sample_scattering(
    tex: texture_3d<f32>,
    samp: sampler,
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    hits_ground: bool,
) -> vec4<f32> {
    let uvwz = scattering_uvwz_from_r_mu_mu_s_nu(r, mu, mu_s, nu, hits_ground);
    let tex_coord_x = uvwz.x * (SCATTERING_TEXTURE_NU_SIZE - 1.0);
    let tex_x = floor(tex_coord_x);
    let lerp = tex_coord_x - tex_x;
    let uvw0 = vec3<f32>(
        (tex_x + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE,
        uvwz.z,
        uvwz.w,
    );
    let uvw1 = vec3<f32>(
        (tex_x + 1.0 + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE,
        uvwz.z,
        uvwz.w,
    );
    return textureSampleLevel(tex, samp, uvw0, 0.0) * (1.0 - lerp)
        + textureSampleLevel(tex, samp, uvw1, 0.0) * lerp;
}

fn sample_single_scattering(
    phase_lut: texture_2d_array<f32>,
    samp: sampler,
    molecular_tex: texture_3d<f32>,
    aerosol0_tex: texture_3d<f32>,
    aerosol1_tex: texture_3d<f32>,
    aerosol2_tex: texture_3d<f32>,
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    hits_ground: bool,
) -> vec4<f32> {
    let molecular = sample_scattering(molecular_tex, samp, r, mu, mu_s, nu, hits_ground)
        * rayleigh_phase(nu);
    let aerosol0 = sample_scattering(aerosol0_tex, samp, r, mu, mu_s, nu, hits_ground)
        * aerosol_phase_at(phase_lut, samp, 0u, nu);
    let aerosol1 = sample_scattering(aerosol1_tex, samp, r, mu, mu_s, nu, hits_ground)
        * aerosol_phase_at(phase_lut, samp, 1u, nu);
    let aerosol2 = sample_scattering(aerosol2_tex, samp, r, mu, mu_s, nu, hits_ground)
        * aerosol_phase_at(phase_lut, samp, 2u, nu);
    return molecular + aerosol0 + aerosol1 + aerosol2;
}

fn irradiance_uv_from_r_mu_s(r: f32, mu_s: f32) -> vec2<f32> {
    let x_r = (r - params.earth_radius_km) / max(params.atmosphere_thickness_km, 1.0e-6);
    let x_mu_s = mu_s * 0.5 + 0.5;
    return vec2<f32>(
        texture_coord_from_unit_range(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
        texture_coord_from_unit_range(x_r, IRRADIANCE_TEXTURE_HEIGHT),
    );
}

fn r_mu_s_from_irradiance_uv(uv: vec2<f32>) -> vec2<f32> {
    let x_mu_s = unit_range_from_texture_coord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    let x_r = unit_range_from_texture_coord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    return vec2<f32>(
        params.earth_radius_km + x_r * params.atmosphere_thickness_km,
        clamp_cosine(2.0 * x_mu_s - 1.0),
    );
}

fn sample_irradiance(tex: texture_2d<f32>, samp: sampler, r: f32, mu_s: f32) -> vec4<f32> {
    return textureSampleLevel(tex, samp, irradiance_uv_from_r_mu_s(r, mu_s), 0.0);
}

fn linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    let m = mat4x3<f32>(
        vec3<f32>(83.460, 1.554, -0.043),
        vec3<f32>(49.968, 86.062, -2.182),
        vec3<f32>(-11.823, 29.205, 29.153),
        vec3<f32>(6.811, -8.283, 104.377),
    );
    return m * l;
}

fn white_balance_rec2020(rgb: vec3<f32>) -> vec3<f32> {
    return rgb * vec3<f32>(0.9441, 0.9888, 1.0761);
}

fn white_balanced_linear_rec2020_from_spectral(l: vec4<f32>) -> vec3<f32> {
    return white_balance_rec2020(linear_rec2020_from_spectral(l));
}
