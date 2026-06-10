struct BrunetonOrder {
    order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var<uniform> bo: BrunetonOrder;
@group(0) @binding(2) var irradiance_lut: texture_2d<f32>;
@group(0) @binding(3) var scattering_lut: texture_3d<f32>;
@group(0) @binding(4) var single_mie_lut: texture_3d<f32>;
@group(0) @binding(5) var delta_scattering_lut: texture_3d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var aerosol_phase_lut: texture_2d_array<f32>;
@group(0) @binding(8) var density_out: texture_storage_3d<rgba16float, write>;

const BRUNETON_GOLDEN_ANGLE: f32 = 2.399963229728653;
const BRUNETON_DENSITY_UNIFORM_SAMPLE_COUNT: u32 = 96u;
const BRUNETON_DENSITY_PHASE_SAMPLE_COUNT: u32 = 32u;
const BRUNETON_DENSITY_SUN_SAMPLE_COUNT: u32 = 32u;
const BRUNETON_DENSITY_SAMPLE_COUNT: u32 = BRUNETON_DENSITY_UNIFORM_SAMPLE_COUNT
    + BRUNETON_DENSITY_PHASE_SAMPLE_COUNT
    + BRUNETON_DENSITY_SUN_SAMPLE_COUNT;
const BRUNETON_DENSITY_IMPORTANCE_G: f32 = 0.8;

fn bruneton_uniform_sphere_dir(sample_index: u32, sample_count: u32) -> vec3<f32> {
    let i = f32(sample_index) + 0.5;
    let y = 1.0 - 2.0 * i / f32(sample_count);
    let r = sqrt(max(1.0 - y * y, 0.0));
    let phi = i * BRUNETON_GOLDEN_ANGLE;
    return vec3<f32>(cos(phi) * r, y, sin(phi) * r);
}

fn bruneton_hg_sample_cosine(u: f32, g: f32) -> f32 {
    if (abs(g) < 1.0e-3) {
        return 1.0 - 2.0 * u;
    }
    let k = (1.0 - g * g) / max(1.0 - g + 2.0 * g * u, 1.0e-6);
    return clamp((1.0 + g * g - k * k) / (2.0 * g), -1.0, 1.0);
}

fn bruneton_hg_pdf(cos_theta: f32, g: f32) -> f32 {
    let denom = max(1.0 + g * g - 2.0 * g * clamp(cos_theta, -1.0, 1.0), 1.0e-6);
    return (1.0 - g * g) / (4.0 * ATM_PI * denom * sqrt(denom));
}

fn bruneton_oriented_dir(axis_in: vec3<f32>, cos_theta: f32, phi: f32) -> vec3<f32> {
    let axis = normalize(axis_in);
    let helper = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(axis.y) > 0.9,
    );
    let tangent = normalize(cross(helper, axis));
    let bitangent = cross(axis, tangent);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return normalize(
        axis * cos_theta
            + tangent * (sin_theta * cos(phi))
            + bitangent * (sin_theta * sin(phi))
    );
}

fn bruneton_hg_dir_around_axis(
    sample_index: u32,
    sample_count: u32,
    axis: vec3<f32>,
    phi_offset: f32,
) -> vec3<f32> {
    let i = f32(sample_index) + 0.5;
    let u = i / f32(sample_count);
    let cos_theta = bruneton_hg_sample_cosine(u, BRUNETON_DENSITY_IMPORTANCE_G);
    let phi = i * BRUNETON_GOLDEN_ANGLE + phi_offset;
    return bruneton_oriented_dir(axis, cos_theta, phi);
}

fn bruneton_density_mixture_pdf(
    incoming_dir: vec3<f32>,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
) -> f32 {
    let total = f32(BRUNETON_DENSITY_SAMPLE_COUNT);
    let uniform_weight = f32(BRUNETON_DENSITY_UNIFORM_SAMPLE_COUNT) / total;
    let phase_weight = f32(BRUNETON_DENSITY_PHASE_SAMPLE_COUNT) / total;
    let sun_weight = f32(BRUNETON_DENSITY_SUN_SAMPLE_COUNT) / total;
    return uniform_weight * ATM_INV_4PI
        + phase_weight * bruneton_hg_pdf(dot(incoming_dir, ray_dir), BRUNETON_DENSITY_IMPORTANCE_G)
        + sun_weight * bruneton_hg_pdf(dot(incoming_dir, sun_dir), BRUNETON_DENSITY_IMPORTANCE_G);
}

fn bruneton_previous_radiance(
    r: f32,
    up: vec3<f32>,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
) -> vec4<f32> {
    let mu = dot(up, ray_dir);
    let mu_s = dot(up, sun_dir);
    let nu = dot(ray_dir, sun_dir);
    if (bo.order == 2u) {
        let altitude = max(r - hp.earth_radius_km, 0.0);
        let rayleigh = bruneton_scattering_from_lut(scattering_lut, lut_sampler, r, mu, mu_s, nu)
            * molecular_phase_function(nu);
        let mie = bruneton_scattering_from_lut(single_mie_lut, lut_sampler, r, mu, mu_s, nu)
            * bruneton_aerosol_phase_from_reduced(altitude, nu);
        return rayleigh + mie;
    }
    return bruneton_scattering_from_lut(delta_scattering_lut, lut_sampler, r, mu, mu_s, nu)
        * molecular_phase_function(nu);
}

fn bruneton_aerosol_phase_from_reduced(altitude_km: f32, view_sun_nu: f32) -> vec4<f32> {
    var weighted_phase = vec4<f32>(0.0);
    var scattering = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
        let c = get_species_coeffs(hp, k, altitude_km);
        scattering += c.scattering;
        weighted_phase += c.scattering * aerosol_phase_at(k, -view_sun_nu);
    }
    return weighted_phase / max(scattering, vec4<f32>(1.0e-9));
}

fn bruneton_aerosol_phase_times_scattering(altitude_km: f32, view_sun_nu: f32) -> vec4<f32> {
    var phase_times_scattering = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < ATM_NUM_AEROSOL_SPECIES; k = k + 1u) {
        let c = get_species_coeffs(hp, k, altitude_km);
        phase_times_scattering += c.scattering * aerosol_phase_at(k, -view_sun_nu);
    }
    return phase_times_scattering;
}

fn bruneton_density_integrand(
    r: f32,
    pos: vec3<f32>,
    up: vec3<f32>,
    altitude: f32,
    coeffs: AtmCoeffs,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    incoming_dir: vec3<f32>,
) -> vec4<f32> {
    var incoming = bruneton_previous_radiance(r, up, incoming_dir, sun_dir);

    let incoming_segment = atmosphere_ray_limit(pos, incoming_dir);
    if (incoming_segment.hits_ground && incoming_segment.t_ground_km >= 0.0) {
        let ground_pos = pos + incoming_dir * incoming_segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        let ground_irradiance = bruneton_ground_irradiance_from_lut(
            irradiance_lut,
            lut_sampler,
            hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM,
            dot(ground_normal, sun_dir),
        );
        incoming += ground_irradiance * hp.ground_albedo_spectral * ATM_INV_PI;
    }

    let phase_cos = dot(ray_dir, incoming_dir);
    let phase_sca = coeffs.molecular_scattering * molecular_phase_function(phase_cos)
        + bruneton_aerosol_phase_times_scattering(altitude, phase_cos);
    return incoming * phase_sca;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(density_out);
    if (gid.x >= dims_u.x || gid.y >= dims_u.y || gid.z >= dims_u.z) {
        return;
    }
    let texel = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));

    let params = bruneton_scattering_params_from_texel(gid, dims_u);
    let pos = vec3<f32>(0.0, params.r, 0.0);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let altitude = max(params.r - hp.earth_radius_km, 0.0);
    let coeffs = get_atmosphere_collision_coefficients(hp, altitude);

    var source = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < BRUNETON_DENSITY_UNIFORM_SAMPLE_COUNT; i = i + 1u) {
        let incoming_dir = bruneton_uniform_sphere_dir(i, BRUNETON_DENSITY_UNIFORM_SAMPLE_COUNT);
        let pdf = bruneton_density_mixture_pdf(incoming_dir, params.ray_dir, params.sun_dir);
        source += bruneton_density_integrand(
            params.r,
            pos,
            up,
            altitude,
            coeffs,
            params.ray_dir,
            params.sun_dir,
            incoming_dir,
        ) / max(pdf, 1.0e-6);
    }

    for (var i: u32 = 0u; i < BRUNETON_DENSITY_PHASE_SAMPLE_COUNT; i = i + 1u) {
        let incoming_dir = bruneton_hg_dir_around_axis(
            i,
            BRUNETON_DENSITY_PHASE_SAMPLE_COUNT,
            params.ray_dir,
            0.0,
        );
        let pdf = bruneton_density_mixture_pdf(incoming_dir, params.ray_dir, params.sun_dir);
        source += bruneton_density_integrand(
            params.r,
            pos,
            up,
            altitude,
            coeffs,
            params.ray_dir,
            params.sun_dir,
            incoming_dir,
        ) / max(pdf, 1.0e-6);
    }

    for (var i: u32 = 0u; i < BRUNETON_DENSITY_SUN_SAMPLE_COUNT; i = i + 1u) {
        let incoming_dir = bruneton_hg_dir_around_axis(
            i,
            BRUNETON_DENSITY_SUN_SAMPLE_COUNT,
            params.sun_dir,
            1.37,
        );
        let pdf = bruneton_density_mixture_pdf(incoming_dir, params.ray_dir, params.sun_dir);
        source += bruneton_density_integrand(
            params.r,
            pos,
            up,
            altitude,
            coeffs,
            params.ray_dir,
            params.sun_dir,
            incoming_dir,
        ) / max(pdf, 1.0e-6);
    }

    source /= f32(BRUNETON_DENSITY_SAMPLE_COUNT);
    textureStore(density_out, texel, vec4<f32>(max(source.rgb, vec3<f32>(0.0)), max(source.a, 0.0)));
}
