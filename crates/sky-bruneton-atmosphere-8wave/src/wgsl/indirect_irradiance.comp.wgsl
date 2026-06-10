struct BrunetonOrder {
    order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var<uniform> bo: BrunetonOrder;
@group(0) @binding(2) var scattering_lut: texture_3d<f32>;
@group(0) @binding(3) var single_mie_lut: texture_3d<f32>;
@group(0) @binding(4) var delta_scattering_lut: texture_3d<f32>;
@group(0) @binding(5) var irradiance_lut: texture_2d<f32>;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var irradiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var delta_irradiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var aerosol_phase_lut: texture_2d_array<f32>;

const BRUNETON_GOLDEN_ANGLE: f32 = 2.399963229728653;
const BRUNETON_IRRADIANCE_SAMPLE_COUNT: u32 = 128u;

fn bruneton_cosine_hemisphere_dir_y_up(sample_index: u32) -> vec3<f32> {
    let u = (f32(sample_index) + 0.5) / f32(BRUNETON_IRRADIANCE_SAMPLE_COUNT);
    let disk_r = sqrt(u);
    let phi = (f32(sample_index) + 0.5) * BRUNETON_GOLDEN_ANGLE;
    return vec3<f32>(
        disk_r * cos(phi),
        sqrt(max(1.0 - u, 0.0)),
        disk_r * sin(phi),
    );
}

fn bruneton_previous_radiance(
    r: f32,
    ray_dir: vec3<f32>,
    sun_dir: vec3<f32>,
) -> vec4<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
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
        weighted_phase += c.scattering * bruneton_aerosol_phase_at(k, view_sun_nu);
    }
    return weighted_phase / max(scattering, vec4<f32>(1.0e-9));
}

fn bruneton_phase_lut_uv(cos_theta: f32) -> f32 {
    return pow(max(0.5 - 0.5 * clamp(cos_theta, -1.0, 1.0), 0.0), 1.0 / 3.0);
}

fn bruneton_aerosol_phase_at(species: u32, view_sun_nu: f32) -> vec4<f32> {
    if (hp.mie_phase_mode == ATM_MIE_PHASE_MODE_CS) {
        return vec4<f32>(cornette_shanks_phase(ATM_CS_G, view_sun_nu));
    }
    return textureSampleLevel(
        aerosol_phase_lut,
        lut_sampler,
        vec2<f32>(bruneton_phase_lut_uv(view_sun_nu), 0.5),
        i32(species),
        0.0,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(irradiance_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / dims;
    let r_mu_s = bruneton_ground_irradiance_params_from_uv(uv, dims);
    let r = r_mu_s.x;
    let mu_s = r_mu_s.y;
    let sun_dir = normalize(vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0));

    var indirect = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < BRUNETON_IRRADIANCE_SAMPLE_COUNT; i = i + 1u) {
        let dir = bruneton_cosine_hemisphere_dir_y_up(i);
        indirect += bruneton_previous_radiance(r, dir, sun_dir);
    }
    indirect *= ATM_PI / f32(BRUNETON_IRRADIANCE_SAMPLE_COUNT);

    let accumulated = textureSampleLevel(irradiance_lut, lut_sampler, uv, 0.0);
    let out_value = accumulated + indirect;
    textureStore(delta_irradiance_out, vec2<i32>(gid.xy), vec4<f32>(max(indirect.rgb, vec3<f32>(0.0)), max(indirect.a, 0.0)));
    textureStore(irradiance_out, vec2<i32>(gid.xy), vec4<f32>(max(out_value.rgb, vec3<f32>(0.0)), max(out_value.a, 0.0)));
}
