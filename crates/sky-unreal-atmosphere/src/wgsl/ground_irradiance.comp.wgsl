@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var ground_irradiance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var aerosol_phase_lut: texture_2d_array<f32>;

const GROUND_IRRADIANCE_RAY_STEPS: u32 = 24u;
const GROUND_IRRADIANCE_THETA_SAMPLES: u32 = 8u;
const GROUND_IRRADIANCE_PHI_SAMPLES: u32 = 16u;

fn ground_irradiance_params_from_uv(uv_in: vec2<f32>, dims: vec2<f32>) -> vec2<f32> {
    let uv = vec2<f32>(
        clamp(atm_from_sub_uvs_to_unit(uv_in.x, dims.x), 0.0, 1.0),
        clamp(atm_from_sub_uvs_to_unit(uv_in.y, dims.y), 0.0, 1.0),
    );
    let bottom = hp.earth_radius_km + ATM_PLANET_RADIUS_OFFSET_KM;
    let top = hp.earth_radius_km + hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM;
    let r = mix(bottom, top, uv.y);
    let mu_s = uv.x * 2.0 - 1.0;
    return vec2<f32>(r, mu_s);
}

fn local_sun_dir_from_mu_s(mu_s: f32) -> vec3<f32> {
    let y = clamp(mu_s, -1.0, 1.0);
    return normalize(vec3<f32>(sqrt(max(1.0 - y * y, 0.0)), y, 0.0));
}

fn hemisphere_dir(theta_index: u32, phi_index: u32) -> vec3<f32> {
    let d_theta = (0.5 * ATM_PI) / f32(GROUND_IRRADIANCE_THETA_SAMPLES);
    let d_phi = (2.0 * ATM_PI) / f32(GROUND_IRRADIANCE_PHI_SAMPLES);
    let theta = (f32(theta_index) + 0.5) * d_theta;
    let phi = (f32(phi_index) + 0.5) * d_phi;
    let sin_theta = sin(theta);
    return vec3<f32>(cos(phi) * sin_theta, cos(theta), sin(phi) * sin_theta);
}

fn hemisphere_dir_solid_angle(theta_index: u32) -> f32 {
    let d_theta = (0.5 * ATM_PI) / f32(GROUND_IRRADIANCE_THETA_SAMPLES);
    let d_phi = (2.0 * ATM_PI) / f32(GROUND_IRRADIANCE_PHI_SAMPLES);
    let theta = (f32(theta_index) + 0.5) * d_theta;
    return sin(theta) * d_theta * d_phi;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(ground_irradiance_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / dims;
    let r_mu_s = ground_irradiance_params_from_uv(uv, dims);
    let r = r_mu_s.x;
    let normalized_alt = clamp((r - hp.earth_radius_km) / hp.atmosphere_thickness_km, 0.0, 1.0);
    let sun_mu = r_mu_s.y;
    let sun_dir = local_sun_dir_from_mu_s(r_mu_s.y);
    let origin = vec3<f32>(0.0, r, 0.0);

    var irradiance_transfer = vec4<f32>(0.0);
    let sun_cos = max(sun_mu, 0.0);
    if (sun_cos > 0.0) {
        irradiance_transfer += transmittance_from_lut(
            transmittance_lut,
            lut_sampler,
            sun_cos,
            normalized_alt,
        ) * sun_cos;
    }

    for (var j: u32 = 0u; j < GROUND_IRRADIANCE_THETA_SAMPLES; j = j + 1u) {
        let domega = hemisphere_dir_solid_angle(j);
        for (var i: u32 = 0u; i < GROUND_IRRADIANCE_PHI_SAMPLES; i = i + 1u) {
            let dir = hemisphere_dir(j, i);
            let segment = atmosphere_ray_limit(origin, dir);
            if (segment.t_max_km >= 0.0 && !segment.hits_ground) {
                let scatter = integrate_scattered_luminance_direct(
                    transmittance_lut,
                    lut_sampler,
                    origin,
                    dir,
                    sun_dir,
                    segment.t_max_km,
                    GROUND_IRRADIANCE_RAY_STEPS,
                    true,
                    false,
                    true,
                );
                irradiance_transfer += scatter.radiance * max(dir.y, 0.0) * domega;
            }
        }
    }

    textureStore(
        ground_irradiance_out,
        vec2<i32>(gid.xy),
        vec4<f32>(max(irradiance_transfer.rgb, vec3<f32>(0.0)), max(irradiance_transfer.a, 0.0)),
    );
}
