@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var multi_scattering_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var aerosol_phase_lut: texture_2d_array<f32>;
@group(0) @binding(5) var ground_irradiance_lut: texture_2d<f32>;

const MULTI_SCATTERING_RAY_STEPS: u32 = 20u;
const MULTI_SCATTERING_SQRT_DIR_SAMPLES: u32 = 8u;
const MULTI_SCATTERING_DIR_SAMPLES: u32 = 64u;

fn uniform_sphere_dir_y_up(sample_index: u32) -> vec3<f32> {
    let sqrt_n = f32(MULTI_SCATTERING_SQRT_DIR_SAMPLES);
    let ix = f32(sample_index / MULTI_SCATTERING_SQRT_DIR_SAMPLES) + 0.5;
    let iy = f32(sample_index % MULTI_SCATTERING_SQRT_DIR_SAMPLES) + 0.5;
    let u = ix / sqrt_n;
    let v = iy / sqrt_n;
    let theta = 2.0 * ATM_PI * u;
    let phi = acos(1.0 - 2.0 * v);
    let sin_phi = sin(phi);
    return vec3<f32>(cos(theta) * sin_phi, cos(phi), sin(theta) * sin_phi);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(multi_scattering_out);
    if (gid.x >= dims_u.x || gid.y >= dims_u.y) {
        return;
    }

    let dims = vec2<f32>(dims_u);
    var uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / dims;
    uv = vec2<f32>(
        clamp(atm_from_sub_uvs_to_unit(uv.x, dims.x), 0.0, 1.0),
        clamp(atm_from_sub_uvs_to_unit(uv.y, dims.y), 0.0, 1.0),
    );

    let sun_mu = uv.x * 2.0 - 1.0;
    let sun_dir = normalize(vec3<f32>(sqrt(max(1.0 - sun_mu * sun_mu, 0.0)), sun_mu, 0.0));
    let view_height = hp.earth_radius_km
        + clamp(uv.y + ATM_PLANET_RADIUS_OFFSET_KM / hp.atmosphere_thickness_km, 0.0, 1.0)
        * (hp.atmosphere_thickness_km - ATM_PLANET_RADIUS_OFFSET_KM);
    let origin = vec3<f32>(0.0, view_height, 0.0);

    var multi_scat_as1_sum = vec4<f32>(0.0);
    var in_scattered_luminance_sum = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < MULTI_SCATTERING_DIR_SAMPLES; i = i + 1u) {
        let dir = uniform_sphere_dir_y_up(i);
        let segment = atmosphere_ray_limit(origin, dir);
        if (segment.t_max_km >= 0.0) {
            let result = integrate_scattered_luminance_direct_with_ground_irradiance(
                transmittance_lut,
                ground_irradiance_lut,
                lut_sampler,
                origin,
                dir,
                sun_dir,
                segment.t_max_km,
                MULTI_SCATTERING_RAY_STEPS,
                false,
                true,
                true,
            );
            multi_scat_as1_sum += result.multi_scat_as1;
            in_scattered_luminance_sum += result.radiance;
        }
    }

    let sphere_solid_angle = 4.0 * ATM_PI;
    let inv_sample_count = 1.0 / f32(MULTI_SCATTERING_DIR_SAMPLES);
    let multi_scat_as1 = multi_scat_as1_sum * (sphere_solid_angle * inv_sample_count * ATM_PHASE_ISOTROPIC);
    let in_scattered = in_scattered_luminance_sum * (sphere_solid_angle * inv_sample_count * ATM_PHASE_ISOTROPIC);
    let transfer = in_scattered / max(vec4<f32>(1.0) - multi_scat_as1, vec4<f32>(1.0e-3));

    textureStore(multi_scattering_out, vec2<i32>(gid.xy), vec4<f32>(max(transfer.rgb, vec3<f32>(0.0)), max(transfer.a, 0.0)));
}
