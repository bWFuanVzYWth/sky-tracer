@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var multi_scattering_lut: texture_2d<f32>;
@group(0) @binding(4) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var aerosol_phase_lut: texture_2d_array<f32>;

const SKY_VIEW_STEPS: u32 = 32u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(sky_view_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(tex_size);
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / dims;
    let params = sky_view_uv_to_params(uv, dims);
    let ray_dir = sky_view_dir_from_params(params);
    var origin = vec3<f32>(0.0, sky_view_height_km(), 0.0);
    origin = move_to_top_atmosphere(origin, ray_dir);
    let segment = atmosphere_ray_limit(origin, ray_dir);

    if (segment.t_max_km < 0.0) {
        textureStore(sky_view_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0));
        return;
    }

    let scatter = integrate_scattered_luminance_with_ms(
        transmittance_lut,
        multi_scattering_lut,
        lut_sampler,
        origin,
        ray_dir,
        hp.sun_dir,
        segment.t_max_km,
        SKY_VIEW_STEPS,
    );
    var radiance = scatter.radiance;
    if (segment.hits_ground) {
        let ground_pos = origin + ray_dir * segment.t_ground_km;
        let ground_normal = normalize(ground_pos);
        let ground_sun_mu = dot(ground_normal, hp.sun_dir);
        let sun_cos = max(ground_sun_mu, 0.0);
        var ground_radiance = vec4<f32>(0.0);
        if (sun_cos > 0.0) {
            ground_radiance += transmittance_from_lut(
                transmittance_lut,
                lut_sampler,
                sun_cos,
                0.0,
            ) * (sun_cos * ATM_INV_PI);
        }
        ground_radiance += multi_scattering_from_lut(
            multi_scattering_lut,
            lut_sampler,
            ground_sun_mu,
            0.0,
        );
        radiance += hp.sun_spectral_irradiance
            * ground_radiance
            * hp.ground_albedo_spectral
            * scatter.transmittance;
    }

    textureStore(
        sky_view_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        max(radiance, vec4<f32>(0.0)),
    );
}
