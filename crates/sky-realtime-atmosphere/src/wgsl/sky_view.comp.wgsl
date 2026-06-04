// SkyView LUT bake.
//
// UE-style 2D SkyView parameterization. The view height comes from
// hp.sky_view_height_km. PT uses this LUT as its primary sky sampling boundary.

@group(0) @binding(0) var<uniform> hp: HillaireParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var aerosol_phase_lut: texture_2d_array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(sky_view_out);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    let dims = vec2<f32>(f32(tex_size.x), f32(tex_size.y));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / dims;
    let params = sky_view_uv_to_params(uv, dims);
    let ray_dir = sky_view_dir_from_params(params);
    let origin = atm_point_from_radius_km(sky_view_height_km());
    let ray = atm_ray_from_point(origin, ray_dir);
    let segment = atm_ray_segment(ray);

    if (segment.t_max_km < 0.0) {
        textureStore(sky_view_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0));
        return;
    }

    let scatter = compute_inscattering(
        transmittance_lut,
        lut_sampler,
        origin.local_pos_km,
        ray.dir,
        segment.t_max_km,
    );
    let rec2020 = max(
        white_balanced_linear_rec2020_from_spectral(scatter.radiance),
        vec3<f32>(0.0),
    );
    textureStore(
        sky_view_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(rec2020, 1.0),
    );
}
