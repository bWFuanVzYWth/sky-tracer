// PT runtime adapter for Hillaire spectral atmosphere.
//
// This wraps the 4-wavelength spectral backend as the PT-side API:
//   sky_radiance(dir) -> vec3<f32>
//   environment_sky_radiance_at(point, dir) -> vec3<f32>
//   view_sky_view_radiance(dir) -> vec3<f32>
//   atm_sun_transmittance_at(point) -> vec3<f32>
//   apply_aerial_perspective(src, uv, dist_m) -> vec3<f32>
//
// Callers must convert render-space positions to AtmPoint explicitly.
//
// Required module-scope bindings:
//   hp, transmittance_lut, lut_sampler, aerosol_phase_lut, sky_view_lut,
//   ap_inscatter_lut, ap_transmittance_lut, camera.

const PT_AP_SLICE_COUNT: f32 = 32.0;
const PT_AP_KM_PER_SLICE: f32 = 4.0;
const PT_M_TO_KM: f32 = 1.0e-3;
const PT_AP_REC2020_WHITE_FROM_FLAT_SPECTRUM: vec3<f32> =
    vec3<f32>(121.2, 107.3, 141.3);

fn pt_rec2020_transmittance_from_spectral(t: vec4<f32>) -> vec3<f32> {
    let rgb = linear_rec2020_from_spectral(t);
    return clamp(
        rgb / PT_AP_REC2020_WHITE_FROM_FLAT_SPECTRUM,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn pt_clean_inscatter(v: vec4<f32>) -> vec3<f32> {
    return max(
        vec3<f32>(
            select(0.0, v.r, v.r == v.r),
            select(0.0, v.g, v.g == v.g),
            select(0.0, v.b, v.b == v.b),
        ),
        vec3<f32>(0.0),
    );
}

fn pt_clean_transmittance(v: vec4<f32>) -> vec4<f32> {
    return clamp(
        vec4<f32>(
            select(1.0, v.r, v.r == v.r),
            select(1.0, v.g, v.g == v.g),
            select(1.0, v.b, v.b == v.b),
            select(1.0, v.a, v.a == v.a),
        ),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}

fn atm_camera_point() -> AtmPoint {
    return atm_point_from_radius_km(hp.eye_distance_to_earth_center_km);
}

fn atm_point_from_render_pos(world_pos: vec3<f32>) -> AtmPoint {
    // PT runtime 接收的是 ray query 的绝对命中点，当前只能在这里做绝对 y 差。
    // 如果 PT TLAS 改成 camera-relative，这个函数也必须同步改为相对高度输入。
    let radius_km = hp.eye_distance_to_earth_center_km
        + (world_pos.y - camera.origin.y) * PT_M_TO_KM;
    return atm_point_from_radius_km(radius_km);
}

fn apply_aerial_perspective(src: vec3<f32>, uv: vec2<f32>, dist_m: f32) -> vec3<f32> {
    let dist_km = max(dist_m, 0.0) * PT_M_TO_KM;
    let slice_value = dist_km / PT_AP_KM_PER_SLICE;
    var weight = 1.0;
    var s = slice_value;
    if (s < 0.5) {
        weight = clamp(s * 2.0, 0.0, 1.0);
        s = 0.5;
    }

    let w_coord = clamp(sqrt(s / PT_AP_SLICE_COUNT), 0.0, 1.0);
    let lut_coord = vec3<f32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), w_coord);
    let ap_inscatter = pt_clean_inscatter(
        textureSampleLevel(ap_inscatter_lut, lut_sampler, lut_coord, 0.0),
    ) * weight;
    let trans_spectral = pt_clean_transmittance(
        textureSampleLevel(ap_transmittance_lut, lut_sampler, lut_coord, 0.0),
    );
    let trans_rgb_sample = pt_rec2020_transmittance_from_spectral(trans_spectral);
    let trans_rgb = mix(vec3<f32>(1.0), trans_rgb_sample, weight);
    return src * trans_rgb + ap_inscatter;
}

fn sky_radiance_at(origin: AtmPoint, dir: vec3<f32>) -> vec3<f32> {
    let ray = atm_ray_from_point(origin, dir);
    let segment = atm_ray_segment(ray);
    if (segment.t_max_km < 0.0) {
        return vec3<f32>(0.0);
    }

    let scatter = compute_inscattering(
        transmittance_lut,
        lut_sampler,
        origin.local_pos_km,
        ray.dir,
        segment.t_max_km,
    );
    let rgb = linear_rec2020_from_spectral(scatter.radiance);
    return max(rgb, vec3<f32>(0.0));
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    return sky_radiance_at(atm_camera_point(), dir);
}

fn view_sky_view_radiance(dir: vec3<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(sky_view_lut));
    let uv = sky_view_uv_from_dir(dir, dims);
    return max(textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb, vec3<f32>(0.0));
}

// Biased approximation: ground/reflection environment lighting reuses the
// current view-height SkyView LUT. This is a deliberate PT architecture
// tradeoff; exact point-based sky integration is too expensive for runtime use.
fn environment_sky_radiance_at(origin: AtmPoint, dir: vec3<f32>) -> vec3<f32> {
    return view_sky_view_radiance(dir);
}

fn atm_sun_transmittance_at(origin: AtmPoint) -> vec3<f32> {
    let normalized_alt = clamp(origin.altitude_km / hp.atmosphere_thickness_km, 0.0, 1.0);
    let cos_theta = clamp(dot(origin.up, hp.sun_dir), -1.0, 1.0);
    let t_spectral = transmittance_from_lut(
        transmittance_lut,
        lut_sampler,
        cos_theta,
        normalized_alt,
    );
    let irr_through = linear_rec2020_from_spectral(hp.sun_spectral_irradiance * t_spectral);
    let irr_total = linear_rec2020_from_spectral(hp.sun_spectral_irradiance);
    let safe_total = max(irr_total, vec3<f32>(1.0e-6));
    return clamp(irr_through / safe_total, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn atm_sun_sample_visible(origin: AtmPoint, dir: vec3<f32>) -> bool {
    return atm_ray_above_horizon(atm_ray_from_point(origin, dir));
}
