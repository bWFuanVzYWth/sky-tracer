const CA_AP_SLICE_COUNT: f32 = 32.0;
const CA_AP_KM_PER_SLICE: f32 = 4.0;
const CA_AP_M_TO_KM: f32 = 1.0e-3;
const CA_AP_REC2020_WHITE_FROM_FLAT_SPECTRUM: vec3<f32> =
    vec3<f32>(128.416, 108.538, 131.305);

fn ca_ap_rec2020_transmittance_from_spectral(t: vec4<f32>) -> vec3<f32> {
    let rgb = ca_atmosphere_linear_rec2020_from_spectral(t);
    return clamp(
        rgb / CA_AP_REC2020_WHITE_FROM_FLAT_SPECTRUM,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn ca_ap_clean_inscatter(v: vec4<f32>) -> vec3<f32> {
    return max(
        vec3<f32>(
            select(0.0, v.r, v.r == v.r),
            select(0.0, v.g, v.g == v.g),
            select(0.0, v.b, v.b == v.b),
        ),
        vec3<f32>(0.0),
    );
}

fn ca_ap_clean_transmittance(v: vec4<f32>) -> vec4<f32> {
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

fn ca_apply_aerial_perspective(
    src: vec3<f32>,
    inscatter_lut: texture_3d<f32>,
    transmittance_lut: texture_3d<f32>,
    lut_sampler: sampler,
    uv: vec2<f32>,
    dist_m: f32,
) -> vec3<f32> {
    let dist_km = max(dist_m, 0.0) * CA_AP_M_TO_KM;
    let slice_value = dist_km / CA_AP_KM_PER_SLICE;
    var weight = 1.0;
    var s = slice_value;
    if (s < 0.5) {
        weight = clamp(s * 2.0, 0.0, 1.0);
        s = 0.5;
    }

    let w_coord = clamp(sqrt(s / CA_AP_SLICE_COUNT), 0.0, 1.0);
    let lut_coord = vec3<f32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), w_coord);
    let inscatter = ca_ap_clean_inscatter(
        textureSampleLevel(inscatter_lut, lut_sampler, lut_coord, 0.0),
    ) * weight;
    let trans_spectral = ca_ap_clean_transmittance(
        textureSampleLevel(transmittance_lut, lut_sampler, lut_coord, 0.0),
    );
    let trans_sample = ca_ap_rec2020_transmittance_from_spectral(trans_spectral);
    let transmittance = mix(vec3<f32>(1.0), trans_sample, weight);
    return src * transmittance + inscatter;
}
