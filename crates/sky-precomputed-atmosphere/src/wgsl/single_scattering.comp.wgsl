@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var molecular_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var aerosol0_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var aerosol1_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var aerosol2_out: texture_storage_3d<rgba16float, write>;

struct ReducedSingleScattering {
    molecular: vec4<f32>,
    aerosol0: vec4<f32>,
    aerosol1: vec4<f32>,
    aerosol2: vec4<f32>,
}

fn single_scattering_integrand(r: f32, mu: f32, mu_s: f32, nu: f32, d: f32, hits_ground: bool) -> ReducedSingleScattering {
    let r_i = clamp_radius(safe_sqrt(d * d + 2.0 * r * mu * d + r * r));
    let h_i = r_i - params.earth_radius_km;
    let mu_i = clamp_cosine((r * mu + d) / max(r_i, 1.0e-6));
    let mu_s_i = clamp_cosine((r * mu_s + d * nu) / max(r_i, 1.0e-6));
    let transmittance =
        get_transmittance(transmittance_lut, lut_sampler, r, mu, d, hits_ground)
        * get_transmittance_to_sun(transmittance_lut, lut_sampler, r_i, mu_s_i);
    let source = params.sun_spectral_irradiance * transmittance;
    return ReducedSingleScattering(
        source * molecular_scattering(h_i),
        source * species_scattering(0u, h_i),
        source * species_scattering(1u, h_i),
        source * species_scattering(2u, h_i),
    );
}

fn compute_single_scattering(r: f32, mu: f32, mu_s: f32, nu: f32, hits_ground: bool) -> ReducedSingleScattering {
    let sample_count: u32 = 50u;
    let dx = distance_to_nearest_atmosphere_boundary(r, mu, hits_ground) / f32(sample_count);
    var reduced = ReducedSingleScattering(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
    for (var i: u32 = 0u; i <= sample_count; i = i + 1u) {
        let weight = select(1.0, 0.5, i == 0u || i == sample_count);
        let sample = single_scattering_integrand(r, mu, mu_s, nu, f32(i) * dx, hits_ground);
        reduced.molecular += sample.molecular * weight * dx;
        reduced.aerosol0 += sample.aerosol0 * weight * dx;
        reduced.aerosol1 += sample.aerosol1 * weight * dx;
        reduced.aerosol2 += sample.aerosol2 * weight * dx;
    }
    return reduced;
}

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(molecular_out);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let sp = scattering_params_from_frag_coord(vec3<f32>(gid) + vec3<f32>(0.5));
    let reduced = compute_single_scattering(sp.r, sp.mu, sp.mu_s, sp.nu, sp.hits_ground);
    textureStore(molecular_out, vec3<i32>(gid), max(reduced.molecular, vec4<f32>(0.0)));
    textureStore(aerosol0_out, vec3<i32>(gid), max(reduced.aerosol0, vec4<f32>(0.0)));
    textureStore(aerosol1_out, vec3<i32>(gid), max(reduced.aerosol1, vec4<f32>(0.0)));
    textureStore(aerosol2_out, vec3<i32>(gid), max(reduced.aerosol2, vec4<f32>(0.0)));
}
