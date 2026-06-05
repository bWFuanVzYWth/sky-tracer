@group(0) @binding(0) var<uniform> params: PrecomputedParams;
@group(0) @binding(1) var<uniform> order: OrderParams;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var phase_lut: texture_2d_array<f32>;
@group(0) @binding(5) var single_molecular: texture_3d<f32>;
@group(0) @binding(6) var single_aerosol0: texture_3d<f32>;
@group(0) @binding(7) var single_aerosol1: texture_3d<f32>;
@group(0) @binding(8) var single_aerosol2: texture_3d<f32>;
@group(0) @binding(9) var multiple_scattering_in: texture_3d<f32>;
@group(0) @binding(10) var irradiance_in: texture_2d<f32>;
@group(0) @binding(11) var density_out: texture_storage_3d<rgba16float, write>;

fn direction_from_spherical(cos_theta: f32, phi: f32) -> vec3<f32> {
    let sin_theta = safe_sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn build_omega_s(mu_s: f32) -> vec3<f32> {
    return vec3<f32>(safe_sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
}

fn build_view_omega(mu: f32, mu_s: f32, nu: f32) -> vec3<f32> {
    let omega_s = build_omega_s(mu_s);
    let sin_theta = safe_sqrt(1.0 - mu * mu);
    if (sin_theta < 1.0e-6) {
        return vec3<f32>(0.0, 0.0, select(-1.0, 1.0, mu >= 0.0));
    }
    let denom = max(safe_sqrt(1.0 - mu_s * mu_s) * sin_theta, 1.0e-6);
    let cos_phi = clamp((nu - mu * mu_s) / denom, -1.0, 1.0);
    let sin_phi = safe_sqrt(1.0 - cos_phi * cos_phi);
    return normalize(vec3<f32>(sin_theta * cos_phi, sin_theta * sin_phi, mu));
}

fn sample_previous_order_scattering(
    r: f32,
    mu: f32,
    mu_s: f32,
    nu: f32,
    hits_ground: bool,
) -> vec4<f32> {
    if (order.scattering_order == 2u) {
        return sample_single_scattering(
            phase_lut,
            lut_sampler,
            single_molecular,
            single_aerosol0,
            single_aerosol1,
            single_aerosol2,
            r,
            mu,
            mu_s,
            nu,
            hits_ground,
        );
    }
    return sample_scattering(multiple_scattering_in, lut_sampler, r, mu, mu_s, nu, hits_ground);
}

fn scattering_density_sample(r: f32, mu: f32, mu_s: f32, nu: f32) -> vec4<f32> {
    let sample_count: u32 = 16u;
    let omega = build_view_omega(mu, mu_s, nu);
    let omega_s = build_omega_s(mu_s);
    let h = r - params.earth_radius_km;
    var radiance_density = vec4<f32>(0.0);

    for (var l: u32 = 0u; l < sample_count; l = l + 1u) {
        let theta = (f32(l) + 0.5) * PI / f32(sample_count);
        let cos_theta = cos(theta);
        let sin_theta = sin(theta);
        let theta_weight = PI / f32(sample_count);
        for (var m: u32 = 0u; m < sample_count * 2u; m = m + 1u) {
            let phi = (f32(m) + 0.5) * PI / f32(sample_count);
            let domega = theta_weight * (PI / f32(sample_count)) * sin_theta;
            let omega_i = direction_from_spherical(cos_theta, phi);
            let incident_mu = clamp_cosine(omega_i.z);
            let incident_hits_ground = ray_intersects_ground(r, incident_mu);
            var incident = sample_previous_order_scattering(
                r,
                incident_mu,
                mu_s,
                clamp_cosine(dot(omega_s, omega_i)),
                incident_hits_ground,
            );

            if (incident_hits_ground) {
                let d_ground = distance_to_bottom_atmosphere_boundary(r, incident_mu);
                let r_ground = params.earth_radius_km;
                let ground_n = normalize(vec3<f32>(
                    d_ground * omega_i.x,
                    d_ground * omega_i.y,
                    r + d_ground * omega_i.z,
                ));
                let ground_mu_s = clamp_cosine(dot(ground_n, omega_s));
                let transmittance_to_ground =
                    get_transmittance(transmittance_lut, lut_sampler, r, incident_mu, d_ground, true);
                let ground_irradiance = sample_irradiance(irradiance_in, lut_sampler, r_ground, ground_mu_s);
                incident += transmittance_to_ground * params.ground_albedo_spectral * ground_irradiance * INV_PI;
            }

            let phase_nu = clamp_cosine(dot(omega, omega_i));
            let scattering =
                molecular_scattering(h) * rayleigh_phase(phase_nu)
                + aerosol_phase_weighted_scattering(phase_lut, lut_sampler, h, phase_nu);
            radiance_density += incident * scattering * domega;
        }
    }

    return radiance_density;
}

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(density_out);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let sp = scattering_params_from_frag_coord(vec3<f32>(gid) + vec3<f32>(0.5));
    let density = scattering_density_sample(sp.r, sp.mu, sp.mu_s, sp.nu);
    textureStore(density_out, vec3<i32>(gid), max(density, vec4<f32>(0.0)));
}
