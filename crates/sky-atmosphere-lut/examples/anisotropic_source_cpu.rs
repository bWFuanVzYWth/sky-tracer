//! Measure angular compression of the local scattering source (not sky L).
use clap::Parser;
use glam::Vec3;
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result, anisotropic as sh,
    asset::{Manifest, fingerprint},
    mapping::State,
    model::{Model, phase_weight},
    quadrature,
    reference_mapping::{ReferenceStencil, radius_nodes},
};
use std::{f32::consts::PI, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 64)]
    angular_mu: usize,
    #[arg(long, default_value_t = 128)]
    angular_phi: usize,
    #[arg(long, default_value_t = 24)]
    degree: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.degree < 2 || a.degree > 32 || a.angular_mu < 8 || a.angular_phi < 16 {
        return Err("invalid angular dimensions".into());
    }
    let start = Instant::now();
    let manifest = Manifest::open(&a.source)?;
    let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
        .map_err(|e| e.to_string())?;
    let model = Model::from_scene(&scene)?;
    if fingerprint(&model)? != manifest.model_fingerprint_fnv1a64 {
        return Err("model mismatch".into());
    }
    let heights = radius_nodes(model.geometry, &manifest.config);
    let poses: [[f32; 2]; 9] = [
        [0.2, 85.0],
        [0.2, 47.0],
        [0.2, 0.0],
        [0.2, -6.0],
        [2.0, -3.0],
        [12.0, -7.0],
        [30.0, -6.0],
        [108.0, -6.0],
        [120.0, 0.0],
    ];
    let degrees: Vec<_> = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32]
        .into_iter()
        .filter(|&d| d <= a.degree)
        .collect();
    let sphere = quadrature::sphere(a.angular_mu, a.angular_phi);
    let g = model.geometry;
    // Split the radial rule at the physical horizon. A partition of unity
    // combines it with a Sun-centered rule that resolves the incident aureole.
    let mut all_rows = Vec::new();
    for band in [7, 13, 20, 27] {
        let lut = manifest.read_band(&a.source, band)?;
        let bm = &model.bands[band];
        let pm = sh::phase_moments(bm, a.degree);
        let rows: Vec<_> = poses.par_iter().map(|&[h, sun_deg]| {
            let sun = Vec3::new(sun_deg.to_radians().cos(), 0.0, sun_deg.to_radians().sin());
            let horizon = g.horizon(h);
            let sun_weight = |v: Vec3| {
                let distance = 0.0004 + (1.0 - v.dot(sun)).max(0.0);
                0.01 / (0.01 + distance * distance)
            };
            let mut samples = Vec::new();
            for &(v,w) in &sphere {
                let v = quadrature::rotate(v, sun);
                samples.push((v,w * sun_weight(v)));
            }
            for (u,w) in quadrature::gauss_legendre(a.angular_mu) {
                for (lo,hi) in [(-1.0,horizon), (horizon,1.0)] {
                    let mu = lo + (hi-lo)*u;
                    for j in 0..a.angular_phi {
                        let phi = 2.0*PI*(j as f32+0.5)/a.angular_phi as f32;
                        let radial = (1.0-mu*mu).max(0.0).sqrt();
                        let v = Vec3::new(radial*phi.cos(),radial*phi.sin(),mu);
                        samples.push((v,w*(hi-lo)*2.0*PI/a.angular_phi as f32*(1.0-sun_weight(v))));
                    }
                }
            }
            let mut moments = vec![0.0; sh::count(a.degree)];
            let mut correction = moments.clone();
            let mut basis = moments.clone();
            let mut light = Vec::with_capacity(samples.len());
            for &(v,w) in &samples {
                let s = State { altitude_km:h, mu:v.z, mu_s:sun.z, nu:v.dot(sun), ground:g.hits_ground(h,v.z) };
                let l = ReferenceStencil::new(g, &lut.config, s, Some(&heights)).sample_with(|i| lut.radiance[i]);
                sh::cosine_sh(v,a.degree,&mut basis);
                for i in 0..moments.len() {
                    let term = l*w*basis[i]-correction[i];
                    let next = moments[i]+term;
                    correction[i] = (next-moments[i])-term;
                    moments[i] = next;
                }
                light.push(l*w);
            }
            let mut directions = Vec::new();
            for az in [0.0_f32,45.0,90.0,180.0] {
                for elev in [-90.0,-20.0,horizon.asin().to_degrees()-0.1,horizon.asin().to_degrees()+0.1,5.0,30.0,60.0,90.0] {
                    let (s,c) = elev.to_radians().sin_cos();
                    directions.push(Vec3::new(c*az.to_radians().cos(),c*az.to_radians().sin(),s));
                }
            }
            for delta in [0.0_f32,0.3,1.0,5.0,20.0] {
                let e = (sun_deg-delta).to_radians();
                directions.push(Vec3::new(e.cos(),0.0,e.sin()));
            }
            let c = bm.coefficients(h);
            let directions:Vec<_> = directions.iter().map(|&view| {
                sh::cosine_sh(view,a.degree,&mut basis);
                let exact:f32 = samples.iter().zip(&light).map(|((v,_),l)| l*phase_weight(c,bm.phases(v.dot(view)))).sum();
                let estimates:Vec<_> = degrees.iter().map(|&d| sh::source(c,&moments,&pm,&basis,d)).collect();
                serde_json::json!({"view":view.to_array(),"source":exact,"sh":estimates})
            }).collect();
            serde_json::json!({"height_km":h,"sun_elevation_deg":sun_deg,"wavelength_nm":bm.info.center_nm,"directions":directions,"phase_a0":pm[0],"phase_a8":pm.get(8)})
        }).collect();
        all_rows.extend(rows);
        eprintln!(
            "{} nm complete, {:.1} s",
            bm.info.center_nm,
            start.elapsed().as_secs_f32()
        );
    }
    fs::create_dir_all(&a.out)?;
    fs::write(
        a.out.join("audit.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"kind":"anisotropic_local_source_angular_audit_v1","source":a.source,"model":manifest.model_fingerprint_fnv1a64,"degrees":degrees,"angular_mu":a.angular_mu,"angular_phi":a.angular_phi,"seconds":start.elapsed().as_secs_f32(),"rows":all_rows,"limitations":["local source error, not integrated sky error","SH and oracle use the same angular incident-field quadrature; compare resolutions separately","frozen reference transport bias is not removed"]}),
        )?,
    )?;
    Ok(())
}
