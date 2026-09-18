//! CPU-only radial probe of existing LUTs; never creates a GPU device.
use clap::Parser;
use glam::Vec3;
use sky_atmosphere_lut::{
    asset::{BandLut, Manifest},
    mapping::State,
    model::{Model, phase_weight},
    quadrature,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/lut_reference_v2")]
    asset: PathBuf,
    #[arg(long)]
    first_order: Option<PathBuf>,
    #[arg(long, default_value_t = 17)]
    band: usize,
    #[arg(long, default_value = "out/sun_ring_profiles.csv")]
    out: PathBuf,
    /// Independently integrate single scattering at one-degree intervals.
    #[arg(long)]
    direct: bool,
    /// CPU-only full-spectrum sun-centered image: raw linear sRGB, 256x256x3 f32.
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long, default_value_t = 85.0)]
    solar: f32,
    #[arg(long, default_value_t = 120.0)]
    fov: f32,
    /// Diagnostic only: align the physical view within each interpolation corner.
    #[arg(long)]
    corner_lookup: bool,
}

fn corner_lookup(lut: &BandLut, s: State, nodes: &[f32], all_axes: bool) -> f32 {
    if all_axes {
        return sky_atmosphere_lut::mapping::sample_radiance_state(
            &lut.radiance,
            lut.geometry,
            &lut.config,
            s,
            nodes,
        );
    }
    let c = lut.geometry.coords_config(s, &lut.config);
    let dims = lut.config.scattering;
    let lo = c.map(|x| x.floor() as usize);
    let t = std::array::from_fn::<_, 4, _>(|i| c[i] - lo[i] as f32);
    let mut solar_values = [0.0; 2];
    for (upper_s, value) in solar_values.iter_mut().enumerate() {
        let si = (lo[2] + upper_s).min(dims[2] - 1);
        for upper_r in 0..2 {
            let ri = (lo[0] + upper_r).min(dims[0] - 1);
            let h = if all_axes {
                lut.geometry.height(ri as f32 / (dims[0] - 1) as f32)
            } else {
                s.altitude_km
            };
            let mus = if all_axes {
                lut.geometry
                    .solar_cosine(h, si as f32 / (dims[2] - 1) as f32)
            } else {
                s.mu_s
            };
            for upper_n in 0..2 {
                let ni = (lo[3] + upper_n).min(dims[3] - 1);
                let nu = lut.geometry.cone_cosine(h, mus, nodes[ni], s.ground);
                let point = State {
                    altitude_km: h,
                    mu_s: mus,
                    nu,
                    ..s
                };
                let mc = lut.geometry.optical_cone_coord(point, dims[1]);
                let mi = mc.floor() as usize;
                let mh = (mi + 1).min(dims[1] - 1);
                let mt = mc - mi as f32;
                let index = |mi| ((ri * dims[1] + mi) * dims[2] + si) * dims[3] + ni;
                let v = lut.radiance[index(mi)] * (1.0 - mt) + lut.radiance[index(mh)] * mt;
                *value += v
                    * if upper_r == 0 { 1.0 - t[0] } else { t[0] }
                    * if upper_n == 0 { 1.0 - t[3] } else { t[3] };
            }
        }
    }
    sky_atmosphere_lut::mapping::log_mix(solar_values[0], solar_values[1], t[2])
}

fn legacy_lookup(lut: &BandLut, s: State) -> f32 {
    sky_atmosphere_lut::mapping::sample_radiance(
        &lut.radiance,
        lut.config.scattering,
        lut.geometry.coords_config(s, &lut.config),
        lut.config.solar_interpolation,
    )
}

fn single_scattering(lut: &BandLut, model: &Model, band: usize, ray: Vec3, sun: Vec3) -> f32 {
    let g = lut.geometry;
    let h = 0.2;
    let s = State {
        altitude_km: h,
        mu: ray.z,
        mu_s: sun.z,
        nu: ray.dot(sun),
        ground: false,
    };
    let disk: Vec<_> = quadrature::sun_disk(4, 16, model.sun_radius)
        .into_iter()
        .map(|(v, w)| {
            let v = quadrature::rotate(v, sun);
            (v.z, ray.dot(v), model.bands[band].phases(ray.dot(v)), w)
        })
        .collect();
    let dx = g.distance(h, ray.z, false) / 2048.0;
    let mut trans = 1.0;
    let mut result = 0.0;
    for j in 0..2048 {
        let d = (j as f32 + 0.5) * dx;
        let point = g.advanced(s, d);
        let c = model.bands[band].coefficients(point.altitude_km);
        let mut source = 0.0;
        for &(sun_z, nu, phases, w) in &disk {
            let mu = ((g.bottom + h) * sun_z + d * nu) / (g.bottom + point.altitude_km);
            if !g.hits_ground(point.altitude_km, mu) {
                source +=
                    w * lut.transmittance(State {
                        mu,
                        ground: false,
                        ..point
                    }) * phase_weight(c, phases);
            }
        }
        let optical = c.extinction * dx;
        let weight = if optical < 0.001 {
            dx * (1.0 - optical * 0.5 + optical * optical / 6.0)
        } else {
            (1.0 - (-optical).exp()) / c.extinction
        };
        result += trans * source * weight * lut.info.solar_irradiance_w_m2;
        trans *= (-optical).exp();
    }
    result
}

fn main() -> sky_atmosphere_lut::Result<()> {
    let a = Args::parse();
    let m = Manifest::open(&a.asset)?;
    let nodes: Vec<_> = (0..m.config.scattering[3])
        .map(|i| {
            sky_atmosphere_lut::mapping::scattering_cosine(
                i as f32 / (m.config.scattering[3] - 1) as f32,
            )
        })
        .collect();
    if let Some(path) = a.image {
        let bands: Vec<_> = m
            .bands
            .iter()
            .enumerate()
            .map(|(index, b)| sky_core::spectrum::SpectralBand {
                index,
                center_nm: b.center_nm,
                lower_nm: b.lower_nm,
                upper_nm: b.upper_nm,
                solar_irradiance_w_m2: b.solar_irradiance_w_m2,
                ozone_cross_section_cm2: 0.0,
            })
            .collect();
        let converter = sky_core::spectrum::SpectralRgbConverter::new_solar_d65(&bands);
        let e = a.solar.to_radians();
        let sun = Vec3::new(e.cos(), 0.0, e.sin());
        let up = Vec3::new(-e.sin(), 0.0, e.cos());
        let rays: Vec<_> = (0..256 * 256)
            .map(|i| {
                let x = (2.0 * (i % 256) as f32 + 1.0) / 256.0 - 1.0;
                let y = 1.0 - (2.0 * (i / 256) as f32 + 1.0) / 256.0;
                (sun + (Vec3::Y * x + up * y) * (a.fov.to_radians() * 0.5).tan()).normalize()
            })
            .collect();
        let mut rgb = vec![[0.0; 3]; rays.len()];
        for band in 0..m.bands.len() {
            let lut = m.read_band(&a.asset, band)?;
            let mut unit = vec![0.0; m.bands.len()];
            unit[band] = 1.0;
            let w = converter.to_linear_srgb(&unit);
            for (rgb, &ray) in rgb.iter_mut().zip(&rays) {
                let value = if a.corner_lookup {
                    lut.sample(0.2, ray, sun, true)?
                } else {
                    let point = State {
                        altitude_km: 0.2,
                        mu: ray.z,
                        mu_s: sun.z,
                        nu: ray.dot(sun),
                        ground: lut.geometry.hits_ground(0.2, ray.z),
                    };
                    legacy_lookup(&lut, point)
                        + (lut.sample(0.2, ray, sun, true)? - lut.sample(0.2, ray, sun, false)?)
                };
                for (v, w) in rgb.iter_mut().zip([w.r, w.g, w.b]) {
                    *v += value * w;
                }
            }
        }
        let mut file = BufWriter::new(File::create(path)?);
        for rgb in rgb {
            for v in rgb {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        return Ok(());
    }
    let lut = m.read_band(&a.asset, a.band)?;
    let first = a
        .first_order
        .as_ref()
        .map(|p| Manifest::open(p)?.read_band(p, a.band))
        .transpose()?;
    let model = if a.direct {
        Some(Model::from_scene(
            &sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)
                .map_err(|e| e.to_string())?,
        )?)
    } else {
        None
    };
    let mut out = BufWriter::new(File::create(a.out)?);
    writeln!(
        out,
        "sun_deg,cone_azimuth_deg,angle_deg,view_mu,cone_coord,radiance,first_order,direct_single,nu_aligned_first,all_aligned_first,all_aligned_total"
    )?;
    for solar in [20.0_f32, 45.0, 47.0, 60.0, 70.0, 80.0, 85.0, 89.0, 90.0] {
        let e = solar.to_radians();
        let sun = Vec3::new(e.cos(), 0.0, e.sin());
        let up = Vec3::new(-e.sin(), 0.0, e.cos());
        for azimuth in [0.0_f32, 90.0, 180.0] {
            let phi = azimuth.to_radians();
            for i in 6..=1400 {
                let angle = i as f32 * 0.05;
                let theta = angle.to_radians();
                let ray = (sun * theta.cos()
                    + (up * phi.cos() + Vec3::Y * phi.sin()) * theta.sin())
                .normalize();
                if ray.z <= 0.01 {
                    continue;
                }
                let state = State {
                    altitude_km: 0.2,
                    mu: ray.z,
                    mu_s: sun.z,
                    nu: ray.dot(sun),
                    ground: false,
                };
                let value = legacy_lookup(&lut, state);
                let single = first
                    .as_ref()
                    .map(|l| legacy_lookup(l, state))
                    .unwrap_or(f32::NAN);
                let aligned = first
                    .as_ref()
                    .map(|l| corner_lookup(l, state, &nodes, false))
                    .unwrap_or(f32::NAN);
                let all_aligned = first
                    .as_ref()
                    .map(|l| corner_lookup(l, state, &nodes, true))
                    .unwrap_or(f32::NAN);
                let total_aligned = corner_lookup(&lut, state, &nodes, true);
                let direct = if i % 20 == 0 && azimuth == 90.0 {
                    model
                        .as_ref()
                        .map(|model| single_scattering(&lut, model, a.band, ray, sun))
                        .unwrap_or(f32::NAN)
                } else {
                    f32::NAN
                };
                writeln!(
                    out,
                    "{solar},{azimuth},{angle},{},{},{value},{single},{direct},{aligned},{all_aligned},{total_aligned}",
                    ray.z,
                    lut.geometry.coords_config(state, &lut.config)[1]
                )?;
            }
        }
        eprintln!("sampled sun elevation {solar}");
    }
    Ok(())
}
