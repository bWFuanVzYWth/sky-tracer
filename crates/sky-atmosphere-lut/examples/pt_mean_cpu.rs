//! Reuse existing high-spp PT regional means; compute only CPU LUT queries.
use clap::Parser;
use sky_atmosphere_lut::{
    Result,
    asset::Manifest,
    mapping::State,
    reference_mapping::{ReferenceStencil, radius_nodes},
};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    source: PathBuf,
    #[arg(long, num_args = 1..)]
    references: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    /// Diagnostic lookup override; the source asset is left unchanged.
    #[arg(long)]
    cubic_view: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let mut m = Manifest::open(&args.source)?;
    if !m.config.mapping.is_reference() {
        return Err("this diagnostic requires the horizon-aligned reference mapping".into());
    }
    let band = m.read_band(&args.source, 17)?;
    if args.cubic_view {
        m.config.view_interpolation = sky_atmosphere_lut::config::ViewInterpolation::MonotoneCubic;
        m.config.validate(m.bands.len())?;
    }
    let g = m.geometry;
    let heights = radius_nodes(g, &m.config);
    let mut results = Vec::new();
    for path in &args.references {
        let r: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        if r["model_fingerprint_fnv1a64"] != m.model_fingerprint_fnv1a64
            || r["pt_transport_version"] != "wgpu-layered-surface-v2"
            || r["lut_pixel_quadrature"] != "16x16"
            || r["bands"][0]["center_nm"] != 550.0
            || r["pt_max_orders"] != serde_json::Value::Null
        {
            return Err("incompatible cached PT reference".into());
        }
        let w = r["width"].as_u64().unwrap() as usize;
        let h = r["height"].as_u64().unwrap() as usize;
        let altitude = r["observer_altitude_km"].as_f64().unwrap() as f32;
        let sun_deg = r["sun_elevation_deg"].as_f64().unwrap() as f32;
        let sun = sun_deg.to_radians();
        let horizon = g.horizon(altitude).asin().to_degrees();
        let mut sum = [0.0_f32; 7];
        let mut count = [0_usize; 7];
        for y in 0..h {
            for x in 0..w {
                let low_e = 90.0 - (y + 1) as f32 * 180.0 / h as f32;
                let high_e = 90.0 - y as f32 * 180.0 / h as f32;
                let low_a = x as f32 * 360.0 / w as f32 - 180.0;
                let high_a = (x + 1) as f32 * 360.0 / w as f32 - 180.0;
                let disk = m.sun_radius_rad.to_degrees();
                let longitude = (m.sun_radius_rad.sin() / sun.cos()).asin().to_degrees();
                if high_e >= sun_deg - disk
                    && low_e <= sun_deg + disk
                    && high_a >= -longitude
                    && low_a <= longitude
                {
                    continue;
                }
                let mut value = 0.0;
                for sy in 0..16 {
                    for sx in 0..16 {
                        let e = (90.0 - (y as f32 + (sy as f32 + 0.5) / 16.0) * 180.0 / h as f32)
                            .to_radians();
                        let a = ((x as f32 + (sx as f32 + 0.5) / 16.0) * 360.0 / w as f32 - 180.0)
                            .to_radians();
                        let s = State {
                            altitude_km: altitude,
                            mu: e.sin(),
                            mu_s: sun.sin(),
                            nu: e.sin() * sun.sin() + e.cos() * sun.cos() * a.cos(),
                            ground: g.hits_ground(altitude, e.sin()),
                        };
                        value += ReferenceStencil::new(g, &m.config, s, Some(&heights))
                            .sample_with(|i| band.radiance[i])
                            / 256.0;
                    }
                }
                let e = (high_e + low_e) * 0.5;
                let a = (low_a + high_a) * 0.5;
                let above = e - horizon;
                let angle = (e.to_radians().sin() * sun.sin()
                    + e.to_radians().cos() * sun.cos() * a.to_radians().cos())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
                let sky = above > 0.0;
                let included = [
                    true,
                    sky,
                    sky && angle < 10.0,
                    sky && (10.0..30.0).contains(&angle),
                    above.abs() < 3.0,
                    sky && above < 12.0 && a.abs() > 150.0,
                    !sky,
                ];
                for i in 0..7 {
                    if included[i] {
                        sum[i] += value;
                        count[i] += 1;
                    }
                }
            }
        }
        let mut regions = serde_json::Map::new();
        for (i, name) in [
            "all",
            "sky",
            "near_sun",
            "solar_aureole",
            "horizon",
            "earth_shadow",
        ]
        .iter()
        .enumerate()
        {
            let reference = if i == 0 {
                &r["bands"][0]
            } else {
                &r["bands"][0]["regions"][name]
            };
            if count[i] as u64 != reference["pixels"].as_u64().unwrap() {
                return Err(format!("{name}: pixel mask differs from original comparison").into());
            }
            let pt = reference["mean_path_traced"].as_f64().unwrap() as f32;
            let mean = sum[i] / count[i] as f32;
            regions.insert(name.to_string(), serde_json::json!({"pixels":count[i],"mean_lut":mean,"mean_pt":pt,"bias_percent":100.0*(mean/pt-1.0)}));
        }
        results.push(serde_json::json!({"reference":path,"seed":r["pt_seed"],"spp":r["spp"],"regions":regions}));
    }
    fs::write(
        args.out,
        serde_json::to_vec_pretty(
            &serde_json::json!({"source":args.source,"lookup_config":m.config,"lookup_override":args.cubic_view,"nm":550,"quadrature":"16x16","results":results}),
        )?,
    )?;
    Ok(())
}
