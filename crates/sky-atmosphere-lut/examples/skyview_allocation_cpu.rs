//! CPU resampling of the frozen RGB reference through alternative SkyView charts.
//! This isolates the cache mapping, not hybrid transport or spectral differences.
use clap::Parser;
use glam::{Vec3, Vec3Swizzles};
use rayon::prelude::*;
use sky_atmosphere_lut::{
    Result,
    asset::Manifest,
    mapping::State,
    reference_mapping::{ReferenceStencil, radius_nodes},
    rgb,
};
use std::{f32::consts::PI, fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "out/lut_reference_v6_rgb")]
    source: PathBuf,
    #[arg(long, default_value = "out/four_wave_sun_v1/queries.json")]
    queries: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    export_curves_only: bool,
}
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    width: usize,
    rows: [usize; 3],
    power: u32,
    space_reuse: bool,
}
struct Chart {
    layout: Layout,
    rows: [usize; 3],
    bounds: [[f32; 2]; 3],
    horizon: f32,
    outer: f32,
    solar: f32,
    h: f32,
    radius: f32,
    top: f32,
}
impl Chart {
    fn new(layout: Layout, h: f32, solar: f32, radius: f32, top: f32) -> Self {
        let horizon = -(h * (2.0 * radius + h)).sqrt() / (radius + h);
        let outer = if h >= top {
            -((radius + top) / (radius + h)).clamp(0.0, 1.0).acos()
        } else {
            horizon.asin()
        };
        let mut rows = layout.rows;
        if h >= top && layout.space_reuse {
            rows[0] += rows[1];
            rows[1] = 0;
        }
        Self {
            layout,
            rows,
            bounds: [
                [horizon.asin(), if h >= top { outer } else { 0.0 }],
                [0.0, PI * 0.5],
                [-PI * 0.5, horizon.asin()],
            ],
            horizon,
            outer,
            solar,
            h,
            radius,
            top,
        }
    }
    fn forward(&self, e: f32, chart: usize) -> f32 {
        let [lo, hi] = self.bounds[chart];
        let e = e.clamp(lo, hi);
        let cdf = |center: f32, width: f32| {
            let a = ((lo - center) / width).atan();
            (((e - center) / width).atan() - a) / (((hi - center) / width).atan() - a).max(1e-10)
        };
        (0.15 * (e - lo) / (hi - lo).max(1e-10)
            + 0.40 * cdf(self.horizon.asin(), PI / 180.0)
            + 0.35 * cdf(self.solar, 0.7 * PI / 180.0)
            + 0.10 * cdf(self.outer, 0.5 * PI / 180.0))
        .clamp(0.0, 1.0)
    }
    fn inverse(&self, u: f32, chart: usize) -> f32 {
        let [mut lo, mut hi] = self.bounds[chart];
        for _ in 0..24 {
            let mid = (lo + hi) * 0.5;
            if self.forward(mid, chart) < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) * 0.5
    }
    fn make(&self, sample: &impl Fn(Vec3) -> Vec3) -> Vec<Vec3> {
        let mut result = Vec::new();
        for chart in 0..3 {
            for y in 0..self.rows[chart] {
                let e = self.inverse(y as f32 / (self.rows[chart] - 1) as f32, chart);
                let mu = if chart == 2 {
                    e.sin().min(self.horizon - 1e-7)
                } else {
                    e.sin().max(self.horizon + 1e-7)
                };
                for x in 0..self.layout.width {
                    if chart == 1 && self.h >= self.top {
                        result.push(Vec3::splat(1e-30_f32.ln()));
                        continue;
                    }
                    let u = x as f32 / (self.layout.width - 1) as f32;
                    let cp = 1.0 - 2.0 * u.powi(self.layout.power as i32);
                    let radial = (1.0 - mu * mu).max(0.0).sqrt();
                    let v = Vec3::new(radial * cp, radial * (1.0 - cp * cp).max(0.0).sqrt(), mu);
                    result.push(sample(v).max(Vec3::splat(1e-30)).ln());
                }
            }
        }
        result
    }
    fn sample(&self, table: &[Vec3], v: Vec3) -> Vec3 {
        let radius = self.radius + self.h;
        let chord = (self.h - self.top) * (radius + self.radius + self.top);
        if self.h >= self.top && (v.z >= 0.0 || (radius * v.z).powi(2) <= chord) {
            return Vec3::ZERO;
        }
        let cp = (v.x / v.xy().length().max(1e-10)).clamp(-1.0, 1.0);
        let u = ((1.0 - cp) * 0.5).powf(1.0 / self.layout.power as f32);
        let hit = v.z < 0.0 && v.z <= self.horizon;
        let chart = if hit {
            2
        } else if v.z >= 0.0 {
            1
        } else {
            0
        };
        let rows = self.rows[chart];
        let start: usize = self.rows[..chart].iter().sum();
        let y = self.forward(v.z.clamp(-1.0, 1.0).asin(), chart) * (rows - 1) as f32;
        let x = u * (self.layout.width - 1) as f32;
        let ix = (x as usize).min(self.layout.width - 2);
        let iy = (y as usize).min(rows - 2);
        let read = |j: usize| {
            table[(start + j) * self.layout.width + ix].lerp(
                table[(start + j) * self.layout.width + ix + 1],
                x - ix as f32,
            )
        };
        let a = read(iy);
        let b = read(iy + 1);
        if self.h >= self.top && !hit && iy == rows - 2 {
            let previous = self.inverse((rows - 2) as f32 / (rows - 1) as f32, 0).sin();
            let previous2 = self.inverse((rows - 3) as f32 / (rows - 1) as f32, 0).sin();
            let d = ((radius * v.z).powi(2) - chord).max(0.0);
            let d0 = ((radius * previous).powi(2) - chord).max(1e-20);
            let d1 = ((radius * previous2).powi(2) - chord).max(d0 + 1e-10);
            let change = (read(iy - 1) - a).exp() * (d0 / d1).sqrt() - Vec3::ONE;
            return (a.exp()
                * (d / d0).clamp(0.0, 1.0).sqrt()
                * (Vec3::ONE + change * ((d - d0) / (d1 - d0))))
                .max(Vec3::ZERO);
        }
        (a.lerp(b, y - iy as f32).exp() - Vec3::splat(1e-30)).max(Vec3::ZERO)
    }
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.out.exists() {
        return Err("choose a new output directory".into());
    }
    fs::create_dir_all(&a.out)?;
    let m = Manifest::open(&a.source)?;
    let channels: Vec<_> = (0..3)
        .map(|k| rgb::read_channel(&m, &a.source, k).map(|x| x.1))
        .collect::<Result<_>>()?;
    let heights = radius_nodes(m.geometry, &m.config);
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(&a.queries)?)?;
    let layouts = [
        Layout {
            name: "current",
            width: 256,
            rows: [96, 96, 64],
            power: 2,
            space_reuse: false,
        },
        Layout {
            name: "azimuth_p4",
            width: 256,
            rows: [96, 96, 64],
            power: 4,
            space_reuse: false,
        },
        Layout {
            name: "rows64_128_64",
            width: 256,
            rows: [64, 128, 64],
            power: 2,
            space_reuse: false,
        },
        Layout {
            name: "combined",
            width: 256,
            rows: [64, 128, 64],
            power: 4,
            space_reuse: true,
        },
        Layout {
            name: "upper160",
            width: 256,
            rows: [64, 160, 32],
            power: 4,
            space_reuse: true,
        },
        Layout {
            name: "wide512",
            width: 512,
            rows: [32, 64, 32],
            power: 4,
            space_reuse: true,
        },
    ];
    let scenes:Vec<_>=plan["images"].as_array().ok_or("no scenes")?.par_iter().map(|scene|->Result<serde_json::Value>{
        let number=|key:&str|scene[key].as_f64().unwrap() as f32;
        let h=number("altitude_km");let solar=number("sun_elevation_deg").to_radians();let sun=Vec3::new(solar.cos(),0.0,solar.sin());let g=m.geometry;
        let sample=|v:Vec3| {
            let s=State{altitude_km:h,mu:v.z,mu_s:sun.z,nu:v.dot(sun),ground:g.hits_ground(h,v.z)};
            if let Some((s,_))=g.atmosphere_entry(s) {let stencil=ReferenceStencil::new(g,&m.config,s,Some(&heights));Vec3::from_array(std::array::from_fn(|k|stencil.sample_with(|i|channels[k][i])))} else {Vec3::ZERO}
        };
        if a.export_curves_only {
            let chart=Chart::new(Layout{name:"fit",width:256,rows:[96,96,64],power:4,space_reuse:true},h,solar,g.bottom,g.top_height());
            let mut curves=Vec::<[f32;4]>::new();
            for k in 0..3 {
                let [lo,hi]=chart.bounds[k];
                for az in [0.0_f32,1.0,5.0,30.0,90.0,180.0] {
                    let az=az.to_radians();
                    for i in 0..2049 {
                        // Dense teacher in the old CDF, with a uniform-angle floor.
                        let u=i as f32/2048.0;
                        let e=0.85*chart.inverse(u,k)+0.15*(lo+(hi-lo)*u);
                        let mu=if k==2 {e.sin().min(chart.horizon-1e-7)} else {e.sin().max(chart.horizon+1e-7)};
                        let r=(1.0-mu*mu).max(0.0).sqrt();
                        let v=Vec3::new(r*az.cos(),r*az.sin(),mu);
                        let l=sample(v).max(Vec3::splat(1e-30)).ln();
                        curves.push([e,l.x,l.y,l.z]);
                    }
                }
            }
            let name=scene["name"].as_str().unwrap();
            fs::write(a.out.join(format!("{name}.curves.f32")),bytemuck::cast_slice(&curves))?;
            return Ok(serde_json::json!({"scene":name,"h":h,"solar":solar,"horizon":chart.horizon.asin(),"outer":chart.outer,"bounds":chart.bounds,"rows":chart.rows,"shape":[3,6,2049,4]}));
        }
        let width=number("width") as usize;let height=number("height") as usize;
        let yaw=number("yaw").to_radians();let pitch=number("pitch").to_radians();
        let forward=Vec3::new(pitch.cos()*yaw.cos(),pitch.cos()*yaw.sin(),pitch.sin());let right=Vec3::new(-yaw.sin(),yaw.cos(),0.0);let down=right.cross(forward);
        let tangent=(number("horizontal_fov").to_radians()*0.5).tan();
        let directions:Vec<_>=(0..width*height).map(|i| {let x=i%width;let y=i/width;let dx=(2.0*(x as f32+0.5)/width as f32-1.0)*tangent;let dy=(2.0*(y as f32+0.5)/height as f32-1.0)*tangent*height as f32/width as f32;(forward+right*dx+down*dy).normalize()}).collect();
        let truth:Vec<_>=directions.iter().map(|&v|sample(v)).collect();let mut variants=Vec::new();
        for layout in layouts {
            let chart=Chart::new(layout,h,solar,g.bottom,g.top_height());let table=chart.make(&sample);
            let values:Vec<_>=directions.iter().map(|&v|chart.sample(&table,v)).collect();
            let mut errors:Vec<_>=values.iter().zip(&truth).filter(|(_,b)|b.length()>1e-8).map(|(a,b)|100.0*(*a-*b).length()/b.length().max(1e-7)).collect();
            if errors.iter().any(|x|!x.is_finite()){return Err("nonfinite chart result".into());}errors.sort_by(f32::total_cmp);
            variants.push(serde_json::json!({"name":layout.name,"p95":errors[errors.len()*95/100],"p99":errors[errors.len()*99/100],"max":errors[errors.len()-1],"pixels":errors.len(),"rows":chart.rows,"size":[layout.width,chart.rows.iter().sum::<usize>()]}));
            if ["sun_close_5","sun_close_47","orbital_twilight"].contains(&scene["name"].as_str().unwrap()) {
                let packed:Vec<_>=values.iter().map(|v|v.extend(1.0).to_array()).collect();
                fs::write(a.out.join(format!("{}_{}.f32",scene["name"].as_str().unwrap(),layout.name)),bytemuck::cast_slice(&packed))?;
            }
        }
        eprintln!("sky chart audit {}",scene["name"].as_str().unwrap());
        Ok(serde_json::json!({"scene":scene["name"],"h":h,"solar":number("sun_elevation_deg"),"variants":variants}))
    }).collect::<Result<_>>()?;
    fs::write(
        a.out.join("audit.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"source":a.source,"model":m.model_fingerprint_fnv1a64,"scenes":scenes,"limits":["frozen full RGB sky, not the four-wave point-Sun hybrid", "isolated SkyView chart/interpolation; no new transport solve", "constant texel count; no GPU time prediction"]}),
        )?,
    )?;
    Ok(())
}
