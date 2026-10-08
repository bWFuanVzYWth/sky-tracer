mod analysis;
mod app;
mod benchmark;
mod output;

use clap::{Args, Parser, Subcommand, ValueEnum};
use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    film,
    gpu::ProgressiveRenderer,
    transport::TransportSettings,
    vdb,
    volume::SparseVolume,
};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

const DEFAULT_VDB: &str = "assets/DisneyCloudDataset/wdas_cloud/wdas_cloud_eighth.vdb";

#[derive(Parser)]
#[command(
    version,
    about = "Independent cloud path tracing reference and progressive viewer"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Inspect grid headers or decode density statistics, using only the CPU.
    Inspect {
        #[arg(default_value = DEFAULT_VDB)]
        vdb: PathBuf,
        #[arg(long, default_value = "density")]
        grid: String,
        #[arg(long)]
        load: bool,
    },
    /// CPU-only majorant cost analysis; does not render or initialize a GPU.
    Analyze {
        #[command(flatten)]
        options: Options,
        #[arg(long, default_value_t = 16)]
        rays_x: u32,
        #[arg(long, default_value_t = 8)]
        rays_y: u32,
    },
    /// Explicit GPU benchmark of global/spatial proposals and shadow roulette.
    Benchmark {
        #[command(flatten)]
        options: Options,
        #[arg(long, default_value_t = 3)]
        rounds: u32,
        /// Include slow no-roulette ablations; thick clouds can exhaust the diagnostic budget.
        #[arg(long)]
        include_no_shadow_roulette: bool,
        #[arg(long, default_value = "out/cloud-benchmark.json")]
        out: PathBuf,
    },
    /// Progressive GPU viewer; changes to camera or light restart accumulation.
    View {
        #[command(flatten)]
        options: Options,
        /// Diagnostic window run: exit after this duration, without exporting a reference.
        #[arg(long)]
        exit_after_seconds: Option<f64>,
        /// Diagnostic window run: exit after this many completed presentations.
        #[arg(long)]
        smoke_frames: Option<u64>,
    },
    /// Save linear EXR and sample statistics. GPU rendering is explicitly selected.
    Render {
        #[command(flatten)]
        options: Options,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
        #[arg(long, default_value = "out/cloud-reference")]
        out: PathBuf,
    },
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}
#[derive(Args)]
struct Options {
    #[arg(long, default_value = DEFAULT_VDB)]
    vdb: PathBuf,
    #[arg(long, default_value = "density")]
    grid: String,
    /// Complete JSON object containing camera, transport and render configuration.
    #[arg(long)]
    scene: Option<PathBuf>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    spp: Option<u32>,
    /// GPU samples per logical batch: 0 selects occupancy-based sizing, 1 uses one sample. Paths advance in bounded chunks.
    #[arg(long)]
    sample_batch_size: Option<u32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    density_scale: Option<f64>,
    #[arg(long, allow_hyphen_values = true)]
    g: Option<f64>,
    #[arg(long)]
    albedo: Option<f64>,
    #[arg(long)]
    no_ground: bool,
    #[arg(long, allow_hyphen_values = true)]
    sun_elevation: Option<f64>,
    #[arg(long, allow_hyphen_values = true)]
    sun_azimuth: Option<f64>,
    /// Exhaustion fails the result; this is not a maximum scattering depth.
    #[arg(long)]
    event_limit: Option<u64>,
    /// Diagnostic ablation of spatial majorants; retains the same density field.
    #[arg(long)]
    global_majorant: bool,
    /// Disable unbiased low-weight shadow-ray roulette for an ablation.
    #[arg(long)]
    no_shadow_roulette: bool,
    /// Display/PNG only; linear EXR and variance remain unchanged.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    exposure: f32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scene {
    camera: Camera,
    transport: TransportSettings,
    render: RenderConfig,
}
impl Options {
    fn resolve_view(&self) -> Result<Scene> {
        let mut scene = self.resolve()?;
        // Explicit scene files retain their chosen dimensions. The viewer's
        // interactive default is separate from the offline reference default.
        if self.scene.is_none() {
            scene.render.width = self.width.unwrap_or(128);
            scene.render.height = self.height.unwrap_or(72);
        }
        scene.render.validate()?;
        Ok(scene)
    }
    fn resolve(&self) -> Result<Scene> {
        let mut scene: Scene = if let Some(path) = &self.scene {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            Scene::default()
        };
        if let Some(v) = self.width {
            scene.render.width = v;
        }
        if let Some(v) = self.height {
            scene.render.height = v;
        }
        if let Some(v) = self.spp {
            scene.render.spp = v;
        }
        if let Some(v) = self.sample_batch_size {
            scene.render.sample_batch_size = v;
        }
        if let Some(v) = self.seed {
            scene.render.seed = v;
        }
        if let Some(v) = self.density_scale {
            scene.transport.extinction_scale = v;
        }
        if let Some(v) = self.g {
            scene.transport.phase_g = v;
        }
        if let Some(v) = self.albedo {
            scene.transport.scattering_albedo = DVec3::splat(v);
        }
        if let Some(v) = self.event_limit {
            scene.transport.event_limit = v;
        }
        if self.no_ground {
            scene.transport.ground = None;
        }
        if self.global_majorant {
            scene.transport.spatial_majorants = false;
        }
        if self.no_shadow_roulette {
            scene.transport.shadow_roulette = false;
        }
        if self.sun_elevation.is_some() || self.sun_azimuth.is_some() {
            let current = scene.transport.sun_direction.normalize();
            let elevation = self
                .sun_elevation
                .unwrap_or_else(|| current.y.asin().to_degrees());
            let azimuth = self
                .sun_azimuth
                .unwrap_or_else(|| current.x.atan2(current.z).to_degrees());
            if !elevation.is_finite()
                || !(-90.0..=90.0).contains(&elevation)
                || !azimuth.is_finite()
            {
                return Err("sun elevation must be finite in [-90, 90] and azimuth finite".into());
            }
            let (se, ce) = elevation.to_radians().sin_cos();
            let (sa, ca) = azimuth.to_radians().sin_cos();
            scene.transport.sun_direction = DVec3::new(sa * ce, se, ca * ce);
        }
        if !self.exposure.is_finite() || !(-30.0..=30.0).contains(&self.exposure) {
            return Err("display exposure must be finite in [-30, 30] EV".into());
        }
        scene.render.validate()?;
        scene.camera.basis()?;
        scene.transport.validate()?;
        Ok(scene)
    }
}
fn load(options: &Options) -> Result<SparseVolume> {
    let start = Instant::now();
    let volume = vdb::load_vdb(&options.vdb, &options.grid)?;
    eprintln!(
        "loaded {} / {}: {} bricks, {} uniform tiles, max density {}, {:.2}s (CPU)",
        options.vdb.display(),
        options.grid,
        volume.stats.stored_brick_count,
        volume.stats.nonzero_tile_count,
        volume.stats.maximum_density,
        start.elapsed().as_secs_f64()
    );
    Ok(volume)
}
fn record(
    options: &Options,
    scene: &Scene,
    volume: &SparseVolume,
    backend: &str,
) -> Result<output::Record> {
    Ok(output::Record {
        source: options.vdb.clone(), source_bytes: fs::metadata(&options.vdb)?.len(),
        grid: options.grid.clone(), volume_stats: volume.stats.clone(),
        camera: scene.camera.clone(), transport: scene.transport.clone(), render: scene.render.clone(),
        backend: backend.into(), adapter: None,
        asset_attribution: options.vdb.file_name().and_then(|n| n.to_str())
            .filter(|n| n.starts_with("wdas_cloud")).map(|_| "Walt Disney Animation Studios Cloud Data Set, Copyright 2017 Disney Enterprises, Inc.; CC BY-SA 3.0; source photograph Kevin Udy / Colorado Clouds Blog".into()),
    })
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect {
            vdb: path,
            grid,
            load: decode,
        } => {
            if decode {
                let volume = vdb::load_vdb(&path, &grid)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "source":path,"stats":volume.stats,"world_bounds":volume.world_bounds(),
                        "transform":volume.transform,"gpu_initialized":false
                    }))?
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&vdb::inspect_vdb(&path)?)?
                );
            }
        }
        Command::View {
            options,
            exit_after_seconds,
            smoke_frames,
        } => {
            let view_options = view_options(exit_after_seconds, smoke_frames)?;
            let scene = options.resolve_view()?;
            let volume = load(&options)?;
            let record = record(&options, &scene, &volume, "gpu-f32")?;
            let packed = volume.pack_gpu()?;
            drop(volume);
            app::run(
                packed,
                scene.camera,
                scene.transport,
                scene.render,
                options.exposure,
                record,
                view_options,
            )?;
        }
        Command::Analyze {
            options,
            rays_x,
            rays_y,
        } => {
            let scene = options.resolve()?;
            let volume = load(&options)?;
            let result = analysis::analyze(
                &volume,
                &scene.camera,
                &scene.render,
                &scene.transport,
                rays_x,
                rays_y,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "source":options.vdb,"grid":options.grid,"camera":scene.camera,
                    "transport":scene.transport,"analysis":result,"gpu_initialized":false,
                    "limits":"Expected Poisson candidate counts along fixed rays; not render time, FPS, or a measurement of multiple-scattering paths."
                }))?
            );
        }
        Command::Benchmark {
            options,
            rounds,
            include_no_shadow_roulette,
            out,
        } => {
            let mut scene = options.resolve()?;
            if options.scene.is_none() {
                if options.width.is_none() {
                    scene.render.width = 64;
                }
                if options.height.is_none() {
                    scene.render.height = 36;
                }
                if options.spp.is_none() {
                    scene.render.spp = 32;
                }
            }
            if out.exists() {
                return Err("benchmark output exists; choose a new file".into());
            }
            let volume = load(&options)?;
            let record = record(&options, &scene, &volume, "gpu-f32")?;
            let packed = volume.pack_gpu()?;
            drop(volume);
            benchmark::run(&packed, record, rounds, include_no_shadow_roulette, &out)?;
        }
        Command::Render {
            options,
            backend,
            out,
        } => {
            if out.exists() {
                return Err("output exists; choose a new directory".into());
            }
            let scene = options.resolve()?;
            let volume = load(&options)?;
            let mut record = record(
                &options,
                &scene,
                &volume,
                match backend {
                    Backend::Cpu => "cpu-f64",
                    Backend::Gpu => "gpu-f32",
                },
            )?;
            let start = Instant::now();
            let film = match backend {
                Backend::Cpu => film::render_cpu(
                    &volume,
                    &scene.camera,
                    &scene.transport,
                    &scene.render,
                    |n| {
                        if n == 1 || n % 16 == 0 || n == scene.render.spp {
                            eprintln!("CPU: {n}/{} spp", scene.render.spp);
                        }
                    },
                )?,
                Backend::Gpu => {
                    let packed = volume.pack_gpu()?;
                    drop(volume);
                    let (device, queue, adapter) = pollster::block_on(app::create_device(None))?;
                    record.adapter = Some(adapter);
                    let mut renderer = ProgressiveRenderer::new(
                        &device,
                        &queue,
                        &packed,
                        &scene.camera,
                        &scene.transport,
                        &scene.render,
                    )?;
                    let mut last_progress_report = Instant::now();
                    let mut chunks = 0u64;
                    while renderer.sample_count() < scene.render.spp {
                        let first = renderer.sample_count();
                        let count = renderer
                            .sample_batch_capacity()
                            .min(scene.render.spp - first);
                        let mut encoder = device.create_command_encoder(&Default::default());
                        renderer.encode_work(&device, &queue, &mut encoder, count)?;
                        queue.submit([encoder.finish()]);
                        let progress = renderer.read_progress(&device, &queue)?;
                        chunks += 1;
                        let n = renderer.sample_count();
                        if progress.batch_finished
                            && (first == 0 || n / 16 != first / 16 || n == scene.render.spp)
                        {
                            eprintln!(
                                "GPU: {n}/{} spp ({count} per batch, {chunks} bounded chunks)",
                                scene.render.spp
                            );
                            last_progress_report = Instant::now();
                        } else if last_progress_report.elapsed() >= Duration::from_secs(1) {
                            eprintln!(
                                "GPU: {n}/{} spp, tile {} / {}, paths {} / {}",
                                scene.render.spp,
                                progress.tile_start,
                                progress.total_pixels,
                                progress.completed_paths,
                                progress.total_paths
                            );
                            last_progress_report = Instant::now();
                        }
                    }
                    renderer.read_film(&device, &queue)?
                }
            };
            output::save(&out, &film, &record, options.exposure)?;
            eprintln!(
                "saved {} in {:.2}s",
                out.display(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

fn view_options(
    exit_after_seconds: Option<f64>,
    smoke_frames: Option<u64>,
) -> Result<app::ViewOptions> {
    let exit_after = if let Some(seconds) = exit_after_seconds {
        if !seconds.is_finite() || !(0.001..=3600.0).contains(&seconds) {
            return Err("--exit-after-seconds must be finite in [0.001, 3600]".into());
        }
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    };
    if smoke_frames.is_some_and(|n| !(1..=10000).contains(&n)) {
        return Err("--smoke-frames must be in [1, 10000]".into());
    }
    Ok(app::ViewOptions {
        exit_after,
        smoke_frames,
    })
}

#[cfg(test)]
mod viewer_cli_tests {
    use super::*;
    #[test]
    fn interactive_defaults_preserve_offline_sample_budget() {
        let cli = Cli::try_parse_from(["cloud-demo", "view"]).unwrap();
        let Command::View { options, .. } = cli.command else {
            panic!("expected view");
        };
        let offline = options.resolve().unwrap();
        let interactive = options.resolve_view().unwrap();
        assert_eq!([offline.render.width, offline.render.height], [640, 360]);
        assert_eq!(
            [interactive.render.width, interactive.render.height],
            [128, 72]
        );
        assert_eq!(interactive.render.spp, 1024);
        assert_eq!(offline.render.spp, interactive.render.spp);
        assert_eq!(
            offline.transport.extinction_scale,
            interactive.transport.extinction_scale
        );
        assert_eq!(
            offline.transport.ground.as_ref().unwrap().height,
            interactive.transport.ground.as_ref().unwrap().height
        );
    }
    #[test]
    fn explicit_view_dimensions_and_smoke_limits_are_checked_on_cpu() {
        let cli = Cli::try_parse_from([
            "cloud-demo",
            "view",
            "--width",
            "192",
            "--height",
            "108",
            "--exit-after-seconds",
            "3",
            "--smoke-frames",
            "4",
        ])
        .unwrap();
        let Command::View {
            options,
            exit_after_seconds,
            smoke_frames,
        } = cli.command
        else {
            panic!("expected view");
        };
        let scene = options.resolve_view().unwrap();
        assert_eq!([scene.render.width, scene.render.height], [192, 108]);
        assert_eq!(
            view_options(exit_after_seconds, smoke_frames)
                .unwrap()
                .exit_after,
            Some(Duration::from_secs(3))
        );
        for duration in [f64::NAN, f64::INFINITY, -1.0, 0.0, 3601.0] {
            assert!(view_options(Some(duration), None).is_err());
        }
        assert!(view_options(None, Some(0)).is_err());
        assert!(view_options(None, Some(10001)).is_err());
    }
}
