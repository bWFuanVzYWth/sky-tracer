mod app;
mod assets;
mod catalog;
mod color;
mod controls;
mod experiment;
mod gpu;
mod passes;
mod snapshot;
mod ui_renderer;
mod view;
mod workbench;

use std::{error::Error, path::PathBuf};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about = "Realtime atmosphere experiment demo")]
struct Cli {
    #[arg(long, default_value = "out/asset.json")]
    asset: PathBuf,
    #[arg(long, value_enum)]
    experiment: Option<app::ExperimentKind>,
    /// Baked LUT directory; use out/four_wave_source_v1 with --experiment four-wave.
    #[arg(long)]
    lut: Option<PathBuf>,
    /// Refuse oversized LUT payloads before creating a GPU device. Excludes
    /// frame targets and other GPU allocations; 0 disables this guard.
    #[arg(long, default_value_t = 1024)]
    lut_budget_mib: u64,
    /// Save LUT/reference/absolute/signed comparison panels without opening a window.
    #[arg(long)]
    snapshot: Option<PathBuf>,
    /// Export four raw linear Rec.2020 f32 panels and a JSON sidecar.
    #[arg(long)]
    snapshot_linear: bool,
    /// Measure GPU timestamps for camera changes, sun changes, and cached frames.
    #[arg(long, default_value_t = 0)]
    benchmark_frames: usize,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    snapshot_yaw_deg: f32,
    #[arg(long, default_value_t = 15.0, allow_hyphen_values = true)]
    snapshot_pitch_deg: f32,
    #[arg(long, default_value_t = 90.0)]
    snapshot_fov_deg: f32,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    snapshot_exposure_ev: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let experiment = cli.experiment.unwrap_or(if cli.snapshot.is_some() {
        app::ExperimentKind::OfflineLut
    } else {
        app::ExperimentKind::Unreal8Wave
    });
    let lut = cli.lut.unwrap_or_else(|| {
        PathBuf::from(if matches!(experiment, app::ExperimentKind::FourWave) {
            "out/four_wave_source_v1"
        } else {
            "out/lut_reference_v6_packed16"
        })
    });
    if matches!(experiment, app::ExperimentKind::Hybrid4d) {
        let scene = sky_core::data::load_scene_data(std::path::Path::new("data"), 0.0, 0.0)?;
        let model = sky_atmosphere_lut::model::Model::from_scene(&scene).map_err(std::io::Error::other)?;
        let resident = sky_hybrid_atmosphere::estimated_resident_bytes(&model,
            &sky_hybrid_atmosphere::Wavelengths::optimized_four(), 0.18, &sky_hybrid_atmosphere::Config::balanced())
            .map_err(std::io::Error::other)?;
        if cli.lut_budget_mib != 0 && resident > cli.lut_budget_mib.saturating_mul(1048576) {
            return Err(format!("Hybrid resident LUTs including SkyView need {:.3} MiB, above --lut-budget-mib {}",
                resident as f32 / 1048576.0, cli.lut_budget_mib).into());
        }
    }
    if matches!(experiment, app::ExperimentKind::FourWave) {
        let resource =
            sky_atmosphere_lut::four_wave::Resource::open(&lut).map_err(std::io::Error::other)?;
        let resident = sky_atmosphere_lut::four_wave::cached::resident_bytes(&resource);
        if cli.lut_budget_mib != 0 && resident > cli.lut_budget_mib.saturating_mul(1048576) {
            return Err(format!(
                "Four-wave resident LUTs including SkyView need {:.3} MiB, above --lut-budget-mib {}",
                resident as f32 / 1048576.0,
                cli.lut_budget_mib
            )
            .into());
        }
    }
    if matches!(experiment, app::ExperimentKind::OfflineLut) {
        let manifest =
            sky_atmosphere_lut::asset::Manifest::open(&lut).map_err(std::io::Error::other)?;
        let (_, bytes) = sky_atmosphere_lut::renderer::storage_budget(&manifest);
        if cli.lut_budget_mib != 0 && bytes > cli.lut_budget_mib.saturating_mul(1048576) {
            return Err(format!("LUT needs {:.1} MiB of GPU payload, above --lut-budget-mib {}. Use the CPU-packed resource (--lut out/lut_reference_v6_packed16), or export-rgb then compress-rgb with sky-atmosphere-lut. Raise --lut-budget-mib explicitly only when enough VRAM is available.",bytes as f32/1048576.0,cli.lut_budget_mib).into());
        }
    }
    if let Some(output) = cli.snapshot {
        return snapshot::render(
            &cli.asset,
            &lut,
            &output,
            experiment,
            cli.snapshot_linear,
            [
                cli.snapshot_yaw_deg,
                cli.snapshot_pitch_deg,
                cli.snapshot_fov_deg,
                cli.snapshot_exposure_ev,
            ],
            cli.benchmark_frames,
        );
    }
    app::run(app::RunConfig {
        asset_path: cli.asset,
        experiment,
        lut_path: lut,
    })
}
