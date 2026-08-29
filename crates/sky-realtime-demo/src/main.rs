mod app;
mod assets;
mod catalog;
mod color;
mod controls;
mod experiment;
mod gpu;
mod passes;
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
    #[arg(long, value_enum, default_value_t = app::ExperimentKind::Unreal8Wave)]
    experiment: app::ExperimentKind,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    app::run(app::RunConfig {
        asset_path: cli.asset,
        experiment: cli.experiment,
    })
}
