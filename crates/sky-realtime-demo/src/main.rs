mod app;
mod assets;
mod color;
mod experiment;
mod gpu;
mod passes;
mod view;

use std::{error::Error, path::PathBuf};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about = "Realtime atmosphere experiment demo")]
struct Cli {
    #[arg(long, default_value = "out/asset.json")]
    asset: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    app::run(app::RunConfig {
        asset_path: cli.asset,
    })
}
