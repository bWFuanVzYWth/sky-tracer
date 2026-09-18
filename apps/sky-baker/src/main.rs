//! CLI and file output; atmospheric transport lives in the solver crates.
mod film_output;
mod pt;
mod reference;
use clap::{Parser, Subcommand};
#[derive(Parser)]
#[command(
    version,
    about = "Bake independent PT images or a spectral reference LUT"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Pt(pt::Options),
    Reference {
        #[command(subcommand)]
        command: reference::Command,
    },
}
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match Cli::parse().command {
        Command::Pt(options) => pt::run(options).map_err(|e| e.to_string().into()),
        Command::Reference { command } => reference::run(command),
    }
}
