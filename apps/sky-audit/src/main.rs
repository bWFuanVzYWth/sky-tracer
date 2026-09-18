mod evaluate;
mod validate;
use clap::{Parser, Subcommand};
#[derive(Parser)]
#[command(version, about = "Reproducible solver evaluation and invariants")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Evaluate(evaluate::Options),
    Validate(validate::Options),
}
fn main() -> sky_realtime::Result<()> {
    match Cli::parse().command {
        Command::Evaluate(a) => evaluate::run(a),
        Command::Validate(a) => validate::run(a),
    }
}
