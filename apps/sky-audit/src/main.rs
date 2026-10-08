mod benchmark;
mod evaluate;
mod profile;
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
    Benchmark(benchmark::Options),
    /// Moving-sun pure-sky frames with projection/display and separate CPU/GPU costs.
    Profile(profile::Options),
}
fn main() -> sky_realtime::Result<()> {
    match Cli::parse().command {
        Command::Evaluate(a) => evaluate::run(a),
        Command::Validate(a) => validate::run(a),
        Command::Benchmark(a) => benchmark::run(a),
        Command::Profile(a) => profile::run(a),
    }
}
