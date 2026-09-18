use clap::Parser;
#[derive(Parser)]
struct Args {
    source: std::path::PathBuf,
    output: std::path::PathBuf,
}
fn main() -> sky_atmosphere_lut::Result<()> {
    let a = Args::parse();
    let manifest = sky_atmosphere_lut::rgb::export(&a.source, &a.output)?;
    sky_atmosphere_lut::rgb::verify(&manifest, &a.output)?;
    println!("Verified Rec.2020 resource: {}", a.output.display());
    Ok(())
}
