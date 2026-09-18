use std::path::Path;

use sky_pt::config::RenderConfig;
use sky_pt::integrator::{RenderError, render};
use sky_pt::physics::data::load_scene_data;

#[test]
fn tiny_render_returns_finite_spectral_film() -> Result<(), Box<dyn std::error::Error>> {
    let config = RenderConfig {
        width: 32,
        height: 16,
        spp: 2,
        ..RenderConfig::default()
    };
    let scene = load_scene_data(&repository_data_dir(), 0.0, 0.0).expect("scene data");
    let film = match render(&scene, &config) {
        Ok(film) => film,
        Err(RenderError::NoAdapter(_)) => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    assert!(film.values().iter().all(|x| x.is_finite() && *x >= 0.0));
    assert!(film.values().iter().any(|x| *x > 0.0));
    Ok(())
}

fn repository_data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}
