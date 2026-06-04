use std::fs;
use std::path::Path;

use sky_tracer::config::RenderConfig;
use sky_tracer::data::load_scene_data;
use sky_tracer::integrator::{RenderError, render};

#[test]
fn tiny_render_writes_all_outputs_and_is_finite() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = Path::new("target/test-output/tiny-render");
    let _ = fs::remove_dir_all(out_dir);

    let config = RenderConfig {
        width: 32,
        height: 16,
        spp: 2,
        out_dir: out_dir.to_owned(),
        ..RenderConfig::default()
    };
    let scene = load_scene_data(&config.data_dir, 0.0, 0.0).expect("scene data");
    let film = match render(&scene, &config) {
        Ok(film) => film,
        Err(RenderError::NoAdapter(_)) => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    assert!(film.values().iter().all(|x| x.is_finite() && *x >= 0.0));
    assert!(film.values().iter().any(|x| *x > 0.0));
    film.write_outputs(&config.out_dir, &scene.bands, config.png_exposure)
        .expect("write outputs");

    assert!(out_dir.join("sky_rgb.exr").exists());
    assert!(out_dir.join("sky_rgb.png").exists());
    for band in &scene.bands {
        assert!(
            out_dir
                .join("bands")
                .join(format!("sky_{:03.0}nm.exr", band.center_nm))
                .exists()
        );
    }
    Ok(())
}
