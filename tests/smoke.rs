use std::fs;
use std::path::Path;

use sky_tracer::config::RenderConfig;
use sky_tracer::data::load_scene_data;
use sky_tracer::integrator::render;

#[test]
fn tiny_render_writes_all_outputs_and_is_finite() {
    let out_dir = Path::new("target/test-output/tiny-render");
    let _ = fs::remove_dir_all(out_dir);

    let config = RenderConfig {
        width: 32,
        height: 16,
        spp: 2,
        out_dir: out_dir.to_owned(),
        max_depth: 4,
        ..RenderConfig::default()
    };
    let scene = load_scene_data(&config.data_dir, 0.0, 0.0).expect("scene data");
    let film = render(&scene, &config);
    assert!(film.values().iter().all(|x| x.is_finite() && *x >= 0.0));
    assert_mirrored_pixels_match(&film, scene.bands.len());
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
}

fn assert_mirrored_pixels_match(film: &sky_tracer::film::Film, band_count: usize) {
    for y in 0..film.height() {
        for x in 0..film.width() {
            let mirror_x = film.width() - 1 - x;
            let pixel = y * film.width() + x;
            let mirror_pixel = y * film.width() + mirror_x;
            for band in 0..band_count {
                let a = film.pixel_spectrum(pixel)[band];
                let b = film.pixel_spectrum(mirror_pixel)[band];
                assert_eq!(a, b);
            }
        }
    }
}
