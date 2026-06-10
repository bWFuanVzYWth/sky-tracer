use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use exr::prelude::{read_first_rgba_layer_from_file, write_rgb_file};
use image::{ImageBuffer, RgbImage};
use sky_core::math::{Rgb, clamp01};
use sky_core::spectrum::{SpectralBand, SpectralRgbConverter};
use sky_spectral_path_tracer::Film;

pub fn write_outputs(
    film: &Film,
    out_dir: &Path,
    bands: &[SpectralBand],
    png_exposure: f32,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(out_dir.join("bands"))?;
    write_band_exrs(film, &out_dir.join("bands"), bands)?;
    write_rgb_outputs(film, out_dir, bands, png_exposure)?;
    Ok(())
}

pub fn write_rgb_outputs(
    film: &Film,
    out_dir: &Path,
    bands: &[SpectralBand],
    png_exposure: f32,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(out_dir)?;
    write_rgb_exr(film, &out_dir.join("sky_rgb.exr"), bands)?;
    write_png(film, &out_dir.join("sky_rgb.png"), bands, png_exposure)?;
    Ok(())
}

pub fn read_film_from_band_exrs(
    out_dir: &Path,
    bands: &[SpectralBand],
) -> Result<Film, Box<dyn Error>> {
    if bands.is_empty() {
        return Err(IoError::new(ErrorKind::InvalidInput, "no spectral bands to rebuild").into());
    }

    let mut width = 0;
    let mut height = 0;
    let mut values = Vec::new();

    for (band_index, band) in bands.iter().enumerate() {
        let path = out_dir
            .join("bands")
            .join(format!("sky_{:03.0}nm.exr", band.center_nm));
        let image = read_band_exr(&path)?;
        if band_index == 0 {
            width = image.width;
            height = image.height;
            values.resize(width * height * bands.len(), 0.0);
        } else if image.width != width || image.height != height {
            return Err(IoError::new(
                ErrorKind::InvalidData,
                format!(
                    "band EXR {} has dimensions {}x{}, expected {}x{}",
                    path.display(),
                    image.width,
                    image.height,
                    width,
                    height
                ),
            )
            .into());
        }

        for (pixel, value) in image.pixels.iter().enumerate() {
            values[pixel * bands.len() + band_index] = *value;
        }
    }

    Ok(Film::from_values(width, height, bands.len(), values))
}

fn write_band_exrs(
    film: &Film,
    out_dir: &Path,
    bands: &[SpectralBand],
) -> Result<(), Box<dyn Error>> {
    for (band_index, band) in bands.iter().enumerate() {
        let path = out_dir.join(format!("sky_{:03.0}nm.exr", band.center_nm));
        write_rgb_file(path, film.width(), film.height(), |x, y| {
            let pixel = y * film.width() + x;
            let v = film.pixel_spectrum(pixel)[band_index];
            (v, v, v)
        })?;
    }
    Ok(())
}

fn write_rgb_exr(film: &Film, path: &Path, bands: &[SpectralBand]) -> Result<(), Box<dyn Error>> {
    let converter = SpectralRgbConverter::new_solar_d65(bands);
    write_rgb_file(path, film.width(), film.height(), |x, y| {
        let pixel = y * film.width() + x;
        let rgb = converter
            .to_linear_srgb(film.pixel_spectrum(pixel))
            .finite_or_black();
        (rgb.r, rgb.g, rgb.b)
    })?;
    Ok(())
}

fn write_png(
    film: &Film,
    path: &Path,
    bands: &[SpectralBand],
    exposure: f32,
) -> Result<(), Box<dyn Error>> {
    let converter = SpectralRgbConverter::new_solar_d65(bands);
    let mut image: RgbImage = ImageBuffer::new(film.width() as u32, film.height() as u32);
    for y in 0..film.height() {
        for x in 0..film.width() {
            let pixel = y * film.width() + x;
            let rgb = converter
                .to_linear_srgb(film.pixel_spectrum(pixel))
                .finite_or_black();
            let mapped = tonemap(rgb, exposure);
            image.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([to_srgb8(mapped.r), to_srgb8(mapped.g), to_srgb8(mapped.b)]),
            );
        }
    }
    image.save(path)?;
    Ok(())
}

#[derive(Debug)]
struct BandExrImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

fn read_band_exr(path: &Path) -> Result<BandExrImage, Box<dyn Error>> {
    let image = read_first_rgba_layer_from_file(
        path,
        |resolution, _channels| {
            let width = resolution.width();
            let height = resolution.height();
            BandExrImage {
                width,
                height,
                pixels: vec![0.0; width * height],
            }
        },
        |image, position, (r, _g, _b, _a): (f32, f32, f32, f32)| {
            image.pixels[position.y() * image.width + position.x()] = r;
        },
    )?;
    Ok(image.layer_data.channel_data.pixels)
}

fn tonemap(rgb: Rgb, exposure: f32) -> Rgb {
    let r = 1.0 - (-rgb.r.max(0.0) * exposure).exp();
    let g = 1.0 - (-rgb.g.max(0.0) * exposure).exp();
    let b = 1.0 - (-rgb.b.max(0.0) * exposure).exp();
    Rgb::new(r, g, b)
}

fn to_srgb8(linear: f32) -> u8 {
    let x = clamp01(linear);
    let srgb = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (clamp01(srgb) * 255.0 + 0.5) as u8
}
