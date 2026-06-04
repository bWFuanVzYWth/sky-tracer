use std::error::Error;
use std::fs;
use std::path::Path;

use exr::prelude::write_rgb_file;
use image::{ImageBuffer, RgbImage};

use sky_core::math::{Rgb, clamp01};
use sky_core::spectrum::{SpectralBand, SpectralRgbConverter};

#[derive(Clone, Debug)]
pub struct Film {
    width: usize,
    height: usize,
    band_count: usize,
    values: Vec<f32>,
}

impl Film {
    pub fn new(width: usize, height: usize, band_count: usize) -> Self {
        Self {
            width,
            height,
            band_count,
            values: vec![0.0; width * height * band_count],
        }
    }

    pub fn from_values(width: usize, height: usize, band_count: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), width * height * band_count);
        Self {
            width,
            height,
            band_count,
            values,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn set_pixel_spectrum(&mut self, pixel: usize, spectrum: &[f32]) {
        assert_eq!(spectrum.len(), self.band_count);
        let start = pixel * self.band_count;
        self.values[start..start + self.band_count].copy_from_slice(spectrum);
    }

    pub fn pixel_spectrum(&self, pixel: usize) -> &[f32] {
        let start = pixel * self.band_count;
        &self.values[start..start + self.band_count]
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn write_outputs(
        &self,
        out_dir: &Path,
        bands: &[SpectralBand],
        png_exposure: f32,
    ) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(out_dir.join("bands"))?;
        self.write_band_exrs(&out_dir.join("bands"), bands)?;
        self.write_rgb_outputs(out_dir, bands, png_exposure)?;
        Ok(())
    }

    pub fn write_rgb_outputs(
        &self,
        out_dir: &Path,
        bands: &[SpectralBand],
        png_exposure: f32,
    ) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(out_dir)?;
        self.write_rgb_exr(&out_dir.join("sky_rgb.exr"), bands)?;
        self.write_png(&out_dir.join("sky_rgb.png"), bands, png_exposure)?;
        Ok(())
    }

    fn write_band_exrs(
        &self,
        out_dir: &Path,
        bands: &[SpectralBand],
    ) -> Result<(), Box<dyn Error>> {
        for (band_index, band) in bands.iter().enumerate() {
            let path = out_dir.join(format!("sky_{:03.0}nm.exr", band.center_nm));
            write_rgb_file(path, self.width, self.height, |x, y| {
                let pixel = y * self.width + x;
                let v = self.pixel_spectrum(pixel)[band_index];
                (v, v, v)
            })?;
        }
        Ok(())
    }

    fn write_rgb_exr(&self, path: &Path, bands: &[SpectralBand]) -> Result<(), Box<dyn Error>> {
        let converter = SpectralRgbConverter::new_solar_d65(bands);
        write_rgb_file(path, self.width, self.height, |x, y| {
            let pixel = y * self.width + x;
            let rgb = converter
                .to_linear_srgb(self.pixel_spectrum(pixel))
                .finite_or_black();
            (rgb.r, rgb.g, rgb.b)
        })?;
        Ok(())
    }

    fn write_png(
        &self,
        path: &Path,
        bands: &[SpectralBand],
        exposure: f32,
    ) -> Result<(), Box<dyn Error>> {
        let converter = SpectralRgbConverter::new_solar_d65(bands);
        let mut image: RgbImage = ImageBuffer::new(self.width as u32, self.height as u32);
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = y * self.width + x;
                let rgb = converter
                    .to_linear_srgb(self.pixel_spectrum(pixel))
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
