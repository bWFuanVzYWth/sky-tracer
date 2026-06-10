#[derive(Clone, Debug)]
pub struct Film {
    width: usize,
    height: usize,
    band_count: usize,
    values: Vec<f32>,
}

impl Film {
    #[must_use]
    pub fn new(width: usize, height: usize, band_count: usize) -> Self {
        Self {
            width,
            height,
            band_count,
            values: vec![0.0; width * height * band_count],
        }
    }

    #[must_use]
    pub fn from_values(width: usize, height: usize, band_count: usize, values: Vec<f32>) -> Self {
        assert_eq!(values.len(), width * height * band_count);
        Self {
            width,
            height,
            band_count,
            values,
        }
    }

    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub const fn band_count(&self) -> usize {
        self.band_count
    }

    pub fn set_pixel_spectrum(&mut self, pixel: usize, spectrum: &[f32]) {
        assert_eq!(spectrum.len(), self.band_count);
        let start = pixel * self.band_count;
        self.values[start..start + self.band_count].copy_from_slice(spectrum);
    }

    #[must_use]
    pub fn pixel_spectrum(&self, pixel: usize) -> &[f32] {
        let start = pixel * self.band_count;
        &self.values[start..start + self.band_count]
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}
