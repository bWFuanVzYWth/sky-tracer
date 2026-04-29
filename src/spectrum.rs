use crate::math::Rgb;

pub const BAND_COUNT: usize = 15;

#[derive(Clone, Copy, Debug)]
pub struct SpectralBand {
    pub index: usize,
    pub center_nm: f32,
    pub lower_nm: f32,
    pub upper_nm: f32,
    pub solar_irradiance_w_m2: f32,
    pub ozone_cross_section_cm2: f32,
}

impl SpectralBand {
    pub fn width_nm(self) -> f32 {
        self.upper_nm - self.lower_nm
    }
}

pub fn default_band_centers() -> [f32; BAND_COUNT] {
    let mut bands = [0.0; BAND_COUNT];
    let start = 380.0_f32;
    let end = 780.0_f32;
    let step = (end - start) / BAND_COUNT as f32;
    let mut i = 0;
    while i < BAND_COUNT {
        bands[i] = start + (i as f32 + 0.5) * step;
        i += 1;
    }
    bands
}

pub fn spectral_to_linear_srgb(bands: &[SpectralBand], values: &[f32]) -> Rgb {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;

    for (band, value) in bands.iter().zip(values.iter()) {
        let (cx, cy, cz) = cie_1931_analytic(band.center_nm);
        let width = band.width_nm();
        x += value * cx * width;
        y += value * cy * width;
        z += value * cz * width;
    }

    Rgb::new(
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    )
}

pub fn cie_1931_analytic(lambda_nm: f32) -> (f32, f32, f32) {
    // Wyman/Sloan/Shirley-style Gaussian fit. It is compact and adequate for
    // preview RGB; the per-band EXR files remain the reference data.
    let x = gaussian_piece(lambda_nm, 442.0, 0.0624, 0.0374, 0.362)
        + gaussian_piece(lambda_nm, 599.8, 0.0264, 0.0323, 1.056)
        - gaussian_piece(lambda_nm, 501.1, 0.0490, 0.0382, 0.065);
    let y = gaussian_piece(lambda_nm, 568.8, 0.0213, 0.0247, 0.821)
        + gaussian_piece(lambda_nm, 530.9, 0.0613, 0.0322, 0.286);
    let z = gaussian_piece(lambda_nm, 437.0, 0.0845, 0.0278, 1.217)
        + gaussian_piece(lambda_nm, 459.0, 0.0385, 0.0725, 0.681);
    (x.max(0.0), y.max(0.0), z.max(0.0))
}

fn gaussian_piece(lambda_nm: f32, center: f32, left_scale: f32, right_scale: f32, amp: f32) -> f32 {
    let scale = if lambda_nm < center {
        left_scale
    } else {
        right_scale
    };
    let t = (lambda_nm - center) * scale;
    amp * (-0.5 * t * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bands_cover_visible_range() {
        let centers = default_band_centers();
        assert_eq!(centers.len(), BAND_COUNT);
        assert!(centers[0] > 380.0);
        assert!(centers[BAND_COUNT - 1] < 780.0);
    }
}
