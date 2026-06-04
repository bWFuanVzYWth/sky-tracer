use std::sync::OnceLock;

use crate::math::Rgb;

pub const BAND_COUNT: usize = 15;
pub const CIE_1931_2DEG_CMF_NAME: &str = "CIE 1931 2 degree standard observer";
pub const CIE_1931_2DEG_CMF_SOURCE: &str = "data/CIE_xyz_1931_2deg.csv";
pub const LINEAR_SRGB_D65_NAME: &str = "linear sRGB D65";
pub const BRADFORD_WHITE_BALANCE_NAME: &str = "Bradford solar-white-to-D65";

pub type Mat3 = [[f32; 3]; 3];

pub const D65_WHITE_XYZ_Y1: [f32; 3] = [0.950_455_9, 1.0, 1.089_057_8];

pub const LINEAR_SRGB_FROM_XYZ_D65: Mat3 = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Xyz {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn from_array(v: [f32; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }

    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn normalized_to_y1(self) -> Self {
        if self.y.abs() <= f32::EPSILON {
            return Self::from_array(D65_WHITE_XYZ_Y1);
        }
        let inv_y = 1.0 / self.y;
        Self::new(self.x * inv_y, 1.0, self.z * inv_y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteBalanceMatrix {
    pub method: &'static str,
    pub source_white_xyz_y1: [f32; 3],
    pub target_white_xyz_y1: [f32; 3],
    pub xyz_from_xyz: Mat3,
}

impl WhiteBalanceMatrix {
    pub const fn identity() -> Self {
        Self {
            method: "identity",
            source_white_xyz_y1: D65_WHITE_XYZ_Y1,
            target_white_xyz_y1: D65_WHITE_XYZ_Y1,
            xyz_from_xyz: identity_mat3(),
        }
    }

    pub fn bradford(source_white_xyz: Xyz, target_white_xyz: Xyz) -> Self {
        let source = source_white_xyz.normalized_to_y1().to_array();
        let target = target_white_xyz.normalized_to_y1().to_array();
        let source_lms = mat3_mul_vec3(BRADFORD_LMS_FROM_XYZ, source);
        let target_lms = mat3_mul_vec3(BRADFORD_LMS_FROM_XYZ, target);
        let lms_scale = [
            safe_ratio(target_lms[0], source_lms[0]),
            safe_ratio(target_lms[1], source_lms[1]),
            safe_ratio(target_lms[2], source_lms[2]),
        ];
        let xyz_from_xyz = mat3_mul(
            XYZ_FROM_BRADFORD_LMS,
            mat3_mul(diagonal_mat3(lms_scale), BRADFORD_LMS_FROM_XYZ),
        );
        Self {
            method: BRADFORD_WHITE_BALANCE_NAME,
            source_white_xyz_y1: source,
            target_white_xyz_y1: target,
            xyz_from_xyz,
        }
    }

    pub fn apply_xyz(self, xyz: Xyz) -> Xyz {
        Xyz::from_array(mat3_mul_vec3(self.xyz_from_xyz, xyz.to_array()))
    }
}

#[derive(Clone, Debug)]
pub struct SpectralRgbConverter {
    band_xyz_weights: Vec<Xyz>,
    white_balance: WhiteBalanceMatrix,
}

impl SpectralRgbConverter {
    pub fn new_unbalanced(bands: &[SpectralBand]) -> Self {
        Self::new_with_white_balance(bands, WhiteBalanceMatrix::identity())
    }

    pub fn new_with_white_balance(
        bands: &[SpectralBand],
        white_balance: WhiteBalanceMatrix,
    ) -> Self {
        Self {
            band_xyz_weights: band_xyz_weights(bands),
            white_balance,
        }
    }

    pub fn new_solar_d65(bands: &[SpectralBand]) -> Self {
        let band_xyz_weights = band_xyz_weights(bands);
        let solar_white = spectrum_to_xyz_with_weights(
            &band_xyz_weights,
            &bands
                .iter()
                .map(|band| band.solar_irradiance_w_m2)
                .collect::<Vec<_>>(),
        );
        let white_balance =
            WhiteBalanceMatrix::bradford(solar_white, Xyz::from_array(D65_WHITE_XYZ_Y1));
        Self {
            band_xyz_weights,
            white_balance,
        }
    }

    pub fn white_balance(&self) -> WhiteBalanceMatrix {
        self.white_balance
    }

    pub fn to_xyz(&self, values: &[f32]) -> Xyz {
        spectrum_to_xyz_with_weights(&self.band_xyz_weights, values)
    }

    pub fn to_white_balanced_xyz(&self, values: &[f32]) -> Xyz {
        self.white_balance.apply_xyz(self.to_xyz(values))
    }

    pub fn to_linear_srgb(&self, values: &[f32]) -> Rgb {
        linear_srgb_from_xyz(self.to_white_balanced_xyz(values))
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

/// Convert band-integrated spectral values to display-debug linear sRGB.
///
/// The conversion remains decomposed as spectral-to-XYZ, a separately stored
/// solar-white-to-D65 matrix, then XYZ-to-linear-sRGB. The current spectral
/// asset stores one value per wavelength band after integration over that
/// band's wavelength range, so the band width is not multiplied in here.
pub fn spectral_to_linear_srgb(bands: &[SpectralBand], values: &[f32]) -> Rgb {
    SpectralRgbConverter::new_solar_d65(bands).to_linear_srgb(values)
}

pub fn spectral_to_xyz(bands: &[SpectralBand], values: &[f32]) -> Xyz {
    spectrum_to_xyz_with_weights(&band_xyz_weights(bands), values)
}

pub fn solar_d65_white_balance(bands: &[SpectralBand]) -> WhiteBalanceMatrix {
    SpectralRgbConverter::new_solar_d65(bands).white_balance()
}

pub fn cie_1931_2deg(lambda_nm: f32) -> Xyz {
    let samples = cie_samples();
    if lambda_nm <= samples[0].lambda_nm {
        return samples[0].xyz;
    }
    let last = samples.len() - 1;
    if lambda_nm >= samples[last].lambda_nm {
        return samples[last].xyz;
    }

    match samples.binary_search_by(|sample| sample.lambda_nm.total_cmp(&lambda_nm)) {
        Ok(index) => samples[index].xyz,
        Err(index) => {
            let a = samples[index - 1];
            let b = samples[index];
            let t = (lambda_nm - a.lambda_nm) / (b.lambda_nm - a.lambda_nm);
            Xyz::new(
                lerp(a.xyz.x, b.xyz.x, t),
                lerp(a.xyz.y, b.xyz.y, t),
                lerp(a.xyz.z, b.xyz.z, t),
            )
        }
    }
}

pub fn linear_srgb_from_xyz(xyz: Xyz) -> Rgb {
    let rgb = mat3_mul_vec3(LINEAR_SRGB_FROM_XYZ_D65, xyz.to_array());
    Rgb::new(rgb[0], rgb[1], rgb[2])
}

fn band_xyz_weights(bands: &[SpectralBand]) -> Vec<Xyz> {
    bands
        .iter()
        .map(|band| cie_1931_2deg(band.center_nm))
        .collect()
}

fn spectrum_to_xyz_with_weights(weights: &[Xyz], values: &[f32]) -> Xyz {
    let mut xyz = Xyz::default();
    for (weight, value) in weights.iter().zip(values.iter()) {
        xyz.x += value * weight.x;
        xyz.y += value * weight.y;
        xyz.z += value * weight.z;
    }
    xyz
}

#[derive(Clone, Copy, Debug)]
struct CieSample {
    lambda_nm: f32,
    xyz: Xyz,
}

fn cie_samples() -> &'static [CieSample] {
    static SAMPLES: OnceLock<Vec<CieSample>> = OnceLock::new();
    SAMPLES.get_or_init(parse_cie_samples).as_slice()
}

fn parse_cie_samples() -> Vec<CieSample> {
    const CSV: &str = include_str!("../../../data/CIE_xyz_1931_2deg.csv");
    let mut samples = Vec::new();
    for (line_index, line) in CSV.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        assert!(
            cols.len() == 4,
            "{} line {}: expected 4 columns",
            CIE_1931_2DEG_CMF_SOURCE,
            line_index + 1
        );
        samples.push(CieSample {
            lambda_nm: parse_cie_f32(cols[0], line_index),
            xyz: Xyz::new(
                parse_cie_f32(cols[1], line_index),
                parse_cie_f32(cols[2], line_index),
                parse_cie_f32(cols[3], line_index),
            ),
        });
    }
    assert!(
        !samples.is_empty(),
        "{} must contain at least one sample",
        CIE_1931_2DEG_CMF_SOURCE
    );
    samples
}

fn parse_cie_f32(value: &str, line_index: usize) -> f32 {
    value.parse::<f32>().unwrap_or_else(|error| {
        panic!(
            "{} line {}: failed to parse float '{}': {error}",
            CIE_1931_2DEG_CMF_SOURCE,
            line_index + 1,
            value
        )
    })
}

const BRADFORD_LMS_FROM_XYZ: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

const XYZ_FROM_BRADFORD_LMS: Mat3 = [
    [0.986_992_9, -0.147_054_3, 0.159_962_7],
    [0.432_305_3, 0.518_360_3, 0.049_291_2],
    [-0.008_528_7, 0.040_042_8, 0.968_486_7],
];

const fn identity_mat3() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

const fn diagonal_mat3(v: [f32; 3]) -> Mat3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]
}

fn mat3_mul_vec3(m: Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn safe_ratio(num: f32, denom: f32) -> f32 {
    if denom.abs() > 1.0e-12 {
        num / denom
    } else {
        1.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
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

    #[test]
    fn cie_table_loads_expected_range() {
        let samples = cie_samples();
        assert_eq!(samples.first().expect("first").lambda_nm, 360.0);
        assert_eq!(samples.last().expect("last").lambda_nm, 830.0);
        let y_555 = cie_1931_2deg(555.0).y;
        assert!((y_555 - 1.0).abs() < 0.01);
    }

    #[test]
    fn rgb_conversion_treats_values_as_band_integrated() {
        let narrow = SpectralBand {
            index: 0,
            center_nm: 550.0,
            lower_nm: 545.0,
            upper_nm: 555.0,
            solar_irradiance_w_m2: 1.0,
            ozone_cross_section_cm2: 0.0,
        };
        let wide = SpectralBand {
            lower_nm: 500.0,
            upper_nm: 600.0,
            ..narrow
        };

        let narrow_rgb = spectral_to_linear_srgb(&[narrow], &[2.0]);
        let wide_rgb = spectral_to_linear_srgb(&[wide], &[2.0]);

        assert_eq!(narrow_rgb.r, wide_rgb.r);
        assert_eq!(narrow_rgb.g, wide_rgb.g);
        assert_eq!(narrow_rgb.b, wide_rgb.b);
    }

    #[test]
    fn solar_white_balance_keeps_matrix_separate_and_neutralizes_white() {
        let bands = [
            SpectralBand {
                index: 0,
                center_nm: 450.0,
                lower_nm: 440.0,
                upper_nm: 460.0,
                solar_irradiance_w_m2: 2.0,
                ozone_cross_section_cm2: 0.0,
            },
            SpectralBand {
                index: 1,
                center_nm: 550.0,
                lower_nm: 540.0,
                upper_nm: 560.0,
                solar_irradiance_w_m2: 3.0,
                ozone_cross_section_cm2: 0.0,
            },
            SpectralBand {
                index: 2,
                center_nm: 650.0,
                lower_nm: 640.0,
                upper_nm: 660.0,
                solar_irradiance_w_m2: 1.5,
                ozone_cross_section_cm2: 0.0,
            },
        ];
        let converter = SpectralRgbConverter::new_solar_d65(&bands);
        let solar_values = bands
            .iter()
            .map(|band| band.solar_irradiance_w_m2)
            .collect::<Vec<_>>();
        let balanced = converter
            .white_balance()
            .apply_xyz(converter.to_xyz(&solar_values))
            .normalized_to_y1();

        assert_eq!(
            converter.white_balance().method,
            BRADFORD_WHITE_BALANCE_NAME
        );
        assert!((balanced.x - D65_WHITE_XYZ_Y1[0]).abs() < 0.001);
        assert!((balanced.y - D65_WHITE_XYZ_Y1[1]).abs() < 0.001);
        assert!((balanced.z - D65_WHITE_XYZ_Y1[2]).abs() < 0.001);
    }
}
