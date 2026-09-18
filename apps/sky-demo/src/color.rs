#[derive(Clone, Copy, Debug)]
pub struct DisplayTransform {
    pub output_space: OutputColorSpace,
    pub exposure: f32,
}

impl Default for DisplayTransform {
    fn default() -> Self {
        Self {
            output_space: OutputColorSpace::SrgbReinhardGamut,
            exposure: 0.10,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputColorSpace {
    SrgbReinhardGamut,
}

impl OutputColorSpace {
    #[allow(dead_code)]
    pub const fn shader_id(self) -> f32 {
        match self {
            Self::SrgbReinhardGamut => 0.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::SrgbReinhardGamut => "srgb-reinhard-gamut",
        }
    }
}
