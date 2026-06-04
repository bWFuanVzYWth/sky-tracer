#[derive(Clone, Copy, Debug)]
pub struct DisplayTransform {
    pub output_space: OutputColorSpace,
    pub exposure: f32,
}

impl Default for DisplayTransform {
    fn default() -> Self {
        Self {
            output_space: OutputColorSpace::Rec2020LinearDebug,
            exposure: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputColorSpace {
    Rec2020LinearDebug,
}

impl OutputColorSpace {
    pub fn shader_id(self) -> f32 {
        match self {
            Self::Rec2020LinearDebug => 0.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rec2020LinearDebug => "rec2020-linear-debug",
        }
    }
}
