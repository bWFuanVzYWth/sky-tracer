use std::path::PathBuf;
use std::str::FromStr;
use std::{fmt, fmt::Display};

use crate::sampling::SamplerKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransmittanceEstimator {
    Ratio,
    ResidualRatio,
}

impl Display for TransmittanceEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ratio => write!(f, "ratio"),
            Self::ResidualRatio => write!(f, "residual"),
        }
    }
}

impl FromStr for TransmittanceEstimator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ratio" | "ratio-tracking" => Ok(Self::Ratio),
            "residual" | "residual-ratio" | "residual-ratio-tracking" => Ok(Self::ResidualRatio),
            _ => Err(format!(
                "unknown transmittance estimator `{s}`; expected `residual` or `ratio`"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollisionEstimator {
    Analog,
    Weighted,
}

impl Display for CollisionEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analog => write!(f, "analog"),
            Self::Weighted => write!(f, "weighted"),
        }
    }
}

impl FromStr for CollisionEstimator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "analog" | "roulette" | "absorption-roulette" => Ok(Self::Analog),
            "weighted" | "weighted-absorption" | "survival-biasing" => Ok(Self::Weighted),
            _ => Err(format!(
                "unknown collision estimator `{s}`; expected `weighted` or `analog`"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub spp: usize,
    pub seed: u64,
    pub out_dir: PathBuf,
    pub data_dir: PathBuf,
    pub sun_elevation_deg: f32,
    pub sun_azimuth_deg: f32,
    pub observer_altitude_km: f32,
    pub use_azimuth_symmetry: bool,
    pub sampler: SamplerKind,
    pub transmittance_estimator: TransmittanceEstimator,
    pub collision_estimator: CollisionEstimator,
    pub max_depth: usize,
    pub png_exposure: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 512,
            spp: 256,
            seed: 0x5EC7_2026_0430,
            out_dir: PathBuf::from("out"),
            data_dir: PathBuf::from("data"),
            sun_elevation_deg: 0.0,
            sun_azimuth_deg: 0.0,
            observer_altitude_km: 0.2,
            use_azimuth_symmetry: true,
            sampler: SamplerKind::RandomizedQmc,
            transmittance_estimator: TransmittanceEstimator::ResidualRatio,
            collision_estimator: CollisionEstimator::Weighted,
            max_depth: 16,
            png_exposure: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmittance_estimator_parses_cli_names() {
        assert_eq!(
            "residual".parse::<TransmittanceEstimator>().unwrap(),
            TransmittanceEstimator::ResidualRatio
        );
        assert_eq!(
            "ratio".parse::<TransmittanceEstimator>().unwrap(),
            TransmittanceEstimator::Ratio
        );
        assert!("bad".parse::<TransmittanceEstimator>().is_err());
    }

    #[test]
    fn collision_estimator_parses_cli_names() {
        assert_eq!(
            "weighted".parse::<CollisionEstimator>().unwrap(),
            CollisionEstimator::Weighted
        );
        assert_eq!(
            "analog".parse::<CollisionEstimator>().unwrap(),
            CollisionEstimator::Analog
        );
        assert!("bad".parse::<CollisionEstimator>().is_err());
    }
}
