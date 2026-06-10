#[path = "aerosol_hi/inso.rs"]
mod inso_hi;
#[path = "aerosol_lo/inso.rs"]
mod inso_lo;
#[path = "aerosol_hi/soot.rs"]
mod soot_hi;
#[path = "aerosol_lo/soot.rs"]
mod soot_lo;
#[path = "aerosol_hi/waso.rs"]
mod waso_hi;
#[path = "aerosol_lo/waso.rs"]
mod waso_lo;

use crate::params::SPECTRAL_GROUP_COUNT;

pub const PHASE_LUT_SPECIES_U32: u32 = 3;
pub const PHASE_LUT_SPECIES: usize = PHASE_LUT_SPECIES_U32 as usize;
pub const PHASE_LUT_COS_BINS_U32: u32 = 1024;
pub const PHASE_LUT_COS_BINS: usize = PHASE_LUT_COS_BINS_U32 as usize;
pub const PHASE_LUT_WAVELENGTHS: usize = 4;

pub const SIGMA_SCA: [[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_SPECIES]; SPECTRAL_GROUP_COUNT] = [
    [
        waso_lo::Waso::SIGMA_SCA,
        inso_lo::Inso::SIGMA_SCA,
        soot_lo::Soot::SIGMA_SCA,
    ],
    [
        waso_hi::Waso::SIGMA_SCA,
        inso_hi::Inso::SIGMA_SCA,
        soot_hi::Soot::SIGMA_SCA,
    ],
];

pub const SIGMA_ABS: [[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_SPECIES]; SPECTRAL_GROUP_COUNT] = [
    [
        waso_lo::Waso::SIGMA_ABS,
        inso_lo::Inso::SIGMA_ABS,
        soot_lo::Soot::SIGMA_ABS,
    ],
    [
        waso_hi::Waso::SIGMA_ABS,
        inso_hi::Inso::SIGMA_ABS,
        soot_hi::Soot::SIGMA_ABS,
    ],
];

pub const PHASE_LUTS: [[&[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS]; PHASE_LUT_SPECIES];
    SPECTRAL_GROUP_COUNT] = [
    [
        waso_lo::Waso::PHASE_LUT,
        inso_lo::Inso::PHASE_LUT,
        soot_lo::Soot::PHASE_LUT,
    ],
    [
        waso_hi::Waso::PHASE_LUT,
        inso_hi::Inso::PHASE_LUT,
        soot_hi::Soot::PHASE_LUT,
    ],
];

pub trait Aerosol {
    const SIGMA_SCA: [f32; PHASE_LUT_WAVELENGTHS];
    const SIGMA_ABS: [f32; PHASE_LUT_WAVELENGTHS];
    const PHASE_LUT: &[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS];
}
