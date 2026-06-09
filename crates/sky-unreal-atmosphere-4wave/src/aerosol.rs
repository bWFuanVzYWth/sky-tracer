mod inso;
mod soot;
mod waso;

pub const PHASE_LUT_SPECIES_U32: u32 = 3;
pub const PHASE_LUT_SPECIES: usize = PHASE_LUT_SPECIES_U32 as usize;
pub const PHASE_LUT_COS_BINS_U32: u32 = 1024;
pub const PHASE_LUT_COS_BINS: usize = PHASE_LUT_COS_BINS_U32 as usize;
pub const PHASE_LUT_WAVELENGTHS: usize = 4;

pub const SIGMA_SCA: [[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_SPECIES] = [
    waso::Waso::SIGMA_SCA,
    inso::Inso::SIGMA_SCA,
    soot::Soot::SIGMA_SCA,
];

pub const SIGMA_ABS: [[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_SPECIES] = [
    waso::Waso::SIGMA_ABS,
    inso::Inso::SIGMA_ABS,
    soot::Soot::SIGMA_ABS,
];

pub const PHASE_LUTS: [&[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS]; PHASE_LUT_SPECIES] = [
    waso::Waso::PHASE_LUT,
    inso::Inso::PHASE_LUT,
    soot::Soot::PHASE_LUT,
];

pub trait Aerosol {
    const SIGMA_SCA: [f32; PHASE_LUT_WAVELENGTHS];
    const SIGMA_ABS: [f32; PHASE_LUT_WAVELENGTHS];
    const PHASE_LUT: &[[f32; PHASE_LUT_WAVELENGTHS]; PHASE_LUT_COS_BINS];
}
