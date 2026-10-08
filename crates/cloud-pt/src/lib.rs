//! Independent cloud reference transport over an explicitly interpolated VDB
//! density field. Loading, CPU reference and shader validation do not use a GPU.
pub mod config;
pub mod film;
pub mod gpu;
pub mod majorant;
pub mod sampling;
pub mod transport;
pub mod vdb;
pub mod volume;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
