//! Offline, phase-inclusive spectral radiative transfer by successive orders.
//!
//! See docs/atmosphere.md and docs/numerics.md for transport and discretization,
//! file format and the distinction between a reference preset and validated accuracy.
pub mod asset;
pub mod bake_schedule;
pub mod config;
pub mod mapping;
pub mod model;
pub mod packed;
pub mod quadrature;
pub mod ray_mapping;
pub mod reference_mapping;
pub mod renderer;
pub mod rgb;
pub mod solver;
pub mod synthesis;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub mod physics;

mod direct;
pub mod spectral_dataset;
