//! 太阳光源兼容 re-export。
//!
//! 太阳是渲染系统共享输入，不属于 Hillaire 私有实现。保留本模块是为了避免
//! 上层调用点立刻大范围改名。

pub use ca_render::atmo::{SUN_IRRADIANCE_REC2020_W_PER_M2, Sun, SunGpu};
