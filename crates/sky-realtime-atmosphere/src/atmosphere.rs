//! 大气球壳几何配置。
//!
//! 长度配置以米为单位，GPU uniform 上传时转换为 km。当前 raster 路径采用
//! 近地局部切平面近似：世界 `Y` 轴为局部上方向，`world_y0_radius_m` 定义
//! 世界 `y = 0` 对应的大气半径。

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillaireAtmosphere {
    /// 大气底半径，单位 m。
    pub bottom_radius_m: f32,
    /// 大气顶半径，单位 m。
    pub top_radius_m: f32,
    /// 世界 y=0 平面对应的大气半径，单位 m。
    pub world_y0_radius_m: f32,
    /// scene unit 到 m 的比例。
    pub scene_units_to_m: f32,
}

impl Default for HillaireAtmosphere {
    fn default() -> Self {
        Self {
            bottom_radius_m: 6_360_000.0,
            top_radius_m: 6_460_000.0,
            world_y0_radius_m: 6_360_500.0,
            scene_units_to_m: 1.0,
        }
    }
}
