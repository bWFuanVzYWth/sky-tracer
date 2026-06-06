//! 单帧 view 输入。
//!
//! `ViewFrame` 只表达当前已迁移 runtime pass 会直接读取的 view 数据。后续如果
//! 引入插值、可见性集合或 render snapshot，应按真实职责拆分，而不是扩成大
//! context。

use bytemuck::{Pod, Zeroable};

/// runtime pass 使用的 view snapshot。
///
/// `clip_from_world` 与 `world_from_clip` 保留绝对世界矩阵，供确实需要绝对反投影的
/// 代码使用。热路径 shader 应优先使用相机相对矩阵和显式 camera basis，避免在
/// `f32` 中从大世界坐标相减恢复小向量。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewFrame {
    pub clip_from_world: [[f32; 4]; 4],
    pub world_from_clip: [[f32; 4]; 4],
    /// 以相机为原点的 world-to-clip 矩阵。渲染热路径使用它避免绝对世界坐标参与投影矩阵抵消。
    pub clip_from_relative_world: [[f32; 4]; 4],
    /// 以相机为原点的 clip-to-world 矩阵。sky/AP 用它反投影方向和距离，不恢复绝对位置。
    pub relative_world_from_clip: [[f32; 4]; 4],
    pub world_position: [f32; 4],
    /// 显式 camera basis，避免从 `world_from_clip` 反投影点再与相机位置做大数相减。
    pub world_forward: [f32; 4],
    pub world_right: [f32; 4],
    pub world_up: [f32; 4],
    /// x = `tan(fov_y / 2)`, y = aspect, z = near plane, w = reserved.
    pub view_params: [f32; 4],
    pub light_dir: [f32; 4],
    pub viewport: [f32; 4],
}
