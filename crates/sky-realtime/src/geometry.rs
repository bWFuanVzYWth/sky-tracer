use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub bottom: f32,
    pub top: f32,
}
impl Geometry {
    pub fn top_height(self) -> f32 {
        self.top - self.bottom
    }
    pub fn rho_squared(self, h: f32) -> f32 {
        h * (2.0 * self.bottom + h)
    }
    pub fn horizon(self, h: f32) -> f32 {
        -self.rho_squared(h).max(0.0).sqrt() / (self.bottom + h)
    }
}
pub fn unit(i: usize, size: usize) -> f32 {
    i as f32 / (size - 1) as f32
}
