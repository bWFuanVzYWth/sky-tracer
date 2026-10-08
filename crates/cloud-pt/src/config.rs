use crate::{Result, transport::Ray};
use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub origin: DVec3,
    pub target: DVec3,
    pub up: DVec3,
    pub horizontal_fov_deg: f64,
}
impl Default for Camera {
    /// Disney's accompanying Mitsuba example camera, including its tilted up.
    fn default() -> Self {
        Self {
            origin: DVec3::new(648.064, -82.473, -63.856),
            target: DVec3::new(6.021, 100.043, -43.679),
            up: DVec3::new(0.273, 0.962, -0.009),
            horizontal_fov_deg: 54.43,
        }
    }
}
impl Camera {
    pub fn basis(&self) -> Result<(DVec3, DVec3, DVec3)> {
        if !self.origin.is_finite()
            || !self.target.is_finite()
            || !self.up.is_finite()
            || !self.horizontal_fov_deg.is_finite()
            || !(0.0..179.0).contains(&self.horizontal_fov_deg)
            || self.horizontal_fov_deg == 0.0
        {
            return Err(
                "camera requires finite vectors and horizontal FOV in (0, 179) degrees".into(),
            );
        }
        let delta = self.target - self.origin;
        if !delta.is_finite() || delta.length_squared() < 1e-20 || self.up.length_squared() < 1e-20
        {
            return Err("camera needs distinct origin/target and a nonzero up vector".into());
        }
        let forward = delta.normalize();
        let normalized_up = self.up.normalize();
        if !forward.is_finite()
            || forward.length_squared() == 0.0
            || !normalized_up.is_finite()
            || normalized_up.length_squared() == 0.0
        {
            return Err("camera vector normalization is not finite and nonzero".into());
        }
        let cross = forward.cross(normalized_up);
        if !cross.is_finite() || cross.length_squared() < 1e-12 {
            return Err("camera up cannot be parallel to its viewing direction".into());
        }
        let right = cross.normalize();
        let up = right.cross(forward).normalize();
        if !right.is_finite() || !up.is_finite() {
            return Err("camera basis normalization is not finite".into());
        }
        Ok((forward, right, up))
    }
    pub fn ray(&self, width: u32, height: u32, x: f64, y: f64) -> Result<Ray> {
        if width == 0 || height == 0 || !x.is_finite() || !y.is_finite() {
            return Err("camera ray needs positive dimensions and finite pixel coordinates".into());
        }
        let (forward, right, up) = self.basis()?;
        let scale = (self.horizontal_fov_deg.to_radians() * 0.5).tan();
        let sx = (2.0 * x / f64::from(width) - 1.0) * scale;
        let sy = (1.0 - 2.0 * y / f64::from(height)) * scale * f64::from(height) / f64::from(width);
        let direction = (forward + right * sx + up * sy).normalize();
        if !direction.is_finite() || direction.length_squared() == 0.0 {
            return Err("camera ray direction is not finite and nonzero".into());
        }
        Ok(Ray {
            origin: self.origin,
            direction,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub seed: u64,
    /// GPU samples traced concurrently per pixel. Zero selects an automatic
    /// capacity, one retains the original single-sample dispatch. CPU ignores it.
    #[serde(default)]
    pub sample_batch_size: u32,
}
impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            spp: 1024,
            seed: 0xC10D_2026_1008,
            sample_batch_size: 0,
        }
    }
}
impl RenderConfig {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.spp == 0 {
            return Err("width, height and samples per pixel must be positive".into());
        }
        if self.width.checked_mul(self.height).is_none() {
            return Err("cloud film dimensions overflow u32 indexing".into());
        }
        Ok(())
    }
}
