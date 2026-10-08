//! Pure CPU camera input. World Y is vertical; held keys translate both eye and target.
use cloud_pt::config::Camera;
use glam::{DQuat, DVec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKey {
    Forward,
    Backward,
    Left,
    Right,
    Down,
    Up,
    Fast,
}
#[derive(Clone, Debug)]
pub struct Controller {
    keys: [bool; 7],
    pub speed: f64,
    heading: DVec3,
}
impl Default for Controller {
    fn default() -> Self {
        Self {
            keys: [false; 7],
            speed: 100.0,
            heading: -DVec3::Z,
        }
    }
}
impl Controller {
    pub fn set_key(&mut self, key: MoveKey, pressed: bool) {
        self.keys[key as usize] = pressed;
    }
    pub fn clear(&mut self) {
        self.keys.fill(false);
    }
    pub fn moving(&self) -> bool {
        self.keys[0] != self.keys[1] || self.keys[2] != self.keys[3] || self.keys[4] != self.keys[5]
    }
    pub fn advance(
        &mut self,
        camera: &Camera,
        seconds: f64,
        ground_height: Option<f64>,
    ) -> Option<Camera> {
        if !self.moving()
            || !seconds.is_finite()
            || seconds <= 0.0
            || !self.speed.is_finite()
            || self.speed <= 0.0
        {
            return None;
        }
        let (forward, _, _) = camera.basis().ok()?;
        let planar = DVec3::new(forward.x, 0.0, forward.z);
        if planar.length_squared() > 1e-12 {
            self.heading = planar.normalize();
        }
        let right = self.heading.cross(DVec3::Y).normalize();
        let axis = |positive: usize, negative: usize| {
            f64::from(self.keys[positive] as u8) - f64::from(self.keys[negative] as u8)
        };
        let direction = self.heading * axis(0, 1) + right * axis(3, 2) + DVec3::Y * axis(5, 4);
        if direction.length_squared() < 1e-20 {
            return None;
        }
        let boost = if self.keys[MoveKey::Fast as usize] {
            4.0
        } else {
            1.0
        };
        let mut delta = direction.normalize() * (self.speed * boost * seconds.min(0.05));
        if let Some(floor) = ground_height {
            if floor.is_finite() {
                delta.y = (camera.origin.y + delta.y).max(floor) - camera.origin.y;
            }
        }
        let mut next = camera.clone();
        next.origin += delta;
        next.target += delta;
        if same_pose(camera, &next) || next.basis().is_err() {
            None
        } else {
            Some(next)
        }
    }
    /// FPS look keeps the eye fixed and reconstructs world-up, avoiding roll.
    pub fn look(camera: &Camera, dx: f64, dy: f64) -> Option<Camera> {
        if !dx.is_finite() || !dy.is_finite() || (dx == 0.0 && dy == 0.0) {
            return None;
        }
        let (forward, _, _) = camera.basis().ok()?;
        let yaw = forward.x.atan2(-forward.z) + dx * 0.005;
        let pitch =
            (forward.y.asin() - dy * 0.005).clamp(-89.0_f64.to_radians(), 89.0_f64.to_radians());
        let direction = DVec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        );
        let mut next = camera.clone();
        next.target = next.origin + direction * (camera.target - camera.origin).length();
        next.up = DVec3::Y;
        if next.basis().is_ok() && !same_pose(camera, &next) {
            Some(next)
        } else {
            None
        }
    }
    pub fn orbit(camera: &Camera, dx: f64, dy: f64) -> Option<Camera> {
        if !dx.is_finite() || !dy.is_finite() || (dx == 0.0 && dy == 0.0) {
            return None;
        }
        let offset = camera.origin - camera.target;
        let up = camera.up.normalize();
        let yawed = DQuat::from_axis_angle(up, -dx * 0.005) * offset;
        let right = (-yawed.normalize()).cross(up).normalize();
        let pitched = DQuat::from_axis_angle(right, -dy * 0.005) * yawed;
        let offset = if pitched.normalize().dot(up).abs() < 0.995 {
            pitched
        } else {
            yawed
        };
        let mut next = camera.clone();
        next.origin = next.target + offset;
        if next.basis().is_ok() && !same_pose(camera, &next) {
            Some(next)
        } else {
            None
        }
    }
    pub fn zoom(camera: &Camera, amount: f64) -> Option<Camera> {
        if !amount.is_finite() || amount == 0.0 {
            return None;
        }
        let offset = camera.origin - camera.target;
        let distance = (offset.length() * (-amount).exp().clamp(0.2, 5.0)).clamp(0.1, 1e9);
        let mut next = camera.clone();
        next.origin = next.target + offset.normalize() * distance;
        if next.basis().is_ok() && !same_pose(camera, &next) {
            Some(next)
        } else {
            None
        }
    }
}
pub fn same_pose(a: &Camera, b: &Camera) -> bool {
    a.origin == b.origin
        && a.target == b.target
        && a.up == b.up
        && a.horizontal_fov_deg == b.horizontal_fov_deg
}
pub fn sun_angles(direction: DVec3) -> [f64; 2] {
    let d = direction.normalize();
    [
        d.y.clamp(-1.0, 1.0).asin().to_degrees(),
        d.x.atan2(d.z).to_degrees(),
    ]
}
pub fn sun_direction(elevation_deg: f64, azimuth_deg: f64) -> DVec3 {
    let (se, ce) = elevation_deg.to_radians().sin_cos();
    let (sa, ca) = azimuth_deg.to_radians().sin_cos();
    DVec3::new(sa * ce, se, ca * ce)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn camera() -> Camera {
        Camera {
            origin: DVec3::new(0.0, 2.0, 0.0),
            target: DVec3::new(0.0, 3.0, -2.0),
            up: DVec3::Y,
            horizontal_fov_deg: 60.0,
        }
    }
    #[test]
    fn idle_and_opposed_keys_produce_no_reset() {
        let mut c = Controller::default();
        let eye = camera();
        assert!(c.advance(&eye, 0.02, None).is_none());
        c.set_key(MoveKey::Forward, true);
        c.set_key(MoveKey::Backward, true);
        assert!(!c.moving());
        assert!(c.advance(&eye, 0.02, None).is_none());
        c.clear();
        assert!(!c.moving());
    }
    #[test]
    fn horizontal_movement_preserves_pitch_and_shift_speed() {
        let eye = camera();
        let mut c = Controller::default();
        c.speed = 20.0;
        c.set_key(MoveKey::Forward, true);
        let next = c.advance(&eye, 0.025, None).unwrap();
        assert_eq!(next.origin.y, eye.origin.y);
        assert!(((next.origin - eye.origin).length() - 0.5).abs() < 1e-12);
        assert_eq!(next.target - next.origin, eye.target - eye.origin);
        c.set_key(MoveKey::Fast, true);
        let fast = c.advance(&eye, 0.025, None).unwrap();
        assert!(((fast.origin - eye.origin).length() - 2.0).abs() < 1e-12);
    }
    #[test]
    fn diagonal_speed_ground_clamp_and_large_dt_are_safe() {
        let eye = camera();
        let mut c = Controller::default();
        c.set_key(MoveKey::Forward, true);
        c.set_key(MoveKey::Right, true);
        let diagonal = c.advance(&eye, 10.0, None).unwrap();
        assert!(((diagonal.origin - eye.origin).length() - 5.0).abs() < 1e-12);
        c.clear();
        c.set_key(MoveKey::Down, true);
        let grounded = c.advance(&eye, 0.05, Some(0.0)).unwrap();
        assert_eq!(grounded.origin.y, 0.0);
        assert!(c.advance(&grounded, 0.05, Some(0.0)).is_none());
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(c.advance(&eye, dt, None).is_none());
        }
    }
    #[test]
    fn look_clamps_pitch_and_preserves_eye() {
        let eye = camera();
        let next = Controller::look(&eye, 400.0, -100000.0).unwrap();
        assert_eq!(next.origin, eye.origin);
        assert_eq!(next.up, DVec3::Y);
        assert!(next.basis().is_ok());
        assert!(next.basis().unwrap().0.y <= 89.0_f64.to_radians().sin() + 1e-12);
        assert!(Controller::look(&eye, 0.0, 0.0).is_none());
    }
    #[test]
    fn sun_azimuth_zero_is_positive_z_and_roundtrips() {
        assert_eq!(sun_direction(0.0, 0.0), DVec3::Z);
        assert!((sun_direction(0.0, 90.0) - DVec3::X).length() < 1e-12);
        let a = sun_angles(sun_direction(-6.0, 25.0));
        assert!((a[0] + 6.0).abs() < 1e-12 && (a[1] - 25.0).abs() < 1e-12);
    }
}
