#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub fov_y_deg: f32,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            yaw_deg: 0.0,
            pitch_deg: 5.0,
            fov_y_deg: 70.0,
        }
    }
}

impl ViewState {
    pub const MIN_PITCH_DEG: f32 = -89.0;
    pub const MAX_PITCH_DEG: f32 = 89.0;
    pub const MIN_FOV_Y_DEG: f32 = 20.0;
    pub const MAX_FOV_Y_DEG: f32 = 120.0;

    pub fn orbit_pixels(&mut self, dx: f64, dy: f64) {
        const DEG_PER_PIXEL: f32 = 0.18;
        self.yaw_deg = wrap_degrees(self.yaw_deg + dx as f32 * DEG_PER_PIXEL);
        self.pitch_deg = (self.pitch_deg - dy as f32 * DEG_PER_PIXEL)
            .clamp(Self::MIN_PITCH_DEG, Self::MAX_PITCH_DEG);
    }

    pub fn zoom_steps(&mut self, steps: f32) {
        self.fov_y_deg =
            (self.fov_y_deg - steps * 3.0).clamp(Self::MIN_FOV_Y_DEG, Self::MAX_FOV_Y_DEG);
    }

    pub fn normalize(&mut self) {
        self.yaw_deg = wrap_degrees(self.yaw_deg);
        self.pitch_deg = self
            .pitch_deg
            .clamp(Self::MIN_PITCH_DEG, Self::MAX_PITCH_DEG);
        self.fov_y_deg = self
            .fov_y_deg
            .clamp(Self::MIN_FOV_Y_DEG, Self::MAX_FOV_Y_DEG);
    }
}

fn wrap_degrees(value: f32) -> f32 {
    value.rem_euclid(360.0)
}

#[cfg(test)]
mod tests {
    use super::ViewState;

    #[test]
    fn orbit_wraps_yaw_and_clamps_pitch() {
        let mut view = ViewState::default();
        view.orbit_pixels(3000.0, 3000.0);
        assert!((0.0..360.0).contains(&view.yaw_deg));
        assert_eq!(view.pitch_deg, ViewState::MIN_PITCH_DEG);
    }

    #[test]
    fn zoom_is_clamped() {
        let mut view = ViewState::default();
        view.zoom_steps(100.0);
        assert_eq!(view.fov_y_deg, ViewState::MIN_FOV_Y_DEG);
        view.zoom_steps(-100.0);
        assert_eq!(view.fov_y_deg, ViewState::MAX_FOV_Y_DEG);
    }
}
