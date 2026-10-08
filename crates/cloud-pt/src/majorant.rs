//! Exact 8^3-cell traversal for piecewise constant proposal majorants.
//!
//! Density and extinction are unchanged. Empty intervals are skipped, while
//! nonempty intervals can use exponential tracking with their own proposal rate.
//! Crossing times are recomputed from integer planes to avoid cumulative drift.

use crate::transport::{Ray, TraceError};
use crate::volume::SparseVolume;

#[derive(Clone, Copy, Debug)]
pub struct MajorantSpan {
    /// Physical distance from the normalized input ray's origin.
    pub start: f64,
    pub end: f64,
    pub max_density: f64,
}

pub struct MajorantSpans<'a> {
    volume: &'a SparseVolume,
    ray: Ray,
    cell: [i32; 3],
    distance: f64,
    endpoint: f64,
    done: bool,
}

impl<'a> MajorantSpans<'a> {
    pub(crate) fn new(
        volume: &'a SparseVolume,
        ray: Ray,
        endpoint: f64,
    ) -> Result<Self, TraceError> {
        if endpoint.is_nan() || endpoint < 0.0 {
            return Err(TraceError::InvalidSettings(
                "majorant endpoint must be nonnegative",
            ));
        }
        // Transport owns normalization. Preserve its exact ray so traversal
        // times and subsequent physical density queries use the same direction.
        if !ray.origin.is_finite()
            || !ray.direction.is_finite()
            || (ray.direction.length_squared() - 1.0).abs() > 1.0e-12
        {
            return Err(TraceError::InvalidRay);
        }
        let bounds = volume.world_bounds();
        if !bounds.valid() {
            return Err(TraceError::InvalidVolume("invalid world bounds"));
        }
        let origin = volume.world_to_index(ray.origin);
        let direction = ray.direction / volume.transform.scale;
        if !origin.is_finite() || !direction.is_finite() {
            return Err(TraceError::NumericalFailure(
                "majorant ray transform overflow",
            ));
        }
        let mut result = Self {
            volume,
            ray,
            cell: [0; 3],
            distance: 0.0,
            endpoint: 0.0,
            done: true,
        };
        let Some((enter, exit)) = bounds.ray_interval(ray) else {
            return Ok(result);
        };
        let exit = exit.min(endpoint);
        if exit <= enter {
            return Ok(result);
        }
        if !enter.is_finite() || !exit.is_finite() {
            return Err(TraceError::NumericalFailure(
                "majorant interval is not finite",
            ));
        }
        // Use the exact same world-point -> index chain as density_world.
        // Transforming origin/direction separately reconstructs boundary points
        // with different rounding and can assign the cell behind the entry.
        let position = volume.world_to_index(ray.at(enter)) / 8.0;
        for axis in 0..3 {
            let mut coordinate = position[axis].floor();
            // A ray on a lattice plane moving toward negative coordinates starts
            // in the cell on the negative side. This changes the traversal label,
            // never the ray origin or physical density evaluation point.
            if direction[axis] < 0.0 && position[axis] == coordinate {
                coordinate -= 1.0;
            }
            if !coordinate.is_finite()
                || coordinate < f64::from(i32::MIN)
                || coordinate > f64::from(i32::MAX)
            {
                return Err(TraceError::NumericalFailure(
                    "majorant cell coordinate overflow",
                ));
            }
            result.cell[axis] = coordinate as i32;
        }
        result.distance = enter;
        result.endpoint = exit;
        result.done = false;
        Ok(result)
    }

    fn crossings(&self) -> [f64; 3] {
        let mut crossing = [f64::INFINITY; 3];
        for (axis, distance) in crossing.iter_mut().enumerate() {
            let direction = self.ray.direction[axis];
            if direction != 0.0 {
                let cell_plane =
                    f64::from(self.cell[axis]) + if direction > 0.0 { 1.0 } else { 0.0 };
                // The bounds and every grid plane use one world-space affine
                // transform. Keep this crossing parameter tied to the original
                // ray; there is no second normalization or accumulated drift.
                let plane = cell_plane * 8.0 * self.volume.transform.scale
                    + self.volume.transform.translation[axis];
                *distance = (plane - self.ray.origin[axis]) / direction;
            }
        }
        crossing
    }

    fn advance_cells(&mut self, crossing: [f64; 3], distance: f64) -> Result<bool, TraceError> {
        let mut advanced = false;
        for (axis, crossing) in crossing.into_iter().enumerate() {
            if crossing <= distance {
                let step = if self.ray.direction[axis] > 0.0 {
                    1
                } else {
                    -1
                };
                self.cell[axis] =
                    self.cell[axis]
                        .checked_add(step)
                        .ok_or(TraceError::NumericalFailure(
                            "majorant traversal cell overflow",
                        ))?;
                advanced = true;
            }
        }
        Ok(advanced)
    }
}

impl Iterator for MajorantSpans<'_> {
    type Item = Result<MajorantSpan, TraceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let crossing = self.crossings();
            let end = crossing.into_iter().fold(self.endpoint, f64::min);
            if crossing.iter().any(|value| value.is_nan()) || !end.is_finite() {
                self.done = true;
                return Some(Err(TraceError::NumericalFailure(
                    "nonfinite majorant crossing",
                )));
            }
            if end <= self.distance {
                // Boundary reconstruction can select the cell on the other
                // side of the current parameter. Only change ownership here:
                // distance stays exactly fixed, so no positive interval is
                // skipped, however small the next crossing separation is.
                match self.advance_cells(crossing, self.distance) {
                    Ok(true) => continue,
                    Ok(false) => {
                        self.done = true;
                        return Some(Err(TraceError::NumericalFailure(
                            "majorant traversal failed to advance",
                        )));
                    }
                    Err(error) => {
                        self.done = true;
                        return Some(Err(error));
                    }
                }
            }
            let span = MajorantSpan {
                start: self.distance,
                end,
                max_density: self.volume.majorant_at_index_cell(self.cell),
            };
            self.distance = end;
            self.done = end >= self.endpoint;
            if !self.done {
                if let Err(error) = self.advance_cells(crossing, end) {
                    self.done = true;
                    return Some(Err(error));
                }
            }
            return Some(Ok(span));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::{BRICK_VOXELS, UniformTransform, VolumeStats};
    use glam::DVec3;
    use std::collections::HashMap;

    fn one_brick(cell: [i32; 3], scale: f64) -> SparseVolume {
        let mut bricks = HashMap::new();
        bricks.insert(cell, Box::new([1.0; BRICK_VOXELS]));
        SparseVolume::new(
            UniformTransform {
                scale,
                translation: DVec3::ZERO,
            },
            VolumeStats::default(),
            bricks,
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn disney_camera_boundary_roundoff_preserves_the_complete_interval() {
        // Captured from the eighth-resolution Disney asset, 8x4 / sample 2.
        let ray = Ray {
            origin: DVec3::new(648.064, -82.473, -63.856),
            direction: DVec3::new(
                -0.8679551565249477,
                0.38697393215038633,
                -0.31129571487224494,
            ),
        };
        let volume = one_brick([14, 8, -17], 1.666_666_626_930_236_8);
        let (enter, exit) = volume.world_bounds().ray_interval(ray).unwrap();
        let spans = MajorantSpans::new(&volume, ray, f64::INFINITY)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!spans.is_empty());
        assert_eq!(spans.first().unwrap().start, enter);
        assert_eq!(spans.last().unwrap().end, exit);
        for span in &spans {
            assert!(span.end > span.start);
        }
        for adjacent in spans.windows(2) {
            assert_eq!(adjacent[0].end, adjacent[1].start);
        }
        assert!(spans.iter().all(|span| span.max_density > 0.0));
    }

    #[test]
    fn near_corner_crossings_keep_the_positive_thin_interval() {
        let volume = one_brick([0, 0, 0], 1.23456789);
        let ray = Ray {
            origin: volume.index_to_world(DVec3::new(-1.0, -1.0, 4.0)),
            direction: DVec3::new(1.0, 1.0 + 1e-14, 0.0).normalize(),
        };
        let (enter, exit) = volume.world_bounds().ray_interval(ray).unwrap();
        let spans = MajorantSpans::new(&volume, ray, f64::INFINITY)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(spans.first().unwrap().start, enter);
        assert_eq!(spans.last().unwrap().end, exit);
        assert!(
            spans
                .iter()
                .any(|span| span.end > span.start && span.end - span.start < 1e-12)
        );
        for adjacent in spans.windows(2) {
            assert_eq!(adjacent[0].end, adjacent[1].start);
        }
    }
}
