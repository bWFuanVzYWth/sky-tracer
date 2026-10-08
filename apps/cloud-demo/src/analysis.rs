//! Reproducible CPU-only estimate of proposal cost; never draws path samples.

use cloud_pt::{
    Result,
    config::{Camera, RenderConfig},
    transport::{Ray, TransportSettings},
    volume::SparseVolume,
};
use serde::Serialize;

#[derive(Default, Serialize)]
pub struct RayCosts {
    rays: u64,
    intersecting_rays: u64,
    spans: u64,
    empty_spans: u64,
    empty_distance: f64,
    medium_distance: f64,
    expected_global_candidates: f64,
    expected_spatial_candidates: f64,
    spatial_to_global_ratio: Option<f64>,
}

impl RayCosts {
    fn add(&mut self, volume: &SparseVolume, ray: Ray, scale: f64) -> Result<()> {
        self.rays += 1;
        let mut hit = false;
        for span in volume.majorant_spans(ray, f64::INFINITY)? {
            let span = span?;
            hit = true;
            let length = span.end - span.start;
            self.spans += 1;
            self.medium_distance += length;
            self.expected_global_candidates += length * volume.maximum_density() * scale;
            self.expected_spatial_candidates += length * span.max_density * scale;
            if span.max_density == 0.0 {
                self.empty_spans += 1;
                self.empty_distance += length;
            }
        }
        self.intersecting_rays += u64::from(hit);
        Ok(())
    }
    fn finish(&mut self) {
        self.spatial_to_global_ratio = (self.expected_global_candidates > 0.0)
            .then(|| self.expected_spatial_candidates / self.expected_global_candidates);
    }
}

#[derive(Serialize)]
pub struct Analysis {
    camera_ray_grid: [u32; 2],
    camera_full_segments: RayCosts,
    sun_from_camera_segment_midpoints: RayCosts,
    sparse_majorant_cells: usize,
    estimated_gpu_storage_bytes: Option<u64>,
}

pub fn analyze(
    volume: &SparseVolume,
    camera: &Camera,
    render: &RenderConfig,
    settings: &TransportSettings,
    rays_x: u32,
    rays_y: u32,
) -> Result<Analysis> {
    if rays_x == 0 || rays_y == 0 || u64::from(rays_x) * u64::from(rays_y) > 1_000_000 {
        return Err("analysis ray grid must be positive with at most 1,000,000 rays".into());
    }
    render.validate()?;
    settings.validate()?;
    let mut camera_costs = RayCosts::default();
    let mut sun_costs = RayCosts::default();
    let sun = settings.sun_direction.normalize();
    for y in 0..rays_y {
        for x in 0..rays_x {
            let px = (f64::from(x) + 0.5) * f64::from(render.width) / f64::from(rays_x);
            let py = (f64::from(y) + 0.5) * f64::from(render.height) / f64::from(rays_y);
            let ray = camera.ray(render.width, render.height, px, py)?;
            camera_costs.add(volume, ray, settings.extinction_scale)?;
            if let Some((enter, exit)) = volume.world_bounds().ray_interval(ray) {
                let shadow = Ray {
                    origin: ray.at(enter + (exit - enter) * 0.5),
                    direction: sun,
                };
                sun_costs.add(volume, shadow, settings.extinction_scale)?;
            }
        }
    }
    camera_costs.finish();
    sun_costs.finish();
    Ok(Analysis {
        camera_ray_grid: [rays_x, rays_y],
        camera_full_segments: camera_costs,
        sun_from_camera_segment_midpoints: sun_costs,
        sparse_majorant_cells: volume.majorant_cell_count(),
        estimated_gpu_storage_bytes: volume.gpu_storage_bytes(),
    })
}
