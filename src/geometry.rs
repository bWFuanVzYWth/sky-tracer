use crate::math::{Ray, Vec3};

pub const RAY_EPSILON_KM: f32 = 1.0e-4;

#[derive(Clone, Copy, Debug)]
pub struct Planet {
    pub ground_radius_km: f32,
    pub atmosphere_radius_km: f32,
}

impl Planet {
    pub fn earth_reference() -> Self {
        Self {
            ground_radius_km: 6360.0,
            atmosphere_radius_km: 6480.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryKind {
    Ground,
    AtmosphereExit,
}

#[derive(Clone, Copy, Debug)]
pub struct BoundaryHit {
    pub t_km: f32,
    pub kind: BoundaryKind,
}

pub fn altitude_km(planet: Planet, position: Vec3) -> f32 {
    position.length() - planet.ground_radius_km
}

pub fn surface_normal(position: Vec3) -> Vec3 {
    position.normalized()
}

pub fn observer_position(planet: Planet) -> Vec3 {
    Vec3::new(0.0, planet.ground_radius_km + 0.001, 0.0)
}

pub fn intersect_sphere(ray: Ray, radius: f32) -> Option<(f32, f32)> {
    let b = ray.origin.dot(ray.dir);
    let c = ray.origin.dot(ray.origin) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }

    let s = discriminant.sqrt();
    Some((-b - s, -b + s))
}

pub fn next_boundary(ray: Ray, planet: Planet) -> Option<BoundaryHit> {
    let (_, atmosphere_exit) = intersect_sphere(ray, planet.atmosphere_radius_km)?;
    let ground_hit =
        intersect_sphere(ray, planet.ground_radius_km).and_then(|(t0, t1)| positive_root(t0, t1));

    let t_exit = atmosphere_exit.max(RAY_EPSILON_KM);
    if let Some(t_ground) = ground_hit
        && t_ground > RAY_EPSILON_KM
        && t_ground < t_exit
    {
        return Some(BoundaryHit {
            t_km: t_ground,
            kind: BoundaryKind::Ground,
        });
    }

    Some(BoundaryHit {
        t_km: t_exit,
        kind: BoundaryKind::AtmosphereExit,
    })
}

pub fn segment_to_sun_or_space(origin: Vec3, dir: Vec3, planet: Planet) -> Option<f32> {
    let ray = Ray::new(origin + dir * RAY_EPSILON_KM, dir);
    let hit = next_boundary(ray, planet)?;
    match hit.kind {
        BoundaryKind::Ground => None,
        BoundaryKind::AtmosphereExit => Some(hit.t_km),
    }
}

fn positive_root(t0: f32, t1: f32) -> Option<f32> {
    if t0 > RAY_EPSILON_KM {
        Some(t0)
    } else if t1 > RAY_EPSILON_KM {
        Some(t1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upward_ray_exits_atmosphere() {
        let planet = Planet::earth_reference();
        let ray = Ray::new(observer_position(planet), Vec3::Y);
        let hit = next_boundary(ray, planet).unwrap();
        assert_eq!(hit.kind, BoundaryKind::AtmosphereExit);
        assert!((hit.t_km - 119.999).abs() < 0.01);
    }

    #[test]
    fn downward_ray_hits_ground() {
        let planet = Planet::earth_reference();
        let ray = Ray::new(observer_position(planet), -Vec3::Y);
        let hit = next_boundary(ray, planet).unwrap();
        assert_eq!(hit.kind, BoundaryKind::Ground);
        assert!(hit.t_km < 0.01);
    }
}
