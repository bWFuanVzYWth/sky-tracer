//! CPU f32 checks of shader geometry against independently evaluated f64
//! intervals and the represented density. These tests never request a device.
use crate::volume::{
    BRICK_VOXELS, SparseGpuVolume, SparseVolume, UniformTile, UniformTransform, VolumeStats,
    brick_hash,
};
use glam::{DVec3, Vec3};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Distance {
    hi: f32,
    lo: f32,
}
impl Distance {
    fn new(hi: f32) -> Self {
        Self { hi, lo: 0.0 }
    }
    fn value(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }
    fn less(self, rhs: Self) -> bool {
        self.hi < rhs.hi || (self.hi == rhs.hi && self.lo < rhs.lo)
    }
    fn sum(a: f32, b: f32) -> Self {
        let hi = a + b;
        let recovered = hi - a;
        let lost_a = a - (hi - recovered);
        let lost_b = b - recovered;
        Self {
            hi,
            lo: lost_a + lost_b,
        }
    }
    fn divide(self, denominator: f32) -> Self {
        let quotient = self.hi / denominator;
        if !quotient.is_finite() {
            return Self::new(if quotient >= 0.0 { f32::MAX } else { -f32::MAX });
        }
        let remainder = (-quotient).mul_add(denominator, self.hi);
        let tail = (remainder + self.lo) / denominator;
        Self::sum(quotient, tail)
    }
    fn product(a: f32, b: f32) -> Self {
        let hi = a * b;
        Self {
            hi,
            lo: a.mul_add(b, -hi),
        }
    }
    fn add(self, rhs: Self) -> Self {
        let leading = Self::sum(self.hi, rhs.hi);
        let trailing = Self::sum(self.lo, rhs.lo);
        let merged = Self::sum(leading.hi, leading.lo + trailing.hi);
        Self::sum(merged.hi, merged.lo + trailing.lo)
    }
    fn subtract(self, rhs: Self) -> Self {
        self.add(Self {
            hi: -rhs.hi,
            lo: -rhs.lo,
        })
    }
    fn min(self, rhs: Self) -> Self {
        if self.less(rhs) { self } else { rhs }
    }
    fn max(self, rhs: Self) -> Self {
        if self.less(rhs) { rhs } else { self }
    }
}

fn world_plane(index: f32, scale: f32, translation: f32) -> Distance {
    Distance::product(index, scale).add(Distance::new(translation))
}

fn paired_world_interval(
    origin: Vec3,
    direction: Vec3,
    scale: f32,
    translation: Vec3,
    min: Vec3,
    max: Vec3,
) -> Option<(Distance, Distance)> {
    let mut begin = Distance::new(0.0);
    let mut end = Distance::new(f32::MAX);
    for axis in 0..3 {
        let minimum = world_plane(min[axis] - 1.0, scale, translation[axis]);
        let maximum = world_plane(max[axis] + 1.0, scale, translation[axis]);
        let point = Distance::new(origin[axis]);
        if direction[axis] == 0.0 {
            if point.less(minimum) || maximum.less(point) {
                return None;
            }
        } else {
            let a = minimum.subtract(point).divide(direction[axis]);
            let b = maximum.subtract(point).divide(direction[axis]);
            begin = begin.max(a.min(b));
            end = end.min(a.max(b));
            if !begin.less(end) {
                return None;
            }
        }
    }
    Some((begin, end))
}

fn paired_index_at(
    origin: Vec3,
    direction: Vec3,
    distance: Distance,
    scale: f32,
    translation: Vec3,
) -> Vec3 {
    Vec3::from_array(std::array::from_fn(|axis| {
        let displacement = Distance::product(distance.hi, direction[axis])
            .add(Distance::product(distance.lo, direction[axis]));
        let world = Distance::new(origin[axis]).add(displacement);
        let index = world
            .subtract(Distance::new(translation[axis]))
            .divide(scale);
        index.hi + index.lo
    }))
}

fn f64_world_interval(
    origin: Vec3,
    direction: Vec3,
    scale: f32,
    translation: Vec3,
    min: Vec3,
    max: Vec3,
) -> Option<(f64, f64)> {
    let mut enter: f64 = 0.0;
    let mut exit = f64::INFINITY;
    for axis in 0..3 {
        let minimum =
            (f64::from(min[axis]) - 1.0) * f64::from(scale) + f64::from(translation[axis]);
        let maximum =
            (f64::from(max[axis]) + 1.0) * f64::from(scale) + f64::from(translation[axis]);
        let point = f64::from(origin[axis]);
        let direction = f64::from(direction[axis]);
        if direction == 0.0 {
            if point < minimum || point > maximum {
                return None;
            }
        } else {
            let a = (minimum - point) / direction;
            let b = (maximum - point) / direction;
            enter = enter.max(a.min(b));
            exit = exit.min(a.max(b));
            if enter >= exit {
                return None;
            }
        }
    }
    Some((enter, exit))
}

#[test]
fn paired_world_slabs_keep_thin_cloud_hits_beyond_plain_f32_resolution() {
    let origin = Vec3::new(2.0_f32.powi(40), 0.0, 0.0);
    let direction = -Vec3::X;
    let (enter, exit) =
        paired_world_interval(origin, direction, 1.0, Vec3::ZERO, Vec3::ZERO, Vec3::ONE).unwrap();
    assert_eq!(enter.value(), f64::from(origin.x) - 2.0);
    assert_eq!(exit.value(), f64::from(origin.x) + 1.0);
    assert_eq!(exit.subtract(enter).value(), 3.0);
    assert_eq!(
        paired_index_at(origin, direction, enter, 1.0, Vec3::ZERO),
        Vec3::new(2.0, 0.0, 0.0)
    );
    // Ordinary f32 cannot distinguish the positive three-unit interval.
    assert_eq!((origin.x - 2.0).to_bits(), (origin.x + 1.0).to_bits());
}

#[test]
fn far_ground_world_entry_matches_f64_through_the_dynamic_pair_budget() {
    let scale = 1.666_666_6_f32;
    let translation = Vec3::new(53.5, -69.125, 23.875);
    let min = Vec3::new(-137.0, -41.0, -185.0);
    let max = Vec3::new(120.0, 136.0, 128.0);
    let mut largest_index_error: f64 = 0.0;
    for exponent in [18, 20, 24, 28, 32, 36, 40] {
        for sign in [-1.0, 1.0] {
            let origin = Vec3::new(sign * 2.0_f32.powi(exponent) * scale, -1000.0, 5000.0);
            let direction =
                Vec3::new(-sign, 1000.0 / origin.x.abs(), -5000.0 / origin.x.abs()).normalize();
            let (enter, exit) =
                paired_world_interval(origin, direction, scale, translation, min, max).unwrap();
            let (exact_enter, exact_exit) =
                f64_world_interval(origin, direction, scale, translation, min, max).unwrap();
            assert!((enter.value() - exact_enter).abs() / f64::from(scale) < 0.01);
            assert!((exit.value() - exact_exit).abs() / f64::from(scale) < 0.01);
            let index = paired_index_at(origin, direction, enter, scale, translation);
            let exact_world = origin.as_dvec3() + direction.as_dvec3() * exact_enter;
            let exact_index = (exact_world - translation.as_dvec3()) / f64::from(scale);
            let error = (index.as_dvec3() - exact_index).abs().max_element();
            largest_index_error = largest_index_error.max(error);
            assert!(
                error < 0.01,
                "2^{exponent}: index error {error} at {index:?} / {exact_index:?}"
            );
            let span = exit.subtract(enter).value();
            assert!(span > 0.0);
            assert!((span - (exact_exit - exact_enter)).abs() / f64::from(scale) < 0.02);
        }
    }
    eprintln!("far-ground paired entry largest index error: {largest_index_error}");
}

#[test]
fn precision_repro_217_far_ground_sun_shadow_is_a_real_miss_not_an_error() {
    // The recorded runtime fixture fails pixel 20/sample 217 before ray/AABB
    // classification. Its actual continuation origin was not captured. This
    // representative far-ground sun ray reproduces that premature guard while
    // independently proving the correct vacuum visibility.
    let origin = Vec3::new(-500_000.0, -1000.0, 30_000.0);
    let direction = Vec3::new(0.5826, 0.766, 0.2717).normalize();
    let scale = 1.666_666_6_f32;
    let min = Vec3::new(-137.0, -41.0, -185.0);
    let max = Vec3::new(120.0, 136.0, 128.0);
    assert!(origin.x.abs() / scale > 262_144.0);
    assert!(origin.x.abs() / scale < 2.0_f32.powi(40));
    assert!(paired_world_interval(origin, direction, scale, Vec3::ZERO, min, max).is_none());
    assert!(f64_world_interval(origin, direction, scale, Vec3::ZERO, min, max).is_none());
    assert!(Distance::new(-1.0).divide(1e-40).hi < 0.0);
}
#[derive(Debug)]
struct Dda {
    origin: Vec3,
    direction: Vec3,
    cell: [i32; 3],
    step: [i32; 3],
    next: [Distance; 3],
    t: Distance,
}
impl Dda {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        let mut result = Self {
            origin,
            direction,
            cell: (origin / 8.0).floor().as_ivec3().to_array(),
            step: [0; 3],
            next: [Distance::new(f32::MAX); 3],
            t: Distance::new(0.0),
        };
        for axis in 0..3 {
            if direction[axis] != 0.0 {
                result.step[axis] = if direction[axis] > 0.0 { 1 } else { -1 };
                if direction[axis] < 0.0 && origin[axis] == (result.cell[axis] * 8) as f32 {
                    result.cell[axis] -= 1;
                }
                result.next[axis] = result.crossing(axis);
            }
        }
        result
    }
    fn crossing(&self, axis: usize) -> Distance {
        let boundary_cell = self.cell[axis] + i32::from(self.step[axis] > 0);
        Distance::sum((boundary_cell * 8) as f32, -self.origin[axis]).divide(self.direction[axis])
    }
    fn segments(mut self, span: f32) -> Vec<([i32; 3], Distance, Distance)> {
        let endpoint = Distance::new(span);
        let mut result = Vec::new();
        while self.t.less(endpoint) {
            assert!(result.len() < 10_000, "geometry must terminate");
            let mut end = endpoint;
            for crossing in self.next {
                if crossing.less(end) {
                    end = crossing;
                }
            }
            assert!(!end.less(self.t));
            if self.t.less(end) {
                result.push((self.cell, self.t, end));
            }
            if end == endpoint {
                break;
            }
            let mut advanced = false;
            for axis in 0..3 {
                if self.step[axis] != 0 && self.next[axis] == end {
                    let previous = self.next[axis];
                    self.cell[axis] += self.step[axis];
                    self.next[axis] = self.crossing(axis);
                    assert!(previous.less(self.next[axis]));
                    advanced = true;
                }
            }
            assert!(advanced);
            self.t = end;
        }
        assert_eq!(result.first().unwrap().1, Distance::new(0.0));
        assert_eq!(result.last().unwrap().2, endpoint);
        for pair in result.windows(2) {
            assert_eq!(pair[0].2, pair[1].1);
        }
        result
    }
}

#[test]
fn dda_owns_negative_planes_and_advances_all_exact_ties() {
    let origin = Vec3::new(0.0, 8.0, 16.0);
    let dda = Dda::new(origin, Vec3::splat(-1.0));
    assert_eq!(dda.cell, [-1, 0, 1]);
    let segments = dda.segments(40.0);
    assert_eq!(segments.len(), 5);
    for (i, (cell, start, end)) in segments.iter().enumerate() {
        let i = i as i32;
        assert_eq!(*cell, [-1 - i, -i, 1 - i]);
        assert_eq!(start.value(), f64::from(i * 8));
        assert_eq!(end.value(), f64::from((i + 1) * 8));
    }
}

#[test]
fn compensated_dda_preserves_thin_slivers_and_nearly_parallel_axes() {
    let origin = Vec3::new(0.0, 1e-5, 0.0);
    let direction = Vec3::new(1.0, 1.0, 0.0);
    let segments = Dda::new(origin, direction).segments(1024.0);
    let thin = segments
        .iter()
        .filter(|(_, start, end)| end.value() - start.value() < 1e-4)
        .count();
    assert_eq!(thin, 128);
    for (cell, start, end) in &segments {
        let middle =
            origin.as_dvec3() + direction.as_dvec3() * ((start.value() + end.value()) * 0.5);
        assert_eq!((middle / 8.0).floor().as_ivec3().to_array(), *cell);
    }
    let almost_parallel = Dda::new(Vec3::ZERO, Vec3::new(1.0, 1e-40, 0.0)).segments(80.0);
    assert_eq!(almost_parallel.len(), 10);
    assert!(
        almost_parallel
            .iter()
            .all(|(cell, _, _)| cell[1] == 0 && cell[2] == 0)
    );
}

#[test]
fn dda_partitions_f64_geometry_and_every_density_is_below_its_proposal() {
    let mut bricks = HashMap::new();
    let mut leaf = Box::new([0.0; BRICK_VOXELS]);
    for (i, density) in leaf.iter_mut().enumerate() {
        *density = (i % 29) as f32 / 28.0;
    }
    bricks.insert([-1, 0, 0], leaf);
    let volume = SparseVolume::new(
        UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        VolumeStats::default(),
        bricks,
        vec![UniformTile {
            origin: [128, 0, 0],
            size: 128,
            density: 0.4,
        }],
    )
    .unwrap();
    for (origin, direction, span) in [
        (Vec3::new(-16.0, 3.0, 3.0), Vec3::X, 180.0),
        (Vec3::new(148.0, 33.0, 33.0), -Vec3::X, 180.0),
        (Vec3::splat(-16.0), Vec3::ONE.normalize(), 300.0),
        (
            Vec3::new(-12.0, 4.0, 4.0),
            Vec3::new(1.0, 0.1, -0.08).normalize(),
            180.0,
        ),
    ] {
        let segments = Dda::new(origin, direction).segments(span);
        let mut proposal_integral = 0.0;
        for (cell, start, end) in segments {
            let bound = volume.majorant_at_index_cell(cell);
            proposal_integral += bound * (end.value() - start.value());
            for fraction in [0.001, 0.25, 0.5, 0.75, 0.999] {
                let t = start.value() + (end.value() - start.value()) * fraction;
                let exact_point = origin.as_dvec3() + direction.as_dvec3() * t;
                assert_eq!((exact_point / 8.0).floor().as_ivec3().to_array(), cell);
                assert!(volume.trilinear(exact_point) <= bound);
            }
        }
        assert!(
            proposal_integral < f64::from(volume.pack_gpu().unwrap().majorant) * f64::from(span)
        );
    }
}

fn find_brick(volume: &SparseGpuVolume, brick: [i32; 3], lookups: &mut u32) -> Option<usize> {
    *lookups += 1;
    let mut slot = brick_hash(brick) as usize & (volume.hash.len() - 1);
    loop {
        let entry = volume.hash[slot];
        if entry[3] == u32::MAX {
            return None;
        }
        if entry[..3] == brick.map(|v| v as u32) {
            return Some(entry[3] as usize);
        }
        slot = (slot + 1) & (volume.hash.len() - 1);
    }
}
fn uniform_density(volume: &SparseGpuVolume, index: [i32; 3]) -> f32 {
    for tile in &volume.tiles {
        if (0..3).all(|i| {
            index[i] >= tile[i] as i32
                && i64::from(index[i]) - i64::from(tile[i] as i32) < i64::from(tile[3])
        }) {
            return f32::from_bits(tile[4]);
        }
    }
    0.0
}
fn packed_lattice(volume: &SparseGpuVolume, index: [i32; 3], lookups: &mut u32) -> f32 {
    let brick = index.map(|v| v.div_euclid(8));
    if let Some(offset) = find_brick(volume, brick, lookups) {
        let local = index.map(|v| v.rem_euclid(8) as usize);
        volume.values[offset + local[0] * 64 + local[1] * 8 + local[2]]
    } else {
        uniform_density(volume, index)
    }
}
fn interpolate(c: [f32; 8], f: Vec3) -> f32 {
    fn mix(a: f32, b: f32, f: f32) -> f32 {
        a * (1.0 - f) + b * f
    }
    let z00 = mix(c[0], c[1], f.z);
    let z01 = mix(c[2], c[3], f.z);
    let z10 = mix(c[4], c[5], f.z);
    let z11 = mix(c[6], c[7], f.z);
    mix(mix(z00, z01, f.y), mix(z10, z11, f.y), f.x)
}
fn packed_trilinear(volume: &SparseGpuVolume, q: Vec3, fast: bool) -> (f32, u32) {
    let base = q.floor().as_ivec3().to_array();
    let frac = q - q.floor();
    let brick = base.map(|v| v.div_euclid(8));
    let local = base.map(|v| v.rem_euclid(8) as usize);
    let mut lookups = 0;
    let corners = if fast && local.into_iter().all(|v| v < 7) {
        if let Some(offset) = find_brick(volume, brick, &mut lookups) {
            let i = offset + local[0] * 64 + local[1] * 8 + local[2];
            [0usize, 1, 8, 9, 64, 65, 72, 73].map(|delta| volume.values[i + delta])
        } else {
            [uniform_density(volume, base); 8]
        }
    } else {
        std::array::from_fn(|i| {
            packed_lattice(
                volume,
                [
                    base[0] + (i / 4) as i32,
                    base[1] + (i / 2 % 2) as i32,
                    base[2] + (i % 2) as i32,
                ],
                &mut lookups,
            )
        })
    };
    (interpolate(corners, frac), lookups)
}

#[test]
fn one_brick_interpolation_preserves_all_corners_and_uses_fewer_hash_lookups() {
    let mut rng = crate::sampling::Pcg32::for_sample(921, 0, 0);
    let mut bricks = HashMap::new();
    for coord in [[-2, -1, 0], [-1, 0, 0], [16, 1, 2]] {
        let mut values = Box::new([0.0; BRICK_VOXELS]);
        for (i, value) in values.iter_mut().enumerate() {
            // Retain nonzero payload even when it would be inactive in VDB.
            *value = if i % 3 == 0 { 0.0 } else { rng.open01() as f32 };
        }
        bricks.insert(coord, values);
    }
    let volume = SparseVolume::new(
        UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        VolumeStats::default(),
        bricks,
        vec![
            UniformTile {
                origin: [-24, 0, 0],
                size: 8,
                density: 0.5,
            },
            UniformTile {
                origin: [128, 0, 0],
                size: 128,
                density: 0.4,
            },
        ],
    )
    .unwrap();
    let packed = volume.pack_gpu().unwrap();
    let mut fast_lookups = 0u64;
    let mut slow_lookups = 0u64;
    let mut fast_points = 0u64;
    let count = 10_000;
    for i in 0..count {
        let origin = match i % 5 {
            0 => Vec3::new(-16.0, -8.0, 0.0),
            1 => Vec3::new(-8.0, 0.0, 0.0),
            2 => Vec3::new(-24.0, 0.0, 0.0),
            3 => Vec3::new(128.0, 8.0, 16.0),
            _ => Vec3::new(240.0, 120.0, 120.0),
        };
        let q = origin
            + Vec3::new(
                rng.open01() as f32,
                rng.open01() as f32,
                rng.open01() as f32,
            ) * 8.0;
        let (fast, nfast) = packed_trilinear(&packed, q, true);
        let (slow, nslow) = packed_trilinear(&packed, q, false);
        assert_eq!(fast.to_bits(), slow.to_bits(), "{q:?}");
        assert!((f64::from(fast) - volume.trilinear(q.as_dvec3())).abs() < 2e-7);
        fast_lookups += u64::from(nfast);
        slow_lookups += u64::from(nslow);
        fast_points += u64::from(nfast == 1);
    }
    // A uniformly sampled brick lands wholly within it with probability (7/8)^3.
    assert!((fast_points as f64 / f64::from(count) - (7.0f64 / 8.0).powi(3)).abs() < 0.02);
    assert!(fast_lookups < slow_lookups / 2);
    for boundary in [
        -24.0f32, -16.0, -8.0, 0.0, 7.0, 8.0, 127.0, 128.0, 255.0, 256.0,
    ] {
        for delta in [-1e-4f32, 0.0, 1e-4] {
            for axis in 0..3 {
                let mut point = Vec3::new(-7.0, 3.0, 3.0);
                point[axis] = boundary + delta;
                assert_eq!(
                    packed_trilinear(&packed, point, true).0.to_bits(),
                    packed_trilinear(&packed, point, false).0.to_bits()
                );
            }
        }
    }
}

fn packed_majorant(volume: &SparseGpuVolume, cell: [i32; 3]) -> f32 {
    let mut slot = brick_hash(cell) as usize & (volume.majorant_hash.len() - 1);
    let mut maximum = 0.0f32;
    loop {
        let entry = volume.majorant_hash[slot];
        if entry[3] == u32::MAX {
            break;
        }
        if entry[..3] == cell.map(|v| v as u32) {
            maximum = f32::from_bits(entry[3]);
            break;
        }
        slot = (slot + 1) & (volume.majorant_hash.len() - 1);
    }
    for tile in &volume.tiles {
        if (0..3).all(|axis| {
            let min = i64::from(cell[axis]) * 8 - 1;
            let max = min + 10;
            let origin = i64::from(tile[axis] as i32);
            min <= origin + i64::from(tile[3]) - 1 && max >= origin
        }) {
            maximum = maximum.max(f32::from_bits(tile[4]));
        }
    }
    maximum * (1.0 + 1.0 / 1_048_576.0)
}

#[test]
fn nonunit_transform_entry_uses_one_shared_index_coordinate_chain() {
    let scale = 1.666_666_6f32;
    let direction = Vec3::new(-0.762_063_26, 0.647_502_6, 0.0);
    assert_eq!(direction.length_squared(), 1.0);
    let origin = Vec3::new(806.529, -670.621_6, 3.333_333_3);
    let index_origin = origin / scale;
    let index_direction = direction / scale;
    let enter = (8.0 - index_origin.x) / index_direction.x;
    let world_entry = origin + enter * direction;
    let shared_index_entry = world_entry / scale;
    let old_separate_entry = index_origin + enter * index_direction;
    assert!(old_separate_entry.x > 8.0);
    assert!(shared_index_entry.x < 8.0 && shared_index_entry.x > 7.0);
    assert_eq!(Dda::new(old_separate_entry, index_direction).cell[0], 1);
    assert_eq!(Dda::new(shared_index_entry, index_direction).cell[0], 0);
    let mut bricks = HashMap::new();
    bricks.insert([0, 0, 0], Box::new([1.0; BRICK_VOXELS]));
    let volume = SparseVolume::new(
        UniformTransform {
            scale: f64::from(scale),
            translation: DVec3::ZERO,
        },
        VolumeStats::default(),
        bricks,
        Vec::new(),
    )
    .unwrap();
    assert_eq!(volume.majorant_at_index_cell([1, 0, 0]), 0.0);
    assert!(volume.trilinear(shared_index_entry.as_dvec3()) > 0.0);
    // Padding independently protects this thin boundary region even if another
    // f32 operation moves a density position into the neighboring brick cell.
    let packed = volume.pack_gpu().unwrap();
    assert!(packed_majorant(&packed, [1, 0, 0]) >= 1.0);
}

#[test]
fn gpu_halo_bounds_one_voxel_position_errors_at_cell_boundaries() {
    let mut bricks = HashMap::new();
    let mut leaf = Box::new([0.0; BRICK_VOXELS]);
    for (i, density) in leaf.iter_mut().enumerate() {
        *density = (i % 17) as f32 / 16.0;
    }
    bricks.insert([30_000, -1, 0], leaf);
    let volume = SparseVolume::new(
        UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        },
        VolumeStats::default(),
        bricks,
        vec![UniformTile {
            origin: [128, 0, 0],
            size: 128,
            density: 0.4,
        }],
    )
    .unwrap();
    let packed = volume.pack_gpu().unwrap();
    assert_eq!(volume.majorant_cell_count(), 8);
    assert_eq!(volume.gpu_majorant_cell_count(), 27);
    for cell in [
        [29_999, -2, -1],
        [30_000, -1, 0],
        [30_001, 0, 1],
        [32, 0, 0],
    ] {
        let bound = packed_majorant(&packed, cell);
        for x in [-0.75, 0.0, 4.0, 7.999, 8.75] {
            for y in [-0.75, 0.0, 4.0, 7.999, 8.75] {
                for z in [-0.75, 0.0, 4.0, 7.999, 8.75] {
                    let q = Vec3::from_array(cell.map(|v| v as f32 * 8.0)) + Vec3::new(x, y, z);
                    let rho = packed_trilinear(&packed, q, true).0;
                    assert!(
                        rho <= bound,
                        "cell {cell:?}: rho {rho} > bound {bound} at {q:?}"
                    );
                }
            }
        }
    }
    assert_eq!(packed_majorant(&packed, [30_002, 0, 0]), 0.0);
    // A large tile's positive boundary is protected without expanding the tile.
    assert_eq!(volume.majorant_at_index_cell([32, 0, 0]), 0.0);
    assert!(packed_trilinear(&packed, Vec3::new(255.75, 2.0, 2.0), true).0 > 0.0);
    assert!(packed_majorant(&packed, [32, 0, 0]) >= 0.4);
}

#[test]
fn f32_sample_positions_stay_inside_the_padded_cell_envelope() {
    for (origin, direction, span) in [
        (
            Vec3::new(240_000.0, -8.0, 0.0),
            Vec3::new(1.0, 1.0e-5, 0.04).normalize(),
            512.0,
        ),
        (
            Vec3::new(-240_000.0, 8.0, 0.0),
            Vec3::new(-1.0, -1.0e-5, -0.04).normalize(),
            512.0,
        ),
        (Vec3::new(0.0, 1.0e-5, 0.0), Vec3::ONE.normalize(), 2048.0),
    ] {
        for (cell, start, end) in Dda::new(origin, direction).segments(span) {
            for fraction in [0.00001, 0.5, 0.99999] {
                let t = start.value() + (end.value() - start.value()) * fraction;
                let pair = Distance::sum(t as f32, (t - f64::from(t as f32)) as f32);
                let ordinary = (origin + pair.hi * direction) + pair.lo * direction;
                let fused = Vec3::from_array(std::array::from_fn(|axis| {
                    direction[axis].mul_add(pair.lo, direction[axis].mul_add(pair.hi, origin[axis]))
                }));
                for point in [ordinary, fused] {
                    let min = Vec3::from_array(cell.map(|v| v as f32 * 8.0)) - Vec3::ONE;
                    let max = min + Vec3::splat(10.0);
                    assert!(
                        point.cmpge(min).all() && point.cmple(max).all(),
                        "{point:?} outside padded cell {cell:?}"
                    );
                    let exact = origin.as_dvec3() + direction.as_dvec3() * t;
                    assert!((point.as_dvec3() - exact).abs().max_element() < 1.0);
                }
            }
        }
    }
}
