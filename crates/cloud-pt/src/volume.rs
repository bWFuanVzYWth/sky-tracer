//! Sparse density on OpenVDB's integer sample lattice.
//!
//! Every value in a leaf is retained, including inactive values. Uniform tiles
//! are stored separately rather than expanded into millions of voxels.

use crate::majorant::MajorantSpans;
use crate::transport::{Bounds, DensityField, Ray, TraceError};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const BRICK_SIZE: usize = 8;
pub const BRICK_VOXELS: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UniformTransform {
    pub scale: f64,
    pub translation: DVec3,
}

impl UniformTransform {
    pub fn index_to_world(self, index: DVec3) -> DVec3 {
        index * self.scale + self.translation
    }

    pub fn world_to_index(self, world: DVec3) -> DVec3 {
        (world - self.translation) / self.scale
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UniformTile {
    pub origin: [i32; 3],
    pub size: i32,
    pub density: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VolumeStats {
    pub grid_name: String,
    pub file_version: u32,
    pub compression: u32,
    pub saved_as_half: bool,
    pub background: f32,
    pub leaf_count: usize,
    pub stored_brick_count: usize,
    pub active_voxel_count: u64,
    pub nonzero_voxel_count: u64,
    pub inactive_nonzero_voxel_count: u64,
    pub nonzero_tile_count: usize,
    pub maximum_density: f32,
    pub metadata_bbox_min: Option<[i32; 3]>,
    pub metadata_bbox_max: Option<[i32; 3]>,
}

#[derive(Debug)]
pub struct SparseVolume {
    pub transform: UniformTransform,
    pub stats: VolumeStats,
    bricks: HashMap<[i32; 3], Box<[f32; BRICK_VOXELS]>>,
    large_tiles: Vec<UniformTile>,
    tile_lookup: [HashMap<[i32; 3], f32>; 3],
    cell_majorants: HashMap<[i32; 3], f32>,
    gpu_cell_majorants: HashMap<[i32; 3], f32>,
    index_bounds: Bounds,
    majorant: f32,
}

impl SparseVolume {
    pub(crate) fn new(
        transform: UniformTransform,
        mut stats: VolumeStats,
        bricks: HashMap<[i32; 3], Box<[f32; BRICK_VOXELS]>>,
        tiles: Vec<UniformTile>,
    ) -> Result<Self, String> {
        if !transform.scale.is_finite()
            || transform.scale <= 0.0
            || !transform.translation.is_finite()
        {
            return Err("cloud transform must have a positive finite uniform scale".into());
        }
        if stats.background != 0.0 {
            return Err("a nonzero VDB background describes an infinite medium; finite cloud bounds are required".into());
        }
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut maximum = 0.0f32;
        let mut cell_majorants = HashMap::new();
        let mut tile_lookup: [HashMap<[i32; 3], f32>; 3] = std::array::from_fn(|_| HashMap::new());
        for (coord, values) in &bricks {
            let mut brick_maximum = 0.0f32;
            for value in values.iter() {
                if !value.is_finite() || *value < 0.0 {
                    return Err("density values must be finite and nonnegative".into());
                }
                maximum = maximum.max(*value);
                brick_maximum = brick_maximum.max(*value);
            }
            contribute_majorant(&mut cell_majorants, *coord, brick_maximum)?;
            let origin = DVec3::new(coord[0] as f64, coord[1] as f64, coord[2] as f64) * 8.0;
            // Linear reconstruction has one voxel of support outside its samples.
            min = min.min(origin - DVec3::ONE);
            max = max.max(origin + DVec3::splat(8.0));
        }
        for tile in &tiles {
            if !tile.density.is_finite() || tile.density < 0.0 || tile.size <= 0 {
                return Err("uniform density tiles must be finite and nonnegative".into());
            }
            maximum = maximum.max(tile.density);
            let level = match tile.size {
                8 => 0,
                128 => 1,
                4096 => 2,
                _ => return Err("unsupported uniform tile size".into()),
            };
            if tile.origin.iter().any(|v| v.rem_euclid(tile.size) != 0) {
                return Err("unaligned uniform tile origin".into());
            }
            let tile_coord = tile.origin.map(|v| v.div_euclid(tile.size));
            if tile_lookup[level]
                .insert(tile_coord, tile.density)
                .is_some()
            {
                return Err("duplicate uniform tile coordinate".into());
            }
            if tile.size == 8 && bricks.contains_key(&tile_coord) {
                return Err("uniform tile overlaps a density leaf".into());
            }
            if tile.size == 8 {
                contribute_majorant(
                    &mut cell_majorants,
                    tile.origin.map(|v| v.div_euclid(8)),
                    tile.density,
                )?;
            }
            let origin = DVec3::from_array(tile.origin.map(f64::from));
            min = min.min(origin - DVec3::ONE);
            max = max.max(origin + DVec3::splat(tile.size as f64));
        }
        if maximum == 0.0 {
            return Err("selected VDB grid has no positive density".into());
        }
        if min.min_element() < f64::from(i32::MIN) || max.max_element() > f64::from(i32::MAX - 1) {
            return Err("density support exceeds safe signed voxel addressing".into());
        }
        let world_bounds = Bounds {
            min: transform.index_to_world(min),
            max: transform.index_to_world(max),
        };
        if !world_bounds.valid() {
            return Err("density world bounds are not finite and nonempty".into());
        }
        let majorant = maximum.next_up();
        if !majorant.is_finite() {
            return Err("density maximum cannot be represented by a finite majorant".into());
        }
        stats.stored_brick_count = bricks.len();
        stats.nonzero_tile_count = tiles.len();
        stats.maximum_density = maximum;
        let mut large_tiles: Vec<_> = tiles.into_iter().filter(|tile| tile.size != 8).collect();
        // Match CPU lookup precedence for a manually constructed overlapping
        // hierarchy; canonical OpenVDB trees themselves contain no overlaps.
        large_tiles.sort_by_key(|tile| tile.size);
        for maximum in cell_majorants.values_mut() {
            *maximum = maximum.next_up();
        }
        // GPU positions and cell-plane distances use f32. A one-voxel position
        // error can reach the previous sample brick, so add the positive-side
        // halo only to GPU proposals. Combined with the ordinary negative halo,
        // this covers sample bricks c-1, c and c+1 on each axis. The CPU's exact
        // trilinear field and tighter proposal map are unchanged.
        let mut gpu_cell_majorants = HashMap::new();
        for (cell, maximum) in &cell_majorants {
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        let offset = [x, y, z];
                        let mut padded = [0; 3];
                        for axis in 0..3 {
                            padded[axis] = cell[axis]
                                .checked_add(offset[axis])
                                .ok_or("GPU majorant halo coordinate overflow")?;
                        }
                        let entry = gpu_cell_majorants.entry(padded).or_insert(0.0f32);
                        *entry = entry.max(*maximum);
                    }
                }
            }
        }
        Ok(Self {
            transform,
            stats,
            bricks,
            large_tiles,
            tile_lookup,
            cell_majorants,
            gpu_cell_majorants,
            index_bounds: Bounds { min, max },
            majorant,
        })
    }

    pub fn index_bounds(&self) -> Bounds {
        self.index_bounds
    }

    pub fn world_bounds(&self) -> Bounds {
        Bounds {
            min: self.transform.index_to_world(self.index_bounds.min),
            max: self.transform.index_to_world(self.index_bounds.max),
        }
    }

    pub fn world_to_index(&self, world: DVec3) -> DVec3 {
        self.transform.world_to_index(world)
    }

    pub fn index_to_world(&self, index: DVec3) -> DVec3 {
        self.transform.index_to_world(index)
    }

    pub fn maximum_density(&self) -> f64 {
        self.majorant as f64
    }

    /// A conservative maximum on index cell [8*c, 8*c+8]. Linear reconstruction
    /// can read the brick at c or c+1 on each axis, so the bound includes all
    /// eight adjacent sample bricks, including inactive values and tiles.
    pub fn majorant_at_index_cell(&self, cell: [i32; 3]) -> f64 {
        let mut maximum = self.cell_majorants.get(&cell).copied().unwrap_or(0.0);
        // Large tiles stay compact. Their sample support is tested against the
        // cell's closed lattice interval rather than expanded into brick cells.
        for tile in &self.large_tiles {
            if (0..3).all(|axis| {
                let min = i64::from(cell[axis]) * 8;
                let max = min + 8;
                let tile_min = i64::from(tile.origin[axis]);
                let tile_max = tile_min + i64::from(tile.size) - 1;
                min <= tile_max && max >= tile_min
            }) {
                maximum = maximum.max(tile.density.next_up());
            }
        }
        f64::from(maximum)
    }

    pub fn majorant_cell_count(&self) -> usize {
        self.cell_majorants.len()
    }

    pub fn gpu_majorant_cell_count(&self) -> usize {
        self.gpu_cell_majorants.len()
    }

    /// Traverse conservative, constant-majorant spans for a unit-direction ray
    /// in physical world distance, preserving the exact supplied direction.
    /// Empty spans have exactly zero density and require no stochastic candidates.
    pub fn majorant_spans(&self, ray: Ray, endpoint: f64) -> Result<MajorantSpans<'_>, TraceError> {
        MajorantSpans::new(self, ray, endpoint)
    }

    pub fn sample_at_index(&self, index: [i32; 3]) -> f32 {
        let brick = index.map(|v| v.div_euclid(8));
        if let Some(values) = self.bricks.get(&brick) {
            let local = index.map(|v| v.rem_euclid(8) as usize);
            return values[local[0] * 64 + local[1] * 8 + local[2]];
        }
        self.tile_value_at_index(index)
    }

    fn tile_value_at_index(&self, index: [i32; 3]) -> f32 {
        for (level, size) in [8, 128, 4096].into_iter().enumerate() {
            if self.tile_lookup[level].is_empty() {
                continue;
            }
            if let Some(density) = self.tile_lookup[level].get(&index.map(|v| v.div_euclid(size))) {
                return *density;
            }
        }
        0.0
    }

    pub fn trilinear(&self, index: DVec3) -> f64 {
        if !index.is_finite()
            || (0..3).any(|axis| {
                index[axis] < self.index_bounds.min[axis]
                    || index[axis] > self.index_bounds.max[axis]
            })
        {
            return 0.0;
        }
        let floor = index.floor();
        let base = floor.as_ivec3().to_array();
        let fraction = index - floor;
        let local = base.map(|v| v.rem_euclid(8) as usize);
        let mut corners = [0.0f32; 8];
        if local.iter().all(|v| *v < 7) {
            // All eight lattice samples live in one leaf. Hash it once and
            // fetch contiguous leaf values; inactive values are ordinary samples.
            let brick = base.map(|v| v.div_euclid(8));
            if let Some(values) = self.bricks.get(&brick) {
                let offset = local[0] * 64 + local[1] * 8 + local[2];
                for x in 0..2 {
                    for y in 0..2 {
                        for z in 0..2 {
                            corners[x * 4 + y * 2 + z] = values[offset + x * 64 + y * 8 + z];
                        }
                    }
                }
            } else {
                // Uniform tiles and absent leaves are constant over this brick.
                corners.fill(self.tile_value_at_index(base));
            }
        } else {
            // Keep the general cross-brick path, including tile/leaf boundaries.
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        corners[x * 4 + y * 2 + z] = self.sample_at_index([
                            base[0] + x as i32,
                            base[1] + y as i32,
                            base[2] + z as i32,
                        ]);
                    }
                }
            }
        }
        let mut result = 0.0;
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    let offset = [x, y, z];
                    let mut weight = 1.0;
                    for axis in 0..3 {
                        weight *= if offset[axis] == 0 {
                            1.0 - fraction[axis]
                        } else {
                            fraction[axis]
                        };
                    }
                    result += weight * f64::from(corners[x * 4 + y * 2 + z]);
                }
            }
        }
        result
    }

    #[cfg(test)]
    fn trilinear_reference(&self, index: DVec3) -> f64 {
        if !index.is_finite()
            || (0..3).any(|axis| {
                index[axis] < self.index_bounds.min[axis]
                    || index[axis] > self.index_bounds.max[axis]
            })
        {
            return 0.0;
        }
        let floor = index.floor();
        let base = floor.as_ivec3().to_array();
        let fraction = index - floor;
        let mut result = 0.0;
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    let offset = [x, y, z];
                    let mut weight = 1.0;
                    for axis in 0..3 {
                        weight *= if offset[axis] == 0 {
                            1.0 - fraction[axis]
                        } else {
                            fraction[axis]
                        };
                    }
                    result += weight
                        * f64::from(self.sample_at_index([base[0] + x, base[1] + y, base[2] + z]));
                }
            }
        }
        result
    }

    pub fn density_world(&self, world: DVec3) -> f64 {
        self.trilinear(self.world_to_index(world))
    }

    pub fn gpu_storage_bytes(&self) -> Option<u64> {
        let brick_count = self.bricks.len().checked_add(self.tile_lookup[0].len())?;
        let tile_count = self.large_tiles.len();
        let capacity = brick_count
            .checked_mul(2)?
            .max(1)
            .checked_next_power_of_two()?;
        let values = brick_count.checked_mul(BRICK_VOXELS)?.checked_mul(4)?;
        let bytes = capacity
            .checked_mul(16)?
            .checked_add(values)?
            .checked_add(tile_count.checked_mul(32)?)?
            .checked_add(
                self.gpu_cell_majorants
                    .len()
                    .checked_mul(2)?
                    .max(1)
                    .checked_next_power_of_two()?
                    .checked_mul(16)?,
            )?;
        u64::try_from(bytes).ok()
    }

    pub fn pack_gpu(&self) -> Result<SparseGpuVolume, String> {
        let brick_count = self
            .bricks
            .len()
            .checked_add(self.tile_lookup[0].len())
            .ok_or("GPU brick count overflow")?;
        let capacity = brick_count
            .checked_mul(2)
            .and_then(|v| v.max(1).checked_next_power_of_two())
            .ok_or("GPU brick hash capacity overflow")?;
        let mut hash = vec![[0, 0, 0, u32::MAX]; capacity];
        let mut values = Vec::with_capacity(
            brick_count
                .checked_mul(BRICK_VOXELS)
                .ok_or("GPU value capacity overflow")?,
        );
        // Stable packing makes dumps and CPU/GPU comparisons reproducible.
        // An 8^3 uniform tile is small enough to pack as an ordinary exact
        // brick. This avoids thousands of fallback tile comparisons for the
        // Disney cloud without altering the reconstructed density field.
        let mut coords: Vec<_> = self
            .bricks
            .keys()
            .chain(self.tile_lookup[0].keys())
            .copied()
            .collect();
        coords.sort_unstable();
        for coord in coords {
            let offset =
                u32::try_from(values.len()).map_err(|_| "GPU density address exceeds u32")?;
            let mut slot = brick_hash(coord) as usize & (capacity - 1);
            while hash[slot][3] != u32::MAX {
                slot = (slot + 1) & (capacity - 1);
            }
            hash[slot] = [coord[0] as u32, coord[1] as u32, coord[2] as u32, offset];
            if let Some(brick) = self.bricks.get(&coord) {
                values.extend_from_slice(brick.as_slice());
            } else {
                values.extend(std::iter::repeat_n(
                    self.tile_lookup[0][&coord],
                    BRICK_VOXELS,
                ));
            }
        }
        let tiles = self
            .large_tiles
            .iter()
            .map(|tile| {
                [
                    tile.origin[0] as u32,
                    tile.origin[1] as u32,
                    tile.origin[2] as u32,
                    tile.size as u32,
                    tile.density.to_bits(),
                    0,
                    0,
                    0,
                ]
            })
            .collect();
        let majorant_capacity = self
            .gpu_cell_majorants
            .len()
            .checked_mul(2)
            .and_then(|v| v.max(1).checked_next_power_of_two())
            .ok_or("GPU majorant hash capacity overflow")?;
        let mut majorant_hash = vec![[0, 0, 0, u32::MAX]; majorant_capacity];
        let mut cells: Vec<_> = self.gpu_cell_majorants.keys().copied().collect();
        cells.sort_unstable();
        for coord in cells {
            let mut slot = brick_hash(coord) as usize & (majorant_capacity - 1);
            while majorant_hash[slot][3] != u32::MAX {
                slot = (slot + 1) & (majorant_capacity - 1);
            }
            majorant_hash[slot] = [
                coord[0] as u32,
                coord[1] as u32,
                coord[2] as u32,
                self.gpu_cell_majorants[&coord].to_bits(),
            ];
        }
        Ok(SparseGpuVolume {
            hash,
            values,
            tiles,
            majorant_hash,
            transform: self.transform,
            index_bounds: self.index_bounds,
            majorant: self.majorant,
        })
    }
}

impl DensityField for SparseVolume {
    fn bounds(&self) -> Bounds {
        self.world_bounds()
    }
    fn density_world(&self, point: DVec3) -> f64 {
        self.density_world(point)
    }
    fn max_density(&self) -> f64 {
        self.maximum_density()
    }
    fn majorant_spans(
        &self,
        ray: Ray,
        endpoint: f64,
    ) -> Result<
        Box<dyn Iterator<Item = Result<crate::majorant::MajorantSpan, TraceError>> + '_>,
        TraceError,
    > {
        Ok(Box::new(SparseVolume::majorant_spans(self, ray, endpoint)?))
    }
}

/// Hash slots contain signed brick coordinates reinterpreted as u32 and a
/// float address into `values`. `u32::MAX` marks an empty slot. Leaf data use
/// x*64 + y*8 + z ordering. Tiles contain origin xyz, size, density bits, padding.
pub struct SparseGpuVolume {
    pub hash: Vec<[u32; 4]>,
    pub values: Vec<f32>,
    pub tiles: Vec<[u32; 8]>,
    /// Conservative density maximum per 8^3 index cell, including a positive
    /// halo for at-most-one-voxel f32 position error. Same hash layout as `hash`,
    /// but w stores f32 bits rather than a density buffer address.
    pub majorant_hash: Vec<[u32; 4]>,
    pub transform: UniformTransform,
    pub index_bounds: Bounds,
    pub majorant: f32,
}

fn contribute_majorant(
    cells: &mut HashMap<[i32; 3], f32>,
    brick: [i32; 3],
    maximum: f32,
) -> Result<(), String> {
    if maximum == 0.0 {
        return Ok(());
    }
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                let offset = [x, y, z];
                let mut cell = [0; 3];
                for axis in 0..3 {
                    cell[axis] = brick[axis]
                        .checked_sub(offset[axis])
                        .ok_or("majorant cell coordinate overflow")?;
                }
                let entry = cells.entry(cell).or_insert(0.0);
                *entry = entry.max(maximum);
            }
        }
    }
    Ok(())
}

pub fn brick_hash(coord: [i32; 3]) -> u32 {
    (coord[0] as u32).wrapping_mul(73_856_093)
        ^ (coord[1] as u32).wrapping_mul(19_349_663)
        ^ (coord[2] as u32).wrapping_mul(83_492_791)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_indices_tiles_and_linear_support_are_preserved() {
        let mut bricks = HashMap::new();
        let mut brick = Box::new([0.0; BRICK_VOXELS]);
        brick[7 * 64 + 7 * 8 + 7] = 2.0;
        bricks.insert([-1, -1, -1], brick);
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 2.0,
                translation: DVec3::ONE,
            },
            VolumeStats::default(),
            bricks,
            vec![
                UniformTile {
                    origin: [8, 0, 0],
                    size: 8,
                    density: 4.0,
                },
                UniformTile {
                    origin: [128, 0, 0],
                    size: 128,
                    density: 4.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(volume.sample_at_index([-1, -1, -1]), 2.0);
        assert_eq!(volume.trilinear(DVec3::splat(-0.5)), 0.25);
        assert_eq!(volume.sample_at_index([8, 0, 0]), 4.0);
        assert_eq!(volume.sample_at_index([16, 0, 0]), 0.0);
        assert!(volume.maximum_density() > 4.0);
        assert_eq!(
            volume.density_world(volume.index_to_world(DVec3::new(8.0, 0.0, 0.0))),
            4.0
        );
        let packed = volume.pack_gpu().unwrap();
        let entry = packed
            .hash
            .iter()
            .find(|v| v[..3] == [u32::MAX; 3])
            .unwrap();
        assert_eq!(&entry[..3], &[u32::MAX; 3]);
        assert_eq!(packed.values[511], 2.0);
        assert_eq!(packed.tiles[0][4], 4.0f32.to_bits());
    }

    #[test]
    fn local_majorants_cover_trilinear_halos_and_signed_coordinates() {
        let mut bricks = HashMap::new();
        let mut values = Box::new([0.0; BRICK_VOXELS]);
        values[0] = 3.0;
        values[511] = 2.0;
        bricks.insert([-1, 0, 1], values);
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 1.75,
                translation: DVec3::splat(2.5),
            },
            VolumeStats::default(),
            bricks,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(volume.majorant_cell_count(), 8);
        for x in -3..=1 {
            for y in -2..=1 {
                for z in -1..=3 {
                    let cell = [x, y, z];
                    let bound = volume.majorant_at_index_cell(cell);
                    let expected_nonzero =
                        (-2..=-1).contains(&x) && (-1..=0).contains(&y) && (0..=1).contains(&z);
                    assert_eq!(bound > 0.0, expected_nonzero);
                    // Include near-boundary coordinates and exact cell planes.
                    for local in [0.0, 0.001, 0.5, 7.5, 7.999_999, 8.0] {
                        let point = DVec3::from_array(cell.map(|v| f64::from(v) * 8.0))
                            + DVec3::splat(local);
                        assert!(volume.trilinear(point) <= bound);
                    }
                }
            }
        }
        let packed = volume.pack_gpu().unwrap();
        assert_eq!(
            packed
                .majorant_hash
                .iter()
                .filter(|v| v[3] != u32::MAX)
                .count(),
            27
        );
        assert!(
            packed
                .majorant_hash
                .iter()
                .filter(|v| v[3] != u32::MAX)
                .all(|v| f32::from_bits(v[3]) > 0.0 && f32::from_bits(v[3]) <= packed.majorant)
        );
    }

    #[test]
    fn large_tiles_keep_compact_conservative_boundary_majorants() {
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 1.0,
                translation: DVec3::ZERO,
            },
            VolumeStats::default(),
            HashMap::new(),
            vec![
                UniformTile {
                    origin: [-128, 0, 128],
                    size: 128,
                    density: 0.75,
                },
                UniformTile {
                    origin: [4096, 0, 0],
                    size: 4096,
                    density: 2.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(volume.majorant_cell_count(), 0);
        let packed = volume.pack_gpu().unwrap();
        assert_eq!(packed.majorant_hash.len(), 1);
        assert_eq!(packed.values.len(), 0);
        assert_eq!(packed.tiles.len(), 2);
        for point in [
            DVec3::new(-128.5, 0.25, 128.5),
            DVec3::new(-0.5, 127.5, 255.5),
            DVec3::new(4095.5, -0.5, -0.5),
            DVec3::new(8191.5, 4095.5, 4095.5),
        ] {
            let cell = (point / 8.0).floor().as_ivec3().to_array();
            assert!(volume.trilinear(point) > 0.0);
            assert!(volume.trilinear(point) <= volume.majorant_at_index_cell(cell));
        }
        assert_eq!(volume.majorant_at_index_cell([-18, 0, 16]), 0.0);
        assert_eq!(volume.majorant_at_index_cell([1024, 0, 0]), 0.0);
    }

    #[test]
    fn dda_spans_partition_empty_space_and_bound_every_sample() {
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 2.0,
                translation: DVec3::ONE,
            },
            VolumeStats::default(),
            HashMap::new(),
            vec![
                UniformTile {
                    origin: [0, 0, 0],
                    size: 8,
                    density: 1.0,
                },
                UniformTile {
                    origin: [32, 0, 0],
                    size: 8,
                    density: 0.5,
                },
            ],
        )
        .unwrap();
        for (origin, direction) in [
            (DVec3::new(-8.0, 2.0, 2.0), DVec3::X),
            (DVec3::new(48.0, 2.0, 2.0), -DVec3::X),
            (DVec3::new(8.0, 2.0, 2.0), -DVec3::X),
            (DVec3::new(8.0, 8.0, 8.0), -DVec3::ONE.normalize()),
            (DVec3::new(-8.0, -8.0, -8.0), DVec3::ONE.normalize()),
        ] {
            let ray = Ray {
                origin: volume.index_to_world(origin),
                direction,
            };
            let (enter, exit) = volume.world_bounds().ray_interval(ray).unwrap();
            let spans = volume
                .majorant_spans(ray, f64::INFINITY)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(spans[0].start, enter);
            assert_eq!(spans.last().unwrap().end, exit);
            for window in spans.windows(2) {
                assert_eq!(window[0].end, window[1].start);
            }
            for span in &spans {
                assert!(span.end > span.start);
                for fraction in [0.001, 0.25, 0.5, 0.75, 0.999] {
                    let point = ray.at(span.start + (span.end - span.start) * fraction);
                    let density = volume.density_world(point);
                    assert!(
                        density <= span.max_density,
                        "{point:?} density {density} exceeds {span:?}"
                    );
                    if span.max_density == 0.0 {
                        assert_eq!(density, 0.0);
                    }
                }
            }
            if direction == DVec3::X {
                assert!(spans.iter().any(|span| span.max_density == 0.0));
            }
        }
        let ray = Ray {
            origin: volume.index_to_world(DVec3::new(-8.0, 2.0, 2.0)),
            direction: DVec3::X,
        };
        let spans = volume
            .majorant_spans(ray, 30.0)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(spans.last().unwrap().end, 30.0);
        assert!(volume.majorant_spans(ray, 0.0).unwrap().next().is_none());
    }

    #[test]
    fn cached_trilinear_matches_reference_bits_for_leaves_tiles_and_boundaries() {
        let mut rng = crate::sampling::Pcg32::new(1984, 19);
        let mut bricks = HashMap::new();
        for coord in [[-2, -1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 1]] {
            let mut values = Box::new([0.0; BRICK_VOXELS]);
            for (index, value) in values.iter_mut().enumerate() {
                *value = if index % 7 == 0 {
                    0.0
                } else {
                    (rng.open01() * 1.5) as f32
                };
            }
            bricks.insert(coord, values);
        }
        let volume = SparseVolume::new(
            UniformTransform {
                scale: 1.666_666_626_930_236_8,
                translation: DVec3::ONE,
            },
            VolumeStats::default(),
            bricks,
            vec![
                UniformTile {
                    origin: [-8, 8, 0],
                    size: 8,
                    density: 0.5,
                },
                UniformTile {
                    origin: [128, 128, 128],
                    size: 128,
                    density: 0.75,
                },
                UniformTile {
                    origin: [-4096, -4096, -4096],
                    size: 4096,
                    density: 0.1,
                },
            ],
        )
        .unwrap();
        for origin in [
            DVec3::new(-16.0, -8.0, 0.0),
            DVec3::ZERO,
            DVec3::new(8.0, 0.0, 0.0),
            DVec3::new(-8.0, 8.0, 0.0),
            DVec3::splat(128.0),
            DVec3::splat(-4096.0),
        ] {
            for _ in 0..1024 {
                let point = origin + DVec3::new(rng.open01(), rng.open01(), rng.open01()) * 9.0
                    - DVec3::ONE;
                assert_eq!(
                    volume.trilinear(point).to_bits(),
                    volume.trilinear_reference(point).to_bits(),
                    "{point:?}"
                );
            }
            for offset in [-1.0, -0.001, 0.0, 0.5, 6.5, 7.0, 7.999_999, 8.0, 8.001] {
                let point = origin + DVec3::splat(offset);
                assert_eq!(
                    volume.trilinear(point).to_bits(),
                    volume.trilinear_reference(point).to_bits(),
                    "{point:?}"
                );
            }
        }
    }

    #[test]
    fn construction_rejects_duplicate_tiles_unsafe_coordinates_and_world_overflow() {
        let transform = UniformTransform {
            scale: 1.0,
            translation: DVec3::ZERO,
        };
        let tile = UniformTile {
            origin: [0, 0, 0],
            size: 8,
            density: 1.0,
        };
        assert!(
            SparseVolume::new(
                transform,
                VolumeStats::default(),
                HashMap::new(),
                vec![tile, tile]
            )
            .is_err()
        );
        let mut bricks = HashMap::new();
        bricks.insert([i32::MAX / 8, 0, 0], Box::new([1.0; BRICK_VOXELS]));
        assert!(SparseVolume::new(transform, VolumeStats::default(), bricks, Vec::new()).is_err());
        assert!(
            SparseVolume::new(
                UniformTransform {
                    scale: 1.0e308,
                    translation: DVec3::ZERO
                },
                VolumeStats::default(),
                HashMap::new(),
                vec![tile]
            )
            .is_err()
        );
        let volume = SparseVolume::new(
            transform,
            VolumeStats::default(),
            HashMap::new(),
            vec![
                UniformTile {
                    origin: [0, 0, 0],
                    size: 4096,
                    density: 0.5,
                },
                UniformTile {
                    origin: [0, 0, 0],
                    size: 128,
                    density: 1.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(volume.sample_at_index([16, 16, 16]), 1.0);
        let packed = volume.pack_gpu().unwrap();
        assert_eq!(packed.tiles[0][3], 128);
        assert_eq!(packed.tiles[1][3], 4096);
    }
}
