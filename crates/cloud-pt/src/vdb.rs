//! Reader for the scalar FloatTree (5,4,3) OpenVDB format used by the Disney
//! cloud dataset. This is deliberately not a general OpenVDB implementation.
//!
//! Supported: file versions 222--224, native/half float values, zero background,
//! uniform scale with optional translation, raw, ZIP and Blosc/LZ4 compression,
//! and active-mask compression. Root/internal tiles and inactive leaf values
//! are retained. Unsupported versions, instancing and maps fail explicitly.
//! Layout references: OpenVDB io/Archive, io/Compression.h and tree/*Node.h.

use crate::volume::{BRICK_VOXELS, SparseVolume, UniformTile, UniformTransform, VolumeStats};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub type VdbError = Box<dyn Error + Send + Sync>;
pub type VdbResult<T> = Result<T, VdbError>;

const ZIP: u32 = 1;
const ACTIVE_MASK: u32 = 2;
const BLOSC: u32 = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetadataValue {
    pub type_name: String,
    pub bytes: Vec<u8>,
}

impl MetadataValue {
    fn vec3i(&self) -> Option<[i32; 3]> {
        if self.type_name != "vec3i" || self.bytes.len() != 12 {
            return None;
        }
        Some(std::array::from_fn(|i| {
            i32::from_le_bytes(self.bytes[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridInfo {
    pub name: String,
    pub grid_type: String,
    /// The canonical descriptor `_HalfFloat` suffix controls buffer decoding.
    /// Informational metadata never overrides the actual serialized type.
    pub saved_as_half: bool,
    pub file_version: u32,
    pub compression: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub transform: UniformTransform,
    pub instance_parent: String,
    pub grid_pos: u64,
    pub block_pos: u64,
    pub end_pos: u64,
    pub topology_pos: u64,
}

/// Read only the file header, metadata and transforms; no density blocks are
/// loaded and no compute backend is touched.
pub fn inspect_vdb(path: &Path) -> VdbResult<Vec<GridInfo>> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    read_descriptors(&mut BufReader::new(file), size)
}

pub fn load_vdb(path: &Path, grid_name: &str) -> VdbResult<SparseVolume> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let grids = read_descriptors(&mut reader, size)?;
    let info = grids
        .into_iter()
        .find(|grid| grid.name == grid_name)
        .ok_or_else(|| {
            format!(
                "VDB grid {grid_name:?} does not exist in {}",
                path.display()
            )
        })?;
    if info.grid_type != "Tree_float_5_4_3" {
        return Err(format!(
            "unsupported VDB grid type {:?}; expected Tree_float_5_4_3",
            info.grid_type
        )
        .into());
    }
    if !info.instance_parent.is_empty() {
        return Err("instanced VDB trees are not supported".into());
    }
    let saved_as_half = info.saved_as_half;
    let mut stats = VolumeStats {
        grid_name: info.name.clone(),
        file_version: info.file_version,
        compression: info.compression,
        saved_as_half,
        metadata_bbox_min: info
            .metadata
            .get("file_bbox_min")
            .and_then(MetadataValue::vec3i),
        metadata_bbox_max: info
            .metadata
            .get("file_bbox_max")
            .and_then(MetadataValue::vec3i),
        ..VolumeStats::default()
    };
    reader.seek(SeekFrom::Start(info.topology_pos))?;
    let buffer_count = read_u32(&mut reader)?;
    if buffer_count != 1 {
        return Err("multi-buffer VDB trees are not supported".into());
    }
    let background = read_f32(&mut reader)?;
    stats.background = background;
    if background != 0.0 || !background.is_finite() {
        return Err("a nonzero or invalid VDB background cannot describe a finite cloud".into());
    }
    let tile_count = read_u32(&mut reader)?;
    let child_count = read_u32(&mut reader)?;
    if tile_count as u64 > size / 17 || child_count as u64 > size / 16 {
        return Err("invalid VDB root node counts".into());
    }
    let mut tiles = Vec::new();
    let mut root_coordinates = HashSet::new();
    for _ in 0..tile_count {
        let origin = read_coord(&mut reader)?;
        claim_root_coordinate(&mut root_coordinates, origin)?;
        let density = read_f32(&mut reader)?;
        let _active = read_u8(&mut reader)?;
        add_tile(&mut tiles, origin, 4096, density)?;
    }
    let mut leaves = Vec::new();
    for _ in 0..child_count {
        let origin = read_coord(&mut reader)?;
        claim_root_coordinate(&mut root_coordinates, origin)?;
        read_internal(
            &mut reader,
            &info,
            saved_as_half,
            background,
            origin,
            5,
            7,
            &mut tiles,
            &mut leaves,
        )?;
    }
    if reader.stream_position()? != info.block_pos {
        return Err(format!(
            "VDB topology ended at {}, expected block offset {}; unsupported topology layout",
            reader.stream_position()?,
            info.block_pos
        )
        .into());
    }
    stats.leaf_count = leaves.len();
    let mut leaf_coordinates = HashSet::with_capacity(leaves.len());
    for leaf in &leaves {
        if leaf.iter().any(|value| value.rem_euclid(8) != 0) {
            return Err("unaligned VDB leaf origin".into());
        }
        if !leaf_coordinates.insert(*leaf) {
            return Err("duplicate VDB leaf coordinate".into());
        }
    }
    let mut bricks = HashMap::with_capacity(leaves.len());
    for leaf in leaves {
        let mask = read_mask(&mut reader, BRICK_VOXELS)?;
        let values = read_compressed_values(
            &mut reader,
            &info,
            saved_as_half,
            background,
            BRICK_VOXELS,
            &mask,
        )?;
        let mut nonzero = false;
        stats.active_voxel_count += mask.iter().map(|v| v.count_ones() as u64).sum::<u64>();
        for (index, value) in values.iter().enumerate() {
            check_density(*value)?;
            if *value != 0.0 {
                nonzero = true;
                stats.nonzero_voxel_count += 1;
                if !mask_on(&mask, index) {
                    stats.inactive_nonzero_voxel_count += 1;
                }
            }
        }
        if nonzero {
            let brick_coord = leaf.map(|v| v.div_euclid(8));
            let boxed: Box<[f32; BRICK_VOXELS]> = values
                .into_boxed_slice()
                .try_into()
                .map_err(|_| "invalid VDB leaf buffer length")?;
            if bricks.insert(brick_coord, boxed).is_some() {
                return Err("duplicate VDB leaf coordinate".into());
            }
        }
    }
    if reader.stream_position()? != info.end_pos {
        return Err(format!(
            "VDB grid ended at {}, expected {}; unsupported multi-pass or trailing data",
            reader.stream_position()?,
            info.end_pos
        )
        .into());
    }
    SparseVolume::new(info.transform, stats, bricks, tiles).map_err(Into::into)
}

fn read_descriptors<R: Read + Seek>(reader: &mut R, file_size: u64) -> VdbResult<Vec<GridInfo>> {
    if read_u64(reader)? != 0x5644_4220 {
        return Err("invalid OpenVDB magic".into());
    }
    let version = read_u32(reader)?;
    if !(222..=224).contains(&version) {
        return Err(format!(
            "unsupported VDB file version {version}; this Disney reader supports 222--224"
        )
        .into());
    }
    let _major = read_u32(reader)?;
    let _minor = read_u32(reader)?;
    if read_u8(reader)? != 1 {
        return Err("VDB file requires grid offsets".into());
    }
    let mut guid = [0; 36];
    reader.read_exact(&mut guid)?;
    let _metadata = read_metadata(reader)?;
    let count = read_u32(reader)?;
    if count > 65_536 {
        return Err("unreasonable VDB grid count".into());
    }
    let mut result = Vec::new();
    for _ in 0..count {
        let name = read_string(reader)?;
        let encoded_grid_type = read_string(reader)?;
        // OpenVDB GridDescriptor flags half buffers with this type suffix,
        // independently of any saved metadata about the grid's precision.
        let saved_as_half = encoded_grid_type.ends_with("_HalfFloat");
        let grid_type = encoded_grid_type
            .strip_suffix("_HalfFloat")
            .unwrap_or(&encoded_grid_type)
            .to_owned();
        let instance_parent = read_string(reader)?;
        let grid_pos = read_u64(reader)?;
        let block_pos = read_u64(reader)?;
        let end_pos = read_u64(reader)?;
        if !(grid_pos <= block_pos && block_pos <= end_pos && end_pos <= file_size) {
            return Err("invalid VDB grid offsets".into());
        }
        reader.seek(SeekFrom::Start(grid_pos))?;
        let compression = read_u32(reader)?;
        // OpenVDB itself gives Blosc precedence when both codec flags are set.
        // This narrow reader rejects that combination instead of ambiguously
        // accepting a ZIP stream under an advertised Blosc flag.
        if compression & !7 != 0 || compression & ZIP != 0 && compression & BLOSC != 0 {
            return Err(format!("unsupported VDB compression flags {compression}").into());
        }
        let metadata = read_metadata(reader)?;
        let transform = read_transform(reader)?;
        let topology_pos = reader.stream_position()?;
        result.push(GridInfo {
            name,
            grid_type,
            saved_as_half,
            file_version: version,
            compression,
            metadata,
            transform,
            instance_parent,
            grid_pos,
            block_pos,
            end_pos,
            topology_pos,
        });
        reader.seek(SeekFrom::Start(end_pos))?;
    }
    Ok(result)
}

fn read_transform<R: Read>(reader: &mut R) -> VdbResult<UniformTransform> {
    let map = read_string(reader)?;
    let translation = match map.as_str() {
        "UniformScaleMap" => DVec3::ZERO,
        "UniformScaleTranslateMap" | "ScaleTranslateMap" => read_dvec3(reader)?,
        _ => {
            return Err(format!(
                "unsupported VDB transform {map:?}; expected a uniform scale/translation map"
            )
            .into());
        }
    };
    let scale_values = read_dvec3(reader)?;
    for _ in 0..4 {
        let _ = read_dvec3(reader)?;
    }
    if !scale_values.is_finite()
        || scale_values.min_element() <= 0.0
        || (scale_values - DVec3::splat(scale_values.x))
            .abs()
            .max_element()
            > scale_values.x.abs() * 1.0e-12
    {
        return Err("anisotropic, reflected or invalid VDB scale is not supported".into());
    }
    Ok(UniformTransform {
        scale: scale_values.x,
        translation,
    })
}

#[allow(clippy::too_many_arguments)]
fn read_internal<R: Read>(
    reader: &mut R,
    info: &GridInfo,
    half: bool,
    background: f32,
    origin: [i32; 3],
    log_dim: usize,
    child_log_size: usize,
    tiles: &mut Vec<UniformTile>,
    leaves: &mut Vec<[i32; 3]>,
) -> VdbResult<()> {
    let count = 1usize << (3 * log_dim);
    let children = read_mask(reader, count)?;
    let values_mask = read_mask(reader, count)?;
    if children
        .iter()
        .zip(&values_mask)
        .any(|(child, value)| child & value != 0)
    {
        return Err("VDB node slot cannot be both a child and an active scalar tile".into());
    }
    let values = read_compressed_values(reader, info, half, background, count, &values_mask)?;
    for (index, value) in values.iter().enumerate() {
        if !mask_on(&children, index) {
            let coord = slot_origin(origin, index, log_dim, child_log_size)?;
            add_tile(tiles, coord, 1 << child_log_size, *value)?;
        }
    }
    for index in 0..count {
        if !mask_on(&children, index) {
            continue;
        }
        let coord = slot_origin(origin, index, log_dim, child_log_size)?;
        if log_dim == 5 {
            read_internal(reader, info, half, background, coord, 4, 3, tiles, leaves)?;
        } else {
            let _ = read_mask(reader, BRICK_VOXELS)?;
            leaves.push(coord);
        }
    }
    Ok(())
}

fn slot_origin(
    origin: [i32; 3],
    index: usize,
    log_dim: usize,
    shift: usize,
) -> VdbResult<[i32; 3]> {
    let mask = (1 << log_dim) - 1;
    let xyz = [
        index >> (2 * log_dim),
        (index >> log_dim) & mask,
        index & mask,
    ];
    let mut coord = [0; 3];
    for axis in 0..3 {
        coord[axis] = origin[axis]
            .checked_add((xyz[axis] << shift) as i32)
            .ok_or("VDB coordinate overflow")?;
    }
    Ok(coord)
}

fn add_tile(
    tiles: &mut Vec<UniformTile>,
    origin: [i32; 3],
    size: i32,
    density: f32,
) -> VdbResult<()> {
    if ![8, 128, 4096].contains(&size) || origin.iter().any(|value| value.rem_euclid(size) != 0) {
        return Err("unaligned or unsupported VDB tile origin/size".into());
    }
    check_density(density)?;
    if density != 0.0 {
        tiles.push(UniformTile {
            origin,
            size,
            density,
        });
    }
    Ok(())
}

fn claim_root_coordinate(seen: &mut HashSet<[i32; 3]>, origin: [i32; 3]) -> VdbResult<()> {
    if origin.iter().any(|value| value.rem_euclid(4096) != 0) {
        return Err("unaligned VDB root child/tile coordinate".into());
    }
    if !seen.insert(origin) {
        return Err("duplicate or overlapping VDB root child/tile coordinate".into());
    }
    Ok(())
}

fn check_density(value: f32) -> VdbResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err("VDB density contains a negative or non-finite value".into());
    }
    Ok(())
}

fn read_compressed_values<R: Read>(
    reader: &mut R,
    info: &GridInfo,
    half: bool,
    background: f32,
    count: usize,
    mask: &[u64],
) -> VdbResult<Vec<f32>> {
    let metadata = read_u8(reader)?;
    if metadata > 6 {
        return Err(format!("invalid VDB node compression metadata {metadata}").into());
    }
    // Inactive constants remain native float even when saved values use half.
    let mut inactive0 = if metadata == 0 {
        background
    } else {
        -background
    };
    let mut inactive1 = background;
    if matches!(metadata, 2 | 4 | 5) {
        inactive0 = read_f32(reader)?;
    }
    if metadata == 5 {
        inactive1 = read_f32(reader)?;
    }
    let selection = if matches!(metadata, 3 | 4 | 5) {
        read_mask(reader, count)?
    } else {
        Vec::new()
    };
    let packed = info.compression & ACTIVE_MASK != 0 && metadata != 6;
    let saved_count = if packed {
        mask.iter().map(|v| v.count_ones() as usize).sum()
    } else {
        count
    };
    let stride = if half { 2 } else { 4 };
    let expected = saved_count * stride;
    // OpenVDB HalfReader/HalfWriter return immediately for zero values, before
    // the codec reads/writes its length header. Native f32 still calls readData
    // at count zero and therefore keeps the compressed block header.
    let bytes = if half && saved_count == 0 {
        Vec::new()
    } else {
        read_value_bytes(reader, info.compression, expected)?
    };
    let saved: Vec<f32> = if half {
        bytes
            .chunks_exact(2)
            .map(|v| half::f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    };
    if !packed || saved_count == count {
        return Ok(saved);
    }
    let mut result = Vec::with_capacity(count);
    let mut source = saved.into_iter();
    for index in 0..count {
        result.push(if mask_on(mask, index) {
            source.next().ok_or("truncated VDB active values")?
        } else if !selection.is_empty() && mask_on(&selection, index) {
            inactive1
        } else {
            inactive0
        });
    }
    Ok(result)
}

fn read_value_bytes<R: Read>(
    reader: &mut R,
    compression: u32,
    expected: usize,
) -> VdbResult<Vec<u8>> {
    if compression & (ZIP | BLOSC) == 0 {
        let mut bytes = vec![0; expected];
        reader.read_exact(&mut bytes)?;
        return Ok(bytes);
    }
    let byte_count = read_i64(reader)?;
    if byte_count <= 0 {
        if byte_count.checked_neg().map(|v| v as u64) != Some(expected as u64) {
            return Err("uncompressed VDB value size does not match its mask".into());
        }
        let mut bytes = vec![0; expected];
        reader.read_exact(&mut bytes)?;
        return Ok(bytes);
    }
    if byte_count > 16 * 1024 * 1024 {
        return Err("unreasonable compressed VDB node size".into());
    }
    let mut input = vec![0; byte_count as usize];
    reader.read_exact(&mut input)?;
    if compression & ZIP != 0 {
        let mut decoder = flate2::read::ZlibDecoder::new(input.as_slice());
        let mut bytes = Vec::with_capacity(expected);
        decoder
            .by_ref()
            .take(expected as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() != expected {
            return Err("ZIP VDB value size does not match its mask".into());
        }
        return Ok(bytes);
    }
    if input.len() < 16 {
        return Err("truncated Blosc header".into());
    }
    let nbytes = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let cbytes = u32::from_le_bytes(input[12..16].try_into().unwrap()) as usize;
    if nbytes != expected || cbytes != input.len() {
        return Err("Blosc VDB value size does not match its mask".into());
    }
    if expected == 0 {
        return Ok(Vec::new());
    }
    let mut output = vec![0; expected];
    // Validate before passing potentially damaged input to the native codec.
    let mut validated = 0;
    let valid = unsafe {
        blosc_src::blosc_cbuffer_validate(input.as_ptr().cast(), input.len(), &mut validated)
    };
    if valid != 0 || validated != expected {
        return Err("invalid Blosc buffer".into());
    }
    let decoded = unsafe {
        blosc_src::blosc_decompress_ctx(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            expected,
            1,
        )
    };
    if decoded < 0 || decoded as usize != expected {
        return Err("Blosc decompression failed (supported codecs: BloscLZ and LZ4)".into());
    }
    Ok(output)
}

fn read_metadata<R: Read>(reader: &mut R) -> VdbResult<HashMap<String, MetadataValue>> {
    let count = read_u32(reader)?;
    if count > 65_536 {
        return Err("unreasonable VDB metadata count".into());
    }
    let mut result = HashMap::new();
    for _ in 0..count {
        let name = read_string(reader)?;
        let type_name = read_string(reader)?;
        let size = read_u32(reader)? as usize;
        if size > 16 * 1024 * 1024 {
            return Err("unreasonable VDB metadata size".into());
        }
        let mut bytes = vec![0; size];
        reader.read_exact(&mut bytes)?;
        result.insert(name, MetadataValue { type_name, bytes });
    }
    Ok(result)
}

fn read_string<R: Read>(reader: &mut R) -> VdbResult<String> {
    let size = read_u32(reader)? as usize;
    if size > 1024 * 1024 {
        return Err("unreasonable VDB string size".into());
    }
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

fn read_mask<R: Read>(reader: &mut R, count: usize) -> VdbResult<Vec<u64>> {
    (0..count / 64).map(|_| read_u64(reader)).collect()
}
fn mask_on(mask: &[u64], index: usize) -> bool {
    mask[index / 64] & (1 << (index % 64)) != 0
}
fn read_coord<R: Read>(reader: &mut R) -> VdbResult<[i32; 3]> {
    Ok([read_i32(reader)?, read_i32(reader)?, read_i32(reader)?])
}
fn read_dvec3<R: Read>(reader: &mut R) -> VdbResult<DVec3> {
    Ok(DVec3::new(
        read_f64(reader)?,
        read_f64(reader)?,
        read_f64(reader)?,
    ))
}
macro_rules! scalar_reader {
    ($name:ident, $type:ty, $count:expr) => {
        fn $name<R: Read>(reader: &mut R) -> VdbResult<$type> {
            let mut bytes = [0; $count];
            reader.read_exact(&mut bytes)?;
            Ok(<$type>::from_le_bytes(bytes))
        }
    };
}
scalar_reader!(read_u8, u8, 1);
scalar_reader!(read_u32, u32, 4);
scalar_reader!(read_i32, i32, 4);
scalar_reader!(read_u64, u64, 8);
scalar_reader!(read_i64, i64, 8);
scalar_reader!(read_f32, f32, 4);
scalar_reader!(read_f64, f64, 8);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn info(compression: u32) -> GridInfo {
        GridInfo {
            name: "density".into(),
            grid_type: "Tree_float_5_4_3".into(),
            saved_as_half: false,
            file_version: 223,
            compression,
            metadata: HashMap::new(),
            transform: UniformTransform {
                scale: 1.0,
                translation: DVec3::ZERO,
            },
            instance_parent: String::new(),
            grid_pos: 0,
            block_pos: 0,
            end_pos: 0,
            topology_pos: 0,
        }
    }

    #[test]
    fn inactive_values_and_half_conversion_are_not_lost() {
        let mut input = vec![5]; // two inactive constants selected by a mask
        input.extend_from_slice(&2.0f32.to_le_bytes());
        input.extend_from_slice(&4.0f32.to_le_bytes());
        input.extend_from_slice(&2u64.to_le_bytes());
        input.extend_from_slice(&half::f16::from_f32(3.0).to_bits().to_le_bytes());
        let values = read_compressed_values(
            &mut Cursor::new(input),
            &info(ACTIVE_MASK),
            true,
            0.0,
            64,
            &[1],
        )
        .unwrap();
        assert_eq!(values[0], 3.0);
        assert_eq!(values[1], 4.0);
        assert!(values[2..].iter().all(|v| *v == 2.0));
    }

    #[test]
    fn all_inactive_half_nodes_do_not_consume_a_codec_header_or_the_next_node() {
        let sentinel = 0xA187_C23D_E459_F06Bu64;
        for codec in [ZIP, BLOSC] {
            let mut input = vec![2u8]; // one non-background inactive constant
            input.extend_from_slice(&0.5f32.to_le_bytes());
            input.extend_from_slice(&sentinel.to_le_bytes());
            let mut cursor = Cursor::new(input);
            let values = read_compressed_values(
                &mut cursor,
                &info(ACTIVE_MASK | codec),
                true,
                0.0,
                64,
                &[0],
            )
            .unwrap();
            assert_eq!(values, vec![0.5; 64]);
            assert_eq!(
                cursor.position(),
                5,
                "zero half values have no codec header"
            );
            assert_eq!(
                read_u64(&mut cursor).unwrap(),
                sentinel,
                "next node remains untouched"
            );
        }
    }

    #[test]
    fn descriptor_half_suffix_controls_decoding_independently_of_metadata() {
        fn string(bytes: &mut Vec<u8>, text: &str) {
            bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        for (grid_type, half, metadata_flag) in [
            ("Tree_float_5_4_3_HalfFloat", true, false),
            ("Tree_float_5_4_3", false, true),
        ] {
            let mut header = 0x5644_4220u64.to_le_bytes().to_vec();
            for value in [223u32, 3, 0] {
                header.extend_from_slice(&value.to_le_bytes());
            }
            header.push(1); // offsets present
            header.extend_from_slice(&[0u8; 36]);
            header.extend_from_slice(&0u32.to_le_bytes()); // file metadata count
            header.extend_from_slice(&1u32.to_le_bytes()); // grid count
            string(&mut header, "density");
            string(&mut header, grid_type);
            string(&mut header, ""); // instance parent
            let grid_pos = header.len() as u64 + 24;
            let mut grid = ACTIVE_MASK.to_le_bytes().to_vec();
            grid.extend_from_slice(&1u32.to_le_bytes()); // grid metadata count
            string(&mut grid, "is_saved_as_half_float");
            string(&mut grid, "bool");
            grid.extend_from_slice(&1u32.to_le_bytes());
            grid.push(u8::from(metadata_flag));
            string(&mut grid, "UniformScaleMap");
            for _ in 0..15 {
                grid.extend_from_slice(&1.0f64.to_le_bytes());
            }
            let end_pos = grid_pos + grid.len() as u64;
            for value in [grid_pos, end_pos, end_pos] {
                header.extend_from_slice(&value.to_le_bytes());
            }
            header.extend_from_slice(&grid);
            let size = header.len() as u64;
            let info = read_descriptors(&mut Cursor::new(header), size)
                .unwrap()
                .pop()
                .unwrap();
            assert_eq!(info.grid_type, "Tree_float_5_4_3");
            assert_eq!(info.saved_as_half, half);
            assert_eq!(
                info.metadata["is_saved_as_half_float"].bytes,
                [u8::from(metadata_flag)]
            );
        }
    }

    #[test]
    fn malformed_tree_origins_and_duplicate_root_slots_are_rejected() {
        let mut seen = HashSet::new();
        claim_root_coordinate(&mut seen, [-4096, 0, 4096]).unwrap();
        assert!(claim_root_coordinate(&mut seen, [-4096, 0, 4096]).is_err());
        assert!(claim_root_coordinate(&mut seen, [1, 0, 0]).is_err());
        let mut tiles = Vec::new();
        assert!(add_tile(&mut tiles, [1, 0, 0], 8, 0.0).is_err());
        assert!(add_tile(&mut tiles, [8, 0, 0], 128, 0.5).is_err());
        add_tile(&mut tiles, [-128, 0, 0], 128, 0.5).unwrap();
    }

    #[test]
    fn zip_and_blosc_blocks_decode_on_cpu() {
        use std::io::Write;
        let original = (0..512)
            .map(|i| (i as f32 / 512.0).to_le_bytes())
            .flatten()
            .collect::<Vec<_>>();
        let mut zip = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        zip.write_all(&original).unwrap();
        let zipped = zip.finish().unwrap();
        let mut bytes = (zipped.len() as i64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&zipped);
        assert_eq!(
            read_value_bytes(&mut Cursor::new(bytes), ZIP, original.len()).unwrap(),
            original
        );
        let mut compressed = vec![0u8; original.len() + 16];
        let count = unsafe {
            blosc_src::blosc_compress_ctx(
                5,
                1,
                4,
                original.len(),
                original.as_ptr().cast(),
                compressed.as_mut_ptr().cast(),
                compressed.len(),
                c"lz4".as_ptr(),
                0,
                1,
            )
        };
        assert!(count > 0);
        compressed.truncate(count as usize);
        let mut bytes = (compressed.len() as i64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&compressed);
        assert_eq!(
            read_value_bytes(&mut Cursor::new(bytes), BLOSC, original.len()).unwrap(),
            original
        );
    }

    // Asset validation is explicit to avoid unexpected disk/CPU work during
    // normal test runs; these tests never initialize a GPU.
    #[test]
    #[ignore = "CPU-only Disney dataset validation; requires the local asset"]
    fn disney_sixteenth_and_eighth_load_without_resampling() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/DisneyCloudDataset/wdas_cloud");
        for (filename, scale, min, max, leaves, nonzero, tile_count, leaf_sample, tile_sample) in [
            (
                "wdas_cloud_sixteenth.vdb",
                3.3333332538604736,
                [-66, -21, -90],
                [59, 64, 63],
                1241,
                296477,
                130,
                [43, 0, 27],
                [-32, 8, -8],
            ),
            (
                "wdas_cloud_eighth.vdb",
                1.6666666269302368,
                [-131, -41, -179],
                [118, 128, 127],
                6823,
                2049049,
                1672,
                [86, 0, 54],
                [-64, -8, -16],
            ),
        ] {
            let volume = load_vdb(&directory.join(filename), "density").unwrap();
            assert_eq!(volume.transform.scale, scale);
            assert_eq!(volume.stats.metadata_bbox_min, Some(min));
            assert_eq!(volume.stats.metadata_bbox_max, Some(max));
            assert_eq!(volume.stats.background, 0.0);
            assert_eq!(volume.stats.leaf_count, leaves);
            assert_eq!(volume.stats.nonzero_voxel_count, nonzero);
            assert_eq!(volume.stats.nonzero_tile_count, tile_count);
            assert_eq!(volume.stats.inactive_nonzero_voxel_count, 0);
            assert_eq!(volume.stats.maximum_density, 1.0);
            assert_eq!(volume.sample_at_index(leaf_sample), 1.0);
            assert_eq!(volume.sample_at_index(tile_sample), 0.5);
            let packed = volume.pack_gpu().unwrap();
            assert!(
                packed.tiles.is_empty(),
                "all nonzero Disney low-resolution tiles have size 8"
            );
            for sample in [leaf_sample, tile_sample] {
                let coordinate = sample.map(|v| v.div_euclid(8));
                let mut slot =
                    crate::volume::brick_hash(coordinate) as usize & (packed.hash.len() - 1);
                loop {
                    let entry = packed.hash[slot];
                    assert_ne!(entry[3], u32::MAX);
                    if entry[..3] == coordinate.map(|v| v as u32) {
                        let local = sample.map(|v| v.rem_euclid(8) as usize);
                        assert_eq!(
                            packed.values
                                [entry[3] as usize + local[0] * 64 + local[1] * 8 + local[2]],
                            volume.sample_at_index(sample)
                        );
                        break;
                    }
                    slot = (slot + 1) & (packed.hash.len() - 1);
                }
            }
            let camera_origin = DVec3::new(648.064, -82.473, -63.856);
            let mut proposal_length = 0.0;
            let mut global_length = 0.0;
            let mut zero_spans = 0;
            for index in 0..128 {
                let mut rng = crate::sampling::Pcg32::for_sample(81, 0, index);
                let target = DVec3::new(
                    (rng.open01() - 0.5) * 450.0,
                    (rng.open01() - 0.5) * 300.0 + 70.0,
                    (rng.open01() - 0.5) * 500.0 - 40.0,
                );
                let ray = crate::transport::Ray {
                    origin: camera_origin,
                    direction: (target - camera_origin).normalize(),
                };
                let spans = volume
                    .majorant_spans(ray, f64::INFINITY)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                for span in &spans {
                    assert!(span.end > span.start);
                    proposal_length += span.max_density * (span.end - span.start);
                    global_length += volume.maximum_density() * (span.end - span.start);
                    zero_spans += usize::from(span.max_density == 0.0);
                    for fraction in [0.001, 0.5, 0.999] {
                        let point = ray.at(span.start + (span.end - span.start) * fraction);
                        assert!(
                            volume.density_world(point) <= span.max_density,
                            "density at {point:?} exceeds local bound {span:?}"
                        );
                    }
                }
                for adjacent in spans.windows(2) {
                    assert_eq!(adjacent[0].end, adjacent[1].start);
                }
            }
            assert!(proposal_length < global_length);
            assert!(zero_spans > 0);
            eprintln!(
                "{filename}: {} majorant cells, {} hash bytes; camera ray integrated proposal/global {:.3}, zero spans {zero_spans}",
                volume.majorant_cell_count(),
                packed.majorant_hash.len() * 16,
                proposal_length / global_length
            );
            eprintln!(
                "{filename}: {:?}, bounds {:?}",
                volume.stats,
                volume.world_bounds()
            );
            // CPU evaluation of the shader's padded table on the same exact
            // physical ray domains. This measures expected proposal work, not
            // GPU execution time, multiple-scattering paths, or frame rate.
            let camera = crate::config::Camera::default();
            let render = crate::config::RenderConfig::default();
            let mut grid_cpu = 0.0;
            let mut grid_gpu = 0.0;
            let mut grid_global = 0.0;
            for y in 0..8 {
                for x in 0..16 {
                    let ray = camera
                        .ray(
                            render.width,
                            render.height,
                            (f64::from(x) + 0.5) * f64::from(render.width) / 16.0,
                            (f64::from(y) + 0.5) * f64::from(render.height) / 8.0,
                        )
                        .unwrap();
                    for span in volume.majorant_spans(ray, f64::INFINITY).unwrap() {
                        let span = span.unwrap();
                        let cell = (volume.world_to_index(ray.at((span.start + span.end) * 0.5))
                            / 8.0)
                            .floor()
                            .as_ivec3()
                            .to_array();
                        let mut slot = crate::volume::brick_hash(cell) as usize
                            & (packed.majorant_hash.len() - 1);
                        let gpu_maximum = loop {
                            let entry = packed.majorant_hash[slot];
                            if entry[3] == u32::MAX {
                                break 0.0;
                            }
                            if entry[..3] == cell.map(|v| v as u32) {
                                break f64::from(f32::from_bits(entry[3]));
                            }
                            slot = (slot + 1) & (packed.majorant_hash.len() - 1);
                        };
                        assert!(gpu_maximum >= span.max_density);
                        let length = span.end - span.start;
                        grid_cpu += span.max_density * length;
                        grid_gpu += gpu_maximum * length;
                        grid_global += volume.maximum_density() * length;
                    }
                }
            }
            eprintln!(
                "{filename}: 16x8 default camera same-domain integral/global CPU {:.9}, GPU padded {:.9}; GPU cells {}, hash {} bytes, total storage {:?} bytes; sigma4 expected candidates global {:.6}, CPU {:.6}, GPU {:.6}",
                grid_cpu / grid_global,
                grid_gpu / grid_global,
                volume.gpu_majorant_cell_count(),
                packed.majorant_hash.len() * 16,
                volume.gpu_storage_bytes(),
                grid_global * 4.0,
                grid_cpu * 4.0,
                grid_gpu * 4.0
            );
        }
    }
}
