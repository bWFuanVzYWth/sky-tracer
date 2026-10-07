//! Bounded, named tensor containers. Axes are stored in C order with the
//! existing LUT's fastest (phase or view) axis last in the safetensors shape.
use crate::{Result, asset::Hash};
use safetensors::tensor::{Dtype, Metadata, TensorInfo, TensorView, serialize_to_file};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// Includes the eight-byte header length. Counted for every file before bake.
pub(crate) const HEADER_RESERVE: u64 = 4096;
pub(crate) type TensorSpec<'a> = (&'a str, Dtype, Vec<usize>);
pub(crate) type TensorData<'a> = (&'a str, Dtype, Vec<usize>, &'a [u8]);

fn metadata(specs: &[TensorSpec<'_>]) -> Result<Metadata> {
    let mut sorted: Vec<_> = specs.iter().collect();
    // Match the library's alignment-first, then name ordering.
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut offset = 0usize;
    let mut tensors = Vec::with_capacity(sorted.len());
    for (name, dtype, shape) in sorted {
        if tensors.iter().any(|(n, _)| n == name) {
            return Err("duplicate LUT tensor name".into());
        }
        let bits = shape.iter().try_fold(dtype.bitsize(), |n, &d| {
            n.checked_mul(d).ok_or("LUT tensor size overflow")
        })?;
        if !bits.is_multiple_of(8) {
            return Err("unaligned LUT tensor size".into());
        }
        let end = offset
            .checked_add(bits / 8)
            .ok_or("LUT tensor size overflow")?;
        tensors.push((
            name.to_string(),
            TensorInfo {
                dtype: *dtype,
                shape: shape.clone(),
                data_offsets: (offset, end),
            },
        ));
        offset = end;
    }
    Ok(Metadata::new(None, tensors)?)
}

/// Allows large auxiliary tables to be appended one wavelength at a time.
pub(crate) fn write_header(writer: &mut impl Write, specs: &[TensorSpec<'_>]) -> Result<u64> {
    let m = metadata(specs)?;
    let mut header = serde_json::to_vec(&m)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    if 8 + header.len() as u64 > HEADER_RESERVE {
        return Err("LUT tensor header exceeds reserved budget".into());
    }
    writer.write_all(&(header.len() as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    Ok(m.data_len() as u64)
}

pub(crate) fn checksum_file(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut hash = Hash::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finish())
}

pub(crate) fn write(path: &Path, tensors: &[TensorData<'_>]) -> Result<String> {
    let specs: Vec<_> = tensors
        .iter()
        .map(|(n, d, s, _)| (*n, *d, s.clone()))
        .collect();
    // Preflight the same header and its bound without allocating the payload.
    write_header(&mut std::io::sink(), &specs)?;
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            Ok((*name, TensorView::new(*dtype, shape.clone(), data)?))
        })
        .collect::<Result<Vec<_>>>()?;
    // safetensors writes a sibling temporary and atomically replaces path.
    serialize_to_file(views, None, path)?;
    OpenOptions::new().write(true).open(path)?.sync_all()?;
    checksum_file(path)
}

pub(crate) fn open_validated(
    path: &Path,
    specs: &[TensorSpec<'_>],
    checksum: &str,
) -> Result<(BufReader<File>, Metadata, u64)> {
    let expected = metadata(specs)?;
    let mut reader = BufReader::new(File::open(path)?);
    let size = reader.get_ref().metadata()?.len();
    if size > expected.data_len() as u64 + HEADER_RESERVE {
        return Err("LUT tensor file exceeds expected size".into());
    }
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > HEADER_RESERVE - 8 {
        return Err("LUT tensor header exceeds reserved budget".into());
    }
    let mut header = vec![0u8; header_len as usize];
    reader.read_exact(&mut header)?;
    // The library validates sizes, contiguous offsets and arithmetic overflow.
    let stored: Metadata = serde_json::from_slice(&header)?;
    if stored.tensors().len() != specs.len()
        || stored.data_len() != expected.data_len()
        || size != 8 + header_len + stored.data_len() as u64
    {
        return Err("LUT tensor file size or tensor count mismatch".into());
    }
    for (name, dtype, shape) in specs {
        let info = stored.info(name).ok_or("missing LUT tensor")?;
        if info.dtype != *dtype || info.shape != *shape {
            return Err(format!("LUT tensor {name} dtype or shape mismatch").into());
        }
    }
    if checksum_file(path)? != checksum {
        return Err("LUT tensor checksum mismatch".into());
    }
    Ok((reader, stored, 8 + header_len))
}

pub(crate) fn read(path: &Path, specs: &[TensorSpec<'_>], checksum: &str) -> Result<Vec<Vec<u8>>> {
    let (mut reader, stored, payload_start) = open_validated(path, specs, checksum)?;
    let mut tensors = Vec::with_capacity(specs.len());
    for (name, _, _) in specs {
        let info = stored.info(name).ok_or("missing LUT tensor")?;
        reader.seek(SeekFrom::Start(payload_start + info.data_offsets.0 as u64))?;
        let mut bytes = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        reader.read_exact(&mut bytes)?;
        tensors.push(bytes);
    }
    Ok(tensors)
}

pub(crate) fn f32_values(bytes: &[u8], nonnegative: bool) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err("unaligned f32 LUT tensor".into());
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            let value = f32::from_le_bytes(*b);
            if !value.is_finite() || (nonnegative && value < 0.0) {
                Err("invalid stored LUT value".into())
            } else {
                Ok(value)
            }
        })
        .collect()
}
