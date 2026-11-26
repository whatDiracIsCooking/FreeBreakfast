//! The two `.npy` shapes this bench moves: a 1-D little-endian float64
//! array, read (the generators' packed lower triangles) and written (the
//! eigenvalues). Kept here rather than taken from `rust/io/npy` so each
//! image builds this crate alone, with no `rust/` workspace in its context.

use std::fs;
use std::path::Path;

/// The float64 values of the 1-D C-order `<f8` array at `path`.
pub fn read_1d(path: &Path) -> Result<Vec<f64>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |what: &str| format!("{}: {what}", path.display());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            (len as usize, 12)
        }
        v => return Err(bad(&format!("unknown .npy version {v}"))),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])
        .map_err(|_| bad("header is not UTF-8"))?;
    if !header.contains("'descr': '<f8'") || !header.contains("'fortran_order': False") {
        return Err(bad(&format!("want a C-order <f8 array, header {header}")));
    }
    let shape = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| bad("no shape"))?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| bad("bad shape")))
        .collect::<Result<_, _>>()?;
    let [len] = dims[..] else {
        return Err(bad(&format!("want 1-D, shape ({shape})")));
    };
    let data = &bytes[start + header_len..];
    if data.len() != 8 * len {
        return Err(bad("data length does not match the shape"));
    }
    Ok(data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect())
}

/// Writes `values` as a 1-D `<f8` array, version 1.0.
pub fn write_1d(path: &Path, values: &[f64]) -> Result<(), String> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    // Magic (6) + version (2) + length (2) + header + '\n', padded to 64.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 8 * values.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}
