//! Just enough of the .npy format to read a 2-D little-endian float64 array:
//! format versions 1-3, C or Fortran order. The result is always column-major,
//! as BLAS and [`fb_gemm::Matrix`] are.

use std::fs;
use std::path::Path;

/// `(shape, values)` of the 2-D `<f8` array in `path`, values column-major.
pub fn read_f64_matrix(path: &Path) -> Result<([usize; 2], Vec<f64>), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |why: &str| format!("{}: {why}", path.display());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    // Version 1 has a 2-byte header length, 2 and 3 a 4-byte one.
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(bad(&format!("unsupported .npy version {v}"))),
    };
    let header = bytes
        .get(start..start + header_len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| bad("truncated header"))?;
    if !header.contains("'descr': '<f8'") {
        return Err(bad("not little-endian float64"));
    }
    let fortran = header.contains("'fortran_order': True");
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| bad("no shape"))?;
    let dims: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().map_err(|_| bad("unreadable shape")))
        .collect::<Result<_, _>>()?;
    let [rows, cols] = dims[..] else {
        return Err(bad(&format!("{}-D, not 2-D", dims.len())));
    };
    let body = &bytes[start + header_len..];
    if body.len() != rows * cols * 8 {
        return Err(bad("data length does not match the shape"));
    }
    let flat: Vec<f64> = body
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let values = if fortran {
        flat
    } else {
        // Row-major on disk: transpose into column-major.
        let mut col = vec![0.0; flat.len()];
        for i in 0..rows {
            for j in 0..cols {
                col[i + j * rows] = flat[i * cols + j];
            }
        }
        col
    };
    Ok(([rows, cols], values))
}
