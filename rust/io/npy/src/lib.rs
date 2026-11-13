//! `src/io/npy` (module `free_breakfast.io.npy`): NumPy's `.npy` binary
//! array format, both directions, for the slice the tree holds: float64
//! (`'<f8'`), column-major.
//!
//! The format
//! (<https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>):
//! the 6-byte magic `"\x93NUMPY"`, a 2-byte version, a little-endian header
//! length (2 bytes in version 1, 4 in versions 2 and 3), a Python-dict
//! header carrying `descr`, `fortran_order` and `shape`, then the raw
//! element bytes. The SAD stamps of `data/sad/` are such files, and a
//! product matrix is written as one, so a file written here loads with a
//! plain `np.load` and an `np.save`d float64 array of either layout reads
//! back here.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `:shared`: `NpyArray` (`shape`, `data`) | [`NpyArray`], the same two fields |
//! | `:shared`: `element_count(shape)` | [`element_count`]`(shape)` |
//! | `:reader`: `parse_npy(string_view)` | [`parse_npy`]`(&[u8])` |
//! | `:reader`: `read_npy(path)` | [`read_npy`]`(path)` |
//! | `:writer`: `npy_header(shape)`, a `std::string` | [`npy_header`]`(shape)`, a `Vec<u8>` |
//! | `:writer`: `npy_to_bytes(array)`, a `std::string` | [`npy_to_bytes`]`(array)`, a `Vec<u8>` |
//! | `:writer`: `write_npy(path, array)` | [`write_npy`]`(path, array)` |
//! | `:format`: `kMagic` | the private `MAGIC` |
//! | the reader's and writer's private helpers (`read_le`, `value_start`, `quoted_value`, `bool_value`, `shape_value`, `c_to_fortran_order`, `format_shape`, `layouts_differ`) | the private functions of those names here |
//! | the `static_assert`s of a little-endian IEEE-754 host, which the C++ `memcpy` relies on | nothing: [`f64::from_le_bytes`] and [`f64::to_le_bytes`] are right on any host, and keep every bit, a NaN's payload included |
//! | `io/base`'s `read_file` / `write_file`, which `read_npy` / `write_npy` go through | [`std::fs::read`] / [`std::fs::write`], a failure an [`ErrorKind::Io`] error naming the path |
//!
//! **The reader** accepts versions 1.0 through 3.0 and both layouts: a
//! Fortran-order payload is taken as is, a C-order one is reordered into
//! column-major. Every refusal of a document is an [`ErrorKind::Io`] error,
//! as in the C++: a bad magic, a version outside 1 to 3, a truncated header
//! length or header, a `descr` that is not quoted or not `'<f8'`, a
//! `fortran_order` that is not `True` / `False`, a missing or unterminated
//! shape tuple or an entry of it that is not a non-negative integer, and a
//! payload whose length the shape disagrees with. One is not: a shape whose
//! element count (or byte count) overflows `usize` is
//! [`ErrorKind::InvalidConfig`], as `element_count`'s is in the C++.
//!
//! **The writer** always emits version 1.0 with NumPy's own canonical
//! header, padded with spaces and a newline so the payload begins on a
//! 64-byte boundary, then the column-major payload as is.
//! `'fortran_order'` is `True` only where the C and Fortran layouts differ,
//! as `np.save` writes it, so for every ordinary shape the bytes are
//! `np.save(path, np.asfortranarray(a))`'s. A shape the data disagrees
//! with, or whose element count overflows, or whose header outgrows the
//! 16-bit length field, is [`ErrorKind::InvalidConfig`]: a malformed array,
//! not a file problem.
//!
//! **The header is text only where it is searched.** The C++ reads it as a
//! `string_view` of raw bytes; here it is decoded with
//! [`String::from_utf8_lossy`], which leaves every ASCII byte in place, so
//! the keys and values found, and the documents accepted, are the same.

#![forbid(unsafe_code)]

use std::path::Path;

use fb_core_error_handling::{Error, Result, Status};

#[cfg(doc)]
use fb_core_error_handling::ErrorKind;

/// The 6-byte magic that opens every `.npy` file: `kMagic`.
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// The bytes of one element, `sizeof(double)`.
const ELEMENT_BYTES: usize = std::mem::size_of::<f64>();

/// An n-dimensional float64 array in Fortran (column-major) order:
/// `NpyArray`.
///
/// `shape` carries the dimensions exactly as the header spells them: `[]`
/// is a NumPy scalar (one element), `[n]` a vector, `[rows, cols]` a
/// matrix. `data` holds the elements flattened in Fortran order, the first
/// axis fastest, so a matrix's element (i, j) sits at `data[i + j * rows]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NpyArray {
    /// The dimensions, as the header spells them.
    pub shape: Vec<usize>,
    /// The elements, first axis fastest.
    pub data: Vec<f64>,
}

/// The number of elements `shape` describes: the product of its
/// dimensions, the empty shape (a NumPy scalar) counting one:
/// `element_count`.
///
/// An overflowing product is an [`ErrorKind::InvalidConfig`] error: such a
/// shape cannot describe a real in-memory array.
pub fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| Error::config("NPY shape element count overflows"))
    })
}

/// The little-endian unsigned integer of `nbytes` bytes at `offset`:
/// `read_le`.
fn read_le(bytes: &[u8], offset: usize, nbytes: usize) -> u64 {
    bytes[offset..offset + nbytes]
        .iter()
        .rev()
        .fold(0, |value, &byte| (value << 8) | u64::from(byte))
}

/// Where the header-dict value after `'key'` starts, spaces after the colon
/// skipped, or `None` with no such key, colon or value: `value_start`.
fn value_start(header: &str, key: &str) -> Option<usize> {
    let key_pos = header.find(&format!("'{key}'"))?;
    let mut pos = key_pos + header[key_pos..].find(':')? + 1;
    let bytes = header.as_bytes();
    while pos < bytes.len() && bytes[pos] == b' ' {
        pos += 1;
    }
    (pos < bytes.len()).then_some(pos)
}

/// The quoted string value of `key` (`descr`'s `<f8`): `quoted_value`.
fn quoted_value<'h>(header: &'h str, key: &str) -> Result<&'h str> {
    let start = value_start(header, key)
        .filter(|&start| header.as_bytes()[start] == b'\'')
        .ok_or_else(|| Error::io(format!("NPY header has no quoted '{key}'")))?;
    let close = header[start + 1..]
        .find('\'')
        .ok_or_else(|| Error::io(format!("NPY header '{key}' quote is unterminated")))?;
    Ok(&header[start + 1..start + 1 + close])
}

/// The `True` / `False` value of `key` (`fortran_order`): `bool_value`.
fn bool_value(header: &str, key: &str) -> Result<bool> {
    match value_start(header, key).map(|start| &header[start..]) {
        Some(value) if value.starts_with("True") => Ok(true),
        Some(value) if value.starts_with("False") => Ok(false),
        _ => Err(Error::io(format!("NPY header has no boolean '{key}'"))),
    }
}

/// The shape tuple: `()` is `[]`, `(5,)` is `[5]`, `(2, 3)` is `[2, 3]`:
/// `shape_value`.
///
/// An entry is the run of ASCII digits where it starts, as `from_chars`
/// reads it, so `(2x)` is refused at the `x`; a run past `usize` is refused
/// too.
fn shape_value(header: &str) -> Result<Vec<usize>> {
    let bytes = header.as_bytes();
    let start = value_start(header, "shape")
        .filter(|&start| bytes[start] == b'(')
        .ok_or_else(|| Error::io("NPY header has no shape tuple"))?;
    let close = start
        + header[start..]
            .find(')')
            .ok_or_else(|| Error::io("NPY header shape tuple is unterminated"))?;
    let mut shape = Vec::new();
    let mut pos = start + 1;
    while pos < close {
        while pos < close && matches!(bytes[pos], b' ' | b',') {
            pos += 1;
        }
        if pos == close {
            break;
        }
        let digits = bytes[pos..close]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        let dim = header[pos..pos + digits]
            .parse::<usize>()
            .map_err(|_| Error::io("NPY header shape entry is not a non-negative integer"))?;
        shape.push(dim);
        pos += digits;
    }
    Ok(shape)
}

/// A C-order payload of `shape`, reordered into Fortran order:
/// `c_to_fortran_order`.
///
/// Walks the output in Fortran order with an odometer over the
/// multi-index, first axis fastest, carrying the matching C-order offset
/// along with it.
fn c_to_fortran_order(c_order: &[f64], shape: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; c_order.len()];
    if out.is_empty() {
        return out;
    }
    let mut c_stride = vec![1_usize; shape.len()];
    for k in (1..shape.len()).rev() {
        c_stride[k - 1] = c_stride[k] * shape[k];
    }
    let mut index = vec![0_usize; shape.len()];
    let mut c_offset = 0;
    for element in &mut out {
        *element = c_order[c_offset];
        for k in 0..shape.len() {
            c_offset += c_stride[k];
            index[k] += 1;
            if index[k] < shape[k] {
                break;
            }
            c_offset -= c_stride[k] * shape[k];
            index[k] = 0;
        }
    }
    out
}

/// Parse a `.npy` document from its bytes: `parse_npy`.
///
/// Every refusal is an [`ErrorKind::Io`] error but an overflowing shape's,
/// [`ErrorKind::InvalidConfig`]; the crate doc lists them.
pub fn parse_npy(npy_bytes: &[u8]) -> Result<NpyArray> {
    if npy_bytes.len() < 8 || &npy_bytes[..6] != MAGIC {
        return Err(Error::io("not an NPY file: bad magic"));
    }
    let major = npy_bytes[6];
    if !(1..=3).contains(&major) {
        return Err(Error::io(format!("unsupported NPY format version {major}")));
    }
    let len_bytes = if major == 1 { 2 } else { 4 };
    if npy_bytes.len() < 8 + len_bytes {
        return Err(Error::io("truncated NPY header length"));
    }
    let header_len = read_le(npy_bytes, 8, len_bytes);
    let data_offset = usize::try_from(header_len)
        .ok()
        .and_then(|header_len| (8 + len_bytes).checked_add(header_len))
        .filter(|&offset| offset <= npy_bytes.len())
        .ok_or_else(|| Error::io("truncated NPY header"))?;
    let header = String::from_utf8_lossy(&npy_bytes[8 + len_bytes..data_offset]);

    let descr = quoted_value(&header, "descr")?;
    if descr != "<f8" {
        return Err(Error::io(format!(
            "unsupported NPY dtype '{descr}' (only '<f8')"
        )));
    }
    let fortran = bool_value(&header, "fortran_order")?;
    let shape = shape_value(&header)?;

    let expected = element_count(&shape)?
        .checked_mul(ELEMENT_BYTES)
        .ok_or_else(|| Error::config("NPY shape element count overflows"))?;
    let payload = &npy_bytes[data_offset..];
    if payload.len() != expected {
        return Err(Error::io(format!(
            "NPY payload is {} bytes but the shape needs {expected}",
            payload.len()
        )));
    }

    let data: Vec<f64> = payload
        .as_chunks::<ELEMENT_BYTES>()
        .0
        .iter()
        .map(|&chunk| f64::from_le_bytes(chunk))
        .collect();
    let data = if fortran {
        data
    } else {
        c_to_fortran_order(&data, &shape)
    };
    Ok(NpyArray { shape, data })
}

/// Read the `.npy` file at `path` through [`parse_npy`]: `read_npy`. A file
/// that cannot be read is an [`ErrorKind::Io`] error naming it.
pub fn read_npy(path: &Path) -> Result<NpyArray> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    parse_npy(&bytes)
}

/// A shape as the header spells it, `()`, `(5,)` or `(2, 3)`:
/// `format_shape`. The one-dimensional trailing comma and the `", "`
/// separator are NumPy's tuple repr, byte for byte.
fn format_shape(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_owned(),
        [n] => format!("({n},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            format!("({})", dims.join(", "))
        }
    }
}

/// Whether the C and Fortran layouts of `shape` differ: only when the array
/// is non-empty and at least two axes are longer than one:
/// `layouts_differ`. `np.save` writes `'fortran_order': True` only then.
fn layouts_differ(shape: &[usize]) -> bool {
    !shape.contains(&0) && shape.iter().filter(|&&dim| dim > 1).count() > 1
}

/// The `.npy` header for a float64 array of `shape`: magic, version, length
/// field and the padded dict, ending where the payload begins, on a
/// multiple of 64 bytes: `npy_header`. [`npy_to_bytes`] is this followed by
/// the payload, so a writer that streams the payload emits the same file.
///
/// An overflowing element count, or a header past the 16-bit length field,
/// is an [`ErrorKind::InvalidConfig`] error.
pub fn npy_header(shape: &[usize]) -> Result<Vec<u8>> {
    element_count(shape)?;

    // The canonical dict, then space padding so magic + version + length
    // field + header is a multiple of 64, the final byte a newline.
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': {}, 'shape': {}, }}",
        if layouts_differ(shape) {
            "True"
        } else {
            "False"
        },
        format_shape(shape)
    );
    let unpadded = MAGIC.len() + 2 + 2 + dict.len() + 1;
    let padded = unpadded.next_multiple_of(64);
    let header_len = u16::try_from(padded - (MAGIC.len() + 2 + 2))
        .map_err(|_| Error::config("NPY header does not fit format version 1.0"))?;

    let mut out = Vec::with_capacity(padded);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]); // format version 1.0
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(padded - 1, b' ');
    out.push(b'\n');
    Ok(out)
}

/// Serialize `array` to `.npy` bytes: `npy_to_bytes`.
///
/// A shape whose element count disagrees with `data.len()`, or overflows,
/// is an [`ErrorKind::InvalidConfig`] error naming the shape.
pub fn npy_to_bytes(array: &NpyArray) -> Result<Vec<u8>> {
    let count = element_count(&array.shape)?;
    if count != array.data.len() {
        return Err(Error::config(format!(
            "NPY shape {} describes {count} elements but data has {}",
            format_shape(&array.shape),
            array.data.len()
        )));
    }
    let mut out = npy_header(&array.shape)?;
    out.reserve(array.data.len() * ELEMENT_BYTES);
    for value in &array.data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// Write `array` to `path` as a `.npy` file: `write_npy`. The bytes are
/// formed first, so a malformed array leaves no file; a file that cannot be
/// written is an [`ErrorKind::Io`] error naming it.
pub fn write_npy(path: &Path, array: &NpyArray) -> Status {
    let bytes = npy_to_bytes(array)?;
    std::fs::write(path, bytes)
        .map_err(|e| Error::io(format!("cannot write '{}': {e}", path.display())))
}
