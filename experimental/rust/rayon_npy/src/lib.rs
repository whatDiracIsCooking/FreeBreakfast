//! One `.npy` file, a column-major (`fortran_order: True`) float64 matrix,
//! written by many threads at once.
//!
//! The header is fixed by the shape alone, so [`NpyWriter::create`] writes it
//! and sizes the file before any data exists. Every column then has a known
//! byte offset, and a thread writes its columns there with a positioned write
//! (`pwrite`, [`FileExt::write_all_at`]) that never touches the file's shared
//! cursor. Writes to disjoint ranges need no lock, so [`NpyWriter`] is `Sync`
//! and its writes take `&self`.
//!
//! [`write_par`] and [`fill_par`] drive it from rayon: the first from a matrix
//! already in memory, the second from a closure that computes a block of
//! columns on whichever thread writes it, so the whole matrix is never held.
//!
//! Neither helper syncs: like `np.save`, the data is in the page cache when
//! they return, visible to any reader. [`NpyWriter::finish`] is the
//! `fdatasync` for a caller that needs it on the device.
//!
//! Unix only (`pwrite`), and little-endian hosts only: the values go to disk
//! as their in-memory bytes, which is what `<f8` means only there.

use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, Error, ErrorKind, Read};
use std::os::unix::fs::FileExt;
use std::path::Path;

#[cfg(not(target_endian = "little"))]
compile_error!(
    "fb_rayon_npy writes values as their native bytes, which is '<f8' only on a little-endian host"
);

const MAGIC: &[u8] = b"\x93NUMPY";
/// numpy pads the header so the data starts on a 64-byte boundary.
const ALIGN: usize = 64;
const F64: usize = size_of::<f64>();

/// A `rows` x `cols` column-major float64 `.npy` file, open for writing.
///
/// The file has its full size from [`create`](Self::create) on; a column
/// nobody writes reads back as zeros. Columns written twice keep the last
/// write. Nothing records which columns were written: completeness is the
/// caller's.
#[derive(Debug)]
pub struct NpyWriter {
    file: File,
    rows: usize,
    cols: usize,
    data_offset: u64,
}

impl NpyWriter {
    /// Creates (or truncates) `path`, writes the version 1.0 header for a
    /// `rows` x `cols` Fortran-order `<f8` array, and extends the file to its
    /// final length.
    pub fn create(path: impl AsRef<Path>, rows: usize, cols: usize) -> io::Result<Self> {
        let data_len = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(F64))
            .ok_or_else(|| invalid(format!("{rows} x {cols} float64 overflows usize bytes")))?;
        let header = header(rows, cols);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all_at(&header, 0)?;
        let data_offset = header.len() as u64;
        file.set_len(data_offset + data_len as u64)?;
        Ok(Self {
            file,
            rows,
            cols,
            data_offset,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Writes whole columns `first_col..first_col + values.len() / rows`,
    /// `values` column-major. Safe from any number of threads at once; ranges
    /// that overlap leave whichever write landed last.
    pub fn write_columns(&self, first_col: usize, values: &[f64]) -> io::Result<()> {
        if self.rows == 0 {
            return if values.is_empty() {
                Ok(())
            } else {
                Err(invalid("a 0-row matrix has no values to write".into()))
            };
        }
        if !values.len().is_multiple_of(self.rows) {
            return Err(invalid(format!(
                "{} values is not a whole number of {}-row columns",
                values.len(),
                self.rows
            )));
        }
        let n = values.len() / self.rows;
        if first_col.checked_add(n).is_none_or(|end| end > self.cols) {
            return Err(invalid(format!(
                "columns {first_col}..{first_col}+{n} run past the matrix's {}",
                self.cols
            )));
        }
        let offset = self.data_offset + (first_col * self.rows * F64) as u64;
        self.file.write_all_at(as_bytes(values), offset)
    }

    /// Flushes the data to the device (`fdatasync`) and closes the file.
    /// Dropping the writer instead closes it without the sync.
    pub fn finish(self) -> io::Result<()> {
        self.file.sync_data()
    }
}

/// Writes `values`, a column-major `rows` x `cols` matrix, to `path`: one
/// rayon task per block of `cols_per_chunk` columns (the last may be short),
/// each writing straight from `values`, no copy.
pub fn write_par(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    cols_per_chunk: usize,
    values: &[f64],
) -> io::Result<()> {
    if cols_per_chunk == 0 {
        return Err(invalid("cols_per_chunk must be at least 1".into()));
    }
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(invalid(format!(
            "{} values for a {rows} x {cols} matrix",
            values.len()
        )));
    }
    let writer = NpyWriter::create(path, rows, cols)?;
    if !values.is_empty() {
        values
            .par_chunks(rows * cols_per_chunk)
            .enumerate()
            .try_for_each(|(k, block)| writer.write_columns(k * cols_per_chunk, block))?;
    }
    Ok(())
}

/// Writes a `rows` x `cols` matrix to `path` without ever holding it: one
/// rayon task per block of `cols_per_chunk` columns, each calling
/// `fill(first_col, block)` to compute the block (column-major, `rows` x
/// `block.len() / rows`; the last block may be short) and writing it. Each
/// thread reuses one block-sized buffer, so memory is `threads x rows x
/// cols_per_chunk` values.
pub fn fill_par<F>(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    cols_per_chunk: usize,
    fill: F,
) -> io::Result<()>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if cols_per_chunk == 0 {
        return Err(invalid("cols_per_chunk must be at least 1".into()));
    }
    let writer = NpyWriter::create(path, rows, cols)?;
    if rows > 0 {
        (0..cols.div_ceil(cols_per_chunk))
            .into_par_iter()
            .try_for_each_init(Vec::new, |buf, k| {
                let first = k * cols_per_chunk;
                let n = cols_per_chunk.min(cols - first);
                buf.resize(rows * n, 0.0);
                fill(first, buf);
                writer.write_columns(first, buf)
            })?;
    }
    Ok(())
}

/// Reads back what this crate writes, and any other 2-D Fortran-order `<f8`
/// `.npy` of format version 1-3: `(rows, cols, values)`, values column-major.
pub fn read(path: impl AsRef<Path>) -> io::Result<(usize, usize, Vec<f64>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let bad = |why: &str| Error::new(ErrorKind::InvalidData, why.to_string());
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(bad("not a .npy file"));
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
        _ => return Err(bad("unsupported .npy version")),
    };
    let header = bytes
        .get(start..start + header_len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| bad("truncated header"))?;
    if !header.contains("'descr': '<f8'") || !header.contains("'fortran_order': True") {
        return Err(bad("not a Fortran-order '<f8' array"));
    }
    let dims: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| bad("no shape"))?
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().map_err(|_| bad("unreadable shape")))
        .collect::<io::Result<_>>()?;
    let [rows, cols] = dims[..] else {
        return Err(bad("not 2-D"));
    };
    let body = &bytes[start + header_len..];
    if body.len() != rows * cols * F64 {
        return Err(bad("data length does not match the shape"));
    }
    let values = body
        .as_chunks::<F64>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    Ok((rows, cols, values))
}

/// The version 1.0 header: magic, version, length, then the dict padded with
/// spaces and a closing newline so the data starts on a 64-byte boundary.
/// Public for `../rayon_npy_mmap`, which writes the same file another way.
pub fn header(rows: usize, cols: usize) -> Vec<u8> {
    let dict = format!("{{'descr': '<f8', 'fortran_order': True, 'shape': ({rows}, {cols}), }}");
    let unpadded = MAGIC.len() + 4 + dict.len() + 1;
    let total = unpadded.next_multiple_of(ALIGN);
    let header_len = total - MAGIC.len() - 4;
    // Two usize values cannot push the dict near version 1.0's u16 limit.
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    out
}

fn as_bytes(values: &[f64]) -> &[u8] {
    // SAFETY: f64 has no padding and every byte pattern is a valid u8; the
    // slice covers exactly `values`' memory and borrows it for its lifetime.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidInput, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fb_rayon_npy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn value(i: usize, j: usize) -> f64 {
        (i as f64) + 1e-3 * (j as f64) - 0.5
    }

    fn reference(rows: usize, cols: usize) -> Vec<f64> {
        (0..cols)
            .flat_map(|j| (0..rows).map(move |i| value(i, j)))
            .collect()
    }

    #[test]
    fn header_is_aligned_and_numpy_shaped() {
        for (r, c) in [(0, 0), (1, 1), (7, 3), (123_456_789, 98_765)] {
            let h = header(r, c);
            assert_eq!(h.len() % ALIGN, 0);
            assert_eq!(*h.last().unwrap(), b'\n');
            let len = u16::from_le_bytes([h[8], h[9]]) as usize;
            assert_eq!(len + 10, h.len());
            let text = std::str::from_utf8(&h[10..]).unwrap();
            assert!(text.starts_with(&format!(
                "{{'descr': '<f8', 'fortran_order': True, 'shape': ({r}, {c}), }}"
            )));
        }
    }

    #[test]
    fn write_par_round_trips_every_chunking() {
        let (rows, cols) = (37, 101);
        let want = reference(rows, cols);
        for cpc in [1, 2, 7, 100, 101, 500] {
            let path = scratch(&format!("write_par_{cpc}.npy"));
            write_par(&path, rows, cols, cpc, &want).unwrap();
            assert_eq!(
                read(&path).unwrap(),
                (rows, cols, want.clone()),
                "cpc {cpc}"
            );
        }
    }

    #[test]
    fn fill_par_round_trips_with_a_short_last_block() {
        let (rows, cols, cpc) = (64, 1000, 3);
        let path = scratch("fill_par.npy");
        fill_par(&path, rows, cols, cpc, |first, block| {
            for (jj, col) in block.chunks_mut(rows).enumerate() {
                for (i, x) in col.iter_mut().enumerate() {
                    *x = value(i, first + jj);
                }
            }
        })
        .unwrap();
        assert_eq!(read(&path).unwrap(), (rows, cols, reference(rows, cols)));
    }

    #[test]
    fn single_column_tasks_from_many_threads() {
        // One task per column on a wide pool, to put as many concurrent
        // pwrites on the one descriptor as possible.
        let (rows, cols) = (3, 20_000);
        let path = scratch("contended.npy");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            fill_par(&path, rows, cols, 1, |first, block| {
                for (i, x) in block.iter_mut().enumerate() {
                    *x = value(i, first);
                }
            })
        })
        .unwrap();
        assert_eq!(read(&path).unwrap(), (rows, cols, reference(rows, cols)));
    }

    #[test]
    fn empty_shapes() {
        for (r, c) in [(0, 0), (0, 5), (5, 0)] {
            let path = scratch(&format!("empty_{r}_{c}.npy"));
            write_par(&path, r, c, 4, &[]).unwrap();
            assert_eq!(read(&path).unwrap(), (r, c, vec![]));
            fill_par(&path, r, c, 4, |_, _| unreachable!()).unwrap();
            assert_eq!(read(&path).unwrap(), (r, c, vec![]));
        }
    }

    #[test]
    fn unwritten_columns_read_as_zero() {
        let path = scratch("partial.npy");
        let w = NpyWriter::create(&path, 2, 3).unwrap();
        w.write_columns(1, &[1.0, 2.0]).unwrap();
        w.finish().unwrap();
        assert_eq!(
            read(&path).unwrap(),
            (2, 3, vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        );
    }

    #[test]
    fn bad_writes_are_refused() {
        let path = scratch("refused.npy");
        let w = NpyWriter::create(&path, 4, 3).unwrap();
        let kind = |r: io::Result<()>| r.unwrap_err().kind();
        assert_eq!(kind(w.write_columns(0, &[1.0; 5])), ErrorKind::InvalidInput);
        assert_eq!(kind(w.write_columns(2, &[1.0; 8])), ErrorKind::InvalidInput);
        assert_eq!(
            kind(w.write_columns(usize::MAX, &[1.0; 4])),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            kind(write_par(&path, 4, 3, 1, &[1.0; 11])),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            kind(write_par(&path, 4, 3, 0, &[1.0; 12])),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            NpyWriter::create(&path, usize::MAX, 2).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
}
