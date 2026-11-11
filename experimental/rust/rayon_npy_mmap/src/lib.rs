//! `../rayon_npy`'s file, a column-major (`fortran_order: True`) float64
//! `.npy`, written by many threads at once through a shared mapping instead
//! of `pwrite`.
//!
//! [`NpyMap::create`] writes the same header, sizes the file, preallocates the
//! data range, and maps the whole file `MAP_SHARED`. The data is then one
//! `&mut [f64]` over the page cache, and rayon's `par_chunks_mut` hands each
//! task a disjoint block of whole columns. A task computes its values straight
//! into the file's pages. That needs no `write()` call, which would take the
//! inode lock and serialize, and no staging buffer to copy from. The only
//! kernel work is a page fault per page touched, and faults on a shared
//! mapping run in parallel.
//!
//! Disjointness is the borrow checker's, not a convention: no `unsafe` hands
//! out a block twice.
//!
//! An I/O error after `create` (disk full, a failing device) cannot come
//! back as an `Err` through a store to mapped memory: the kernel delivers
//! `SIGBUS` instead. Preallocating the data range with `fallocate` at create
//! moves the disk-full case into `create`'s `Err`. A filesystem without
//! `fallocate` falls back to a sparse file and keeps that exposure.
//!
//! Unix only, little-endian only, as `../rayon_npy`.

use memmap2::{MmapMut, MmapOptions};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{self, Error, ErrorKind};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub use fb_rayon_npy::read;

const F64: usize = size_of::<f64>();

/// A `rows` x `cols` column-major float64 `.npy` file, mapped for writing.
/// Unwritten values read back as zeros. Dropping it unmaps the file without
/// syncing (the kernel writes the dirty pages back later, as for `np.save`);
/// [`finish`](Self::finish) is the `msync`.
pub struct NpyMap {
    map: MmapMut,
    rows: usize,
    cols: usize,
    data_offset: usize,
}

impl NpyMap {
    /// Creates (or truncates) `path` with `../rayon_npy`'s header for a `rows`
    /// x `cols` Fortran-order `<f8` array, preallocates it at its final length,
    /// and maps it.
    pub fn create(path: impl AsRef<Path>, rows: usize, cols: usize) -> io::Result<Self> {
        let data_len = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(F64))
            .ok_or_else(|| invalid(format!("{rows} x {cols} float64 overflows usize bytes")))?;
        let header = fb_rayon_npy::header(rows, cols);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all_at(&header, 0)?;
        let total = header.len() + data_len;
        file.set_len(total as u64)?;
        preallocate(&file, header.len(), data_len)?;
        // SAFETY: the mapping is only sound while nobody else truncates or
        // rewrites the file under it. This crate created it and holds the
        // only handle; another process doing that is outside what it can
        // guard, as for any mapped file.
        let map = unsafe { MmapOptions::new().len(total).map_mut(&file)? };
        Ok(Self {
            map,
            rows,
            cols,
            data_offset: header.len(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The whole matrix, column-major, over the file's pages.
    pub fn data_mut(&mut self) -> &mut [f64] {
        let n = self.rows * self.cols;
        let bytes = &mut self.map[self.data_offset..];
        // SAFETY: the mapping is page-aligned and the header length is a
        // multiple of 64, so the data is 8-aligned (asserted, not assumed);
        // `bytes` holds exactly `n * 8` bytes and every bit pattern is a valid
        // f64. The slice reborrows `self.map` mutably, so it is unique.
        assert_eq!(bytes.as_ptr().align_offset(align_of::<f64>()), 0);
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<f64>(), n) }
    }

    /// Writes the dirty pages back and waits for them (`msync(MS_SYNC)`), then
    /// unmaps.
    pub fn finish(self) -> io::Result<()> {
        self.map.flush()
    }
}

/// Copies `values`, a column-major `rows` x `cols` matrix, into `path`: one
/// rayon task per block of `cols_per_chunk` columns (the last may be short).
pub fn write_par(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    cols_per_chunk: usize,
    values: &[f64],
) -> io::Result<()> {
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(invalid(format!(
            "{} values for a {rows} x {cols} matrix",
            values.len()
        )));
    }
    for_each_block(path, rows, cols, cols_per_chunk, |first, block| {
        block.copy_from_slice(&values[first * rows..first * rows + block.len()]);
    })
}

/// Computes a `rows` x `cols` matrix into `path` in place: one rayon task
/// per block of `cols_per_chunk` columns, each calling `fill(first_col,
/// block)` with `block` the file's own pages for those columns (column-major,
/// `rows` x `block.len() / rows`; the last block may be short). No buffer:
/// memory is the page cache the file needs anyway.
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
    for_each_block(path, rows, cols, cols_per_chunk, fill)
}

fn for_each_block<F>(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    cols_per_chunk: usize,
    f: F,
) -> io::Result<()>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if cols_per_chunk == 0 {
        return Err(invalid("cols_per_chunk must be at least 1".into()));
    }
    let mut map = NpyMap::create(path, rows, cols)?;
    let data = map.data_mut();
    if !data.is_empty() {
        data.par_chunks_mut(rows * cols_per_chunk)
            .enumerate()
            .for_each(|(k, block)| f(k * cols_per_chunk, block));
    }
    Ok(())
}

/// Reserves the data range's blocks, so a full disk fails here rather than
/// as `SIGBUS` on a later store. Mode 0 keeps the sparse zeros' meaning (the
/// range reads as zeros). A filesystem that cannot do it keeps the sparse
/// file.
fn preallocate(file: &std::fs::File, offset: usize, len: usize) -> io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    // SAFETY: a plain syscall on an fd this function borrows open.
    let rc = unsafe { libc::fallocate(file.as_raw_fd(), 0, offset as i64, len as i64) };
    if rc == 0 {
        return Ok(());
    }
    let e = Error::last_os_error();
    match e.raw_os_error() {
        Some(libc::EOPNOTSUPP) => Ok(()),
        _ => Err(e),
    }
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidInput, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fb_rayon_npy_mmap_{}", std::process::id()));
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

    fn fill(rows: usize) -> impl Fn(usize, &mut [f64]) + Sync {
        move |first, block| {
            for (jj, col) in block.chunks_mut(rows).enumerate() {
                for (i, x) in col.iter_mut().enumerate() {
                    *x = value(i, first + jj);
                }
            }
        }
    }

    #[test]
    fn byte_identical_to_the_pwrite_writer() {
        let (rows, cols) = (37, 101);
        let a = scratch("same_mmap.npy");
        let b = scratch("same_pwrite.npy");
        fill_par(&a, rows, cols, 7, fill(rows)).unwrap();
        fb_rayon_npy::fill_par(&b, rows, cols, 7, fill(rows)).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
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
    fn single_column_tasks_from_many_threads() {
        // Three rows: many blocks share each page, so neighbouring tasks
        // store into the same page at once.
        let (rows, cols) = (3, 20_000);
        let path = scratch("contended.npy");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| fill_par(&path, rows, cols, 1, fill(rows)))
            .unwrap();
        assert_eq!(read(&path).unwrap(), (rows, cols, reference(rows, cols)));
    }

    #[test]
    fn empty_shapes() {
        for (r, c) in [(0, 0), (0, 5), (5, 0)] {
            let path = scratch(&format!("empty_{r}_{c}.npy"));
            write_par(&path, r, c, 4, &[]).unwrap();
            assert_eq!(read(&path).unwrap(), (r, c, vec![]));
        }
    }

    #[test]
    fn unwritten_values_read_as_zero_and_finish_syncs() {
        let path = scratch("partial.npy");
        let mut m = NpyMap::create(&path, 2, 3).unwrap();
        m.data_mut()[2..4].copy_from_slice(&[1.0, 2.0]);
        m.finish().unwrap();
        assert_eq!(
            read(&path).unwrap(),
            (2, 3, vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        );
    }

    #[test]
    fn bad_calls_are_refused() {
        let path = scratch("refused.npy");
        let kind = |r: io::Result<()>| r.unwrap_err().kind();
        assert_eq!(
            kind(write_par(&path, 4, 3, 1, &[1.0; 11])),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            kind(write_par(&path, 4, 3, 0, &[1.0; 12])),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            NpyMap::create(&path, usize::MAX, 2).err().unwrap().kind(),
            ErrorKind::InvalidInput
        );
    }
}
