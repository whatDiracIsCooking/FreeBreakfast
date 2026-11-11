//! `../rayon_npy`'s file, a column-major (`fortran_order: True`) float64
//! `.npy`, written by many threads at once with `O_DIRECT`: each write goes
//! from the thread's buffer to the device, bypassing the page cache.
//!
//! `O_DIRECT` needs every write's file offset, length and memory address
//! aligned to the device's logical block, taken here as [`ALIGN`] = 4 KiB. A
//! block of whole columns is almost never aligned (a column is `rows * 8`
//! bytes, and the data starts after a 64- or 128-byte header), so tasks split
//! the *file*, not the matrix: task `s` owns bytes `[s * stripe, (s + 1) *
//! stripe)`, which hold a contiguous range of column-major elements. Stripe 0
//! also carries the header. The file is byte-identical to `np.save`'s.
//!
//! So `fill` takes a flat, column-major range of elements, not a block of
//! columns: `fill(first, values)` fills elements `first..first +
//! values.len()`, element `k` being row `k % rows` of column `k / rows`. A
//! range may start and end mid-column.
//!
//! Two things are meant to let the writes run concurrently on ext4, both done
//! before the first write. They follow my reading of ext4's
//! `ext4_dio_write_checks`, not a measurement: on the disk this was tried on,
//! the device was the bottleneck long before any lock (see the README).
//!
//! - **The file is `fallocate`d.** ext4 runs a direct write under the inode
//!   lock *shared* only when it overwrites blocks that are already allocated
//!   (unwritten extents count). A write into a hole allocates and takes the
//!   lock exclusive, one writer at a time.
//! - **The file is extended to its length rounded up to 4 KiB.** No write
//!   extends the file (another exclusive case), including the last stripe's
//!   padded tail. The file is truncated to its exact length at the end.
//!
//! Unix only, Linux `fallocate` and `O_DIRECT`, little-endian only, as
//! `../rayon_npy`.

use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, Error, ErrorKind};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

pub use fb_rayon_npy::read;

/// The alignment `O_DIRECT` is given: file offset, length and buffer
/// address. 4 KiB covers both 512-byte and 4 KiB logical-block devices.
pub const ALIGN: usize = 4096;
const F64: usize = size_of::<f64>();

/// One aligned page; a `Vec<Page>` is an aligned buffer with no `unsafe`
/// allocation.
#[derive(Clone, Copy)]
#[repr(C, align(4096))]
struct Page([u8; ALIGN]);

/// Writes a `rows` x `cols` matrix to `path` with `O_DIRECT`: one rayon task
/// per `stripe_bytes` of the file (a positive multiple of [`ALIGN`]), each
/// calling `fill(first, values)` for the column-major elements `first..first +
/// values.len()` that stripe holds, then writing it. Each thread reuses one
/// stripe-sized aligned buffer.
///
/// The data is on the device (not necessarily through the device's own
/// cache) when this returns; the allocation metadata is not synced.
pub fn fill_par<F>(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    stripe_bytes: usize,
    fill: F,
) -> io::Result<()>
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if stripe_bytes == 0 || !stripe_bytes.is_multiple_of(ALIGN) {
        return Err(invalid(format!(
            "stripe_bytes {stripe_bytes} is not a positive multiple of {ALIGN}"
        )));
    }
    let data_len = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(F64))
        .ok_or_else(|| invalid(format!("{rows} x {cols} float64 overflows usize bytes")))?;
    let header = fb_rayon_npy::header(rows, cols);
    let h = header.len();
    let total = h + data_len;
    let padded = total.next_multiple_of(ALIGN);

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?;
    file.set_len(padded as u64)?;
    preallocate(&file, padded)?;

    (0..padded.div_ceil(stripe_bytes))
        .into_par_iter()
        .try_for_each_init(
            || vec![Page([0; ALIGN]); stripe_bytes / ALIGN],
            |buf, s| {
                let start = s * stripe_bytes;
                let len = stripe_bytes.min(padded - start);
                let bytes = &mut as_bytes_mut(buf)[..len];
                // The stripe's bytes, in file terms: header (stripe 0 only),
                // then data up to `total`, then padding up to `padded`.
                let lo = if start == 0 {
                    bytes[..h].copy_from_slice(&header);
                    h
                } else {
                    0
                };
                let hi = (total - start).min(len);
                if hi > lo {
                    let first = (start + lo - h) / F64;
                    fill(first, as_f64_mut(&mut bytes[lo..hi]));
                }
                bytes[hi.max(lo)..].fill(0);
                file.write_all_at(bytes, start as u64)
            },
        )?;
    file.set_len(total as u64)
}

/// Copies `values`, a column-major `rows` x `cols` matrix, into `path` with
/// `O_DIRECT`, as [`fill_par`]. The copy into the aligned buffers is
/// unavoidable: `values` is not page-aligned in general.
pub fn write_par(
    path: impl AsRef<Path>,
    rows: usize,
    cols: usize,
    stripe_bytes: usize,
    values: &[f64],
) -> io::Result<()> {
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(invalid(format!(
            "{} values for a {rows} x {cols} matrix",
            values.len()
        )));
    }
    fill_par(path, rows, cols, stripe_bytes, |first, out| {
        out.copy_from_slice(&values[first..first + out.len()]);
    })
}

/// Reserves `[0, len)`'s blocks. Mode 0 keeps them reading as zeros. Unlike
/// `../rayon_npy_mmap`, a filesystem without `fallocate` is refused: every
/// write would then allocate under the exclusive lock, and the parallelism
/// would be gone without a word.
fn preallocate(file: &File, len: usize) -> io::Result<()> {
    // SAFETY: a plain syscall on an fd this function borrows open.
    let rc = unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len as i64) };
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::last_os_error())
    }
}

fn as_bytes_mut(pages: &mut [Page]) -> &mut [u8] {
    // SAFETY: `Page` is `repr(C)` over a byte array with no padding (its size
    // is its alignment); the slice covers exactly `pages`' memory.
    unsafe { std::slice::from_raw_parts_mut(pages.as_mut_ptr().cast(), size_of_val(pages)) }
}

fn as_f64_mut(bytes: &mut [u8]) -> &mut [f64] {
    // Callers pass a range starting at a multiple of 8 within a page-aligned
    // buffer, with a length that is a multiple of 8; checked, not assumed.
    assert!(bytes.as_ptr().align_offset(align_of::<f64>()) == 0 && bytes.len().is_multiple_of(F64));
    // SAFETY: aligned and sized as just asserted; every bit pattern is a
    // valid f64; the result reborrows `bytes` mutably, so it is unique.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / F64) }
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidInput, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Under `target/`, not the temp dir: the container's `/tmp` is overlayfs,
    /// and the point is a filesystem that does `O_DIRECT` and `fallocate`.
    fn scratch(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("test-scratch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn value(k: usize) -> f64 {
        k as f64 * 0.25 - 3.0
    }

    fn reference(n: usize) -> Vec<f64> {
        (0..n).map(value).collect()
    }

    fn fill(first: usize, out: &mut [f64]) {
        for (t, x) in out.iter_mut().enumerate() {
            *x = value(first + t);
        }
    }

    #[test]
    fn byte_identical_to_the_pwrite_writer() {
        // Shapes whose data ends inside stripe 0, exactly on a page, and
        // mid-page after several stripes; stripes of one page and of several.
        for (rows, cols) in [(1, 1), (3, 5), (37, 101), (509, 8), (512, 15), (1000, 77)] {
            let want = scratch(&format!("same_pwrite_{rows}_{cols}.npy"));
            fb_rayon_npy::fill_par(&want, rows, cols, 7, |first, block| {
                fill(first * rows, block)
            })
            .unwrap();
            let want = std::fs::read(&want).unwrap();
            for stripe in [ALIGN, 3 * ALIGN, 64 * ALIGN] {
                let got = scratch(&format!("same_direct_{rows}_{cols}_{stripe}.npy"));
                fill_par(&got, rows, cols, stripe, fill).unwrap();
                assert_eq!(
                    std::fs::read(&got).unwrap(),
                    want,
                    "{rows} x {cols}, stripe {stripe}"
                );
            }
        }
    }

    #[test]
    fn write_par_round_trips() {
        let (rows, cols) = (123, 457);
        let want = reference(rows * cols);
        let path = scratch("write_par.npy");
        write_par(&path, rows, cols, 2 * ALIGN, &want).unwrap();
        assert_eq!(read(&path).unwrap(), (rows, cols, want));
    }

    #[test]
    fn one_page_stripes_from_many_threads() {
        let (rows, cols) = (300, 1000);
        let path = scratch("contended.npy");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| fill_par(&path, rows, cols, ALIGN, fill))
            .unwrap();
        assert_eq!(read(&path).unwrap(), (rows, cols, reference(rows * cols)));
    }

    #[test]
    fn empty_shapes() {
        for (r, c) in [(0, 0), (0, 5), (5, 0)] {
            let path = scratch(&format!("empty_{r}_{c}.npy"));
            write_par(&path, r, c, ALIGN, &[]).unwrap();
            assert_eq!(read(&path).unwrap(), (r, c, vec![]));
        }
    }

    #[test]
    fn bad_calls_are_refused() {
        let path = scratch("refused.npy");
        let kind = |r: io::Result<()>| r.unwrap_err().kind();
        assert_eq!(
            kind(write_par(&path, 4, 3, ALIGN, &[1.0; 11])),
            ErrorKind::InvalidInput
        );
        for stripe in [0, 512, ALIGN + 8] {
            assert_eq!(
                kind(write_par(&path, 4, 3, stripe, &[1.0; 12])),
                ErrorKind::InvalidInput
            );
        }
        assert_eq!(
            kind(fill_par(&path, usize::MAX, 2, ALIGN, fill)),
            ErrorKind::InvalidInput
        );
    }
}
