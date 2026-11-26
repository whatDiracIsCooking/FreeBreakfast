//! Density fitting's packed shell-pair map on the device: its per-column
//! index tables staged and uploaded by tasks of this crate
//! ([`add_packed_tables`]), and the two launches that read them
//! ([`PackedTables::pack`], [`PackedTables::gather`]).
//!
//! The map is `fb-integrals-eri-density-fitting-pair-map`'s, whose crate
//! doc has the packed layout: column `c` of a packed vector stands at
//! `(m, n)` of the `nao × nao` matrix and has the weight `w_c` in a packed
//! density, 1 on a diagonal shell pair's block and 2 on any other. Its
//! `Packing::write_tables` fills the tables, and its `Packing::pack` and
//! `Packing::unpack` are the host's reading of them, which these launches
//! equal bitwise: each is a copy, or one multiply by 1 or 2, exact in fp64.
//!
//! - **pack** a symmetric density: `dp[c] = w_c D[m nao + n]`.
//! - **gather** a packed batch of `b` aux functions, `b × n_columns`
//!   pair-fastest (element `(c, q)` at `c + n_columns q`, the layout of a
//!   batch of B), into a square slab of `b` matrices of `nao × nao`
//!   (element `(m, n)` of function `q` at `n + nao (m + nao q)`): every
//!   element written, zero where no kept pair covers it, `(m, n)` and
//!   `(n, m)` the same bits, a diagonal block read from its lower triangle.
//!   With `b = 1` it is the unpack of one packed vector, a J.
//!
//! There is no C++ counterpart: the C++ B is square, `[m][n][P]`, and
//! resident on the device.

use contracts::requires;
use fb_core_arena::{Bufs, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_log::{eq, fits_i32, le};
use fb_core_planner::{CpuTask, Dag};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::Stream;

use crate::{fb_packed_gather, fb_packed_pack, launched};

/// A packed map's index tables on one GPU worker: per column its row `m`,
/// its column `n` and its weight, three device rows the uploads of
/// [`add_packed_tables`] wrote. A task that launches over them declares
/// the read with [`PackedTables::read`] and calls [`PackedTables::pack`] or
/// [`PackedTables::gather`] from its `run`.
///
/// **Why a launcher may trust it.** The kernels store at `(m, n)` and
/// `(n, m)` of the matrix and read `m nao + n` of the density at the
/// indices the tables hold, and those are device data, which no host check
/// at the launch can read. So the type carries them, as
/// [`ClassRows`](crate::ClassRows) carries its records: its fields are
/// private, and the one place that makes a `PackedTables` is
/// [`add_packed_tables`], which adds, with it, the tasks that write its
/// rows.
///
/// - **What the rows hold.** The `packed tables` task writes the pinned
///   rows through the caller's fill and then checks every `m` and `n`
///   inside `[0, nao)`, before the uploads copy them to the device rows.
///   The pinned rows are named only inside this module, so no other task
///   can declare a write to them.
/// - **Who writes them.** Each device row is an output of its upload, so
///   no other task can write it ([`Bufs::take_mut`] refuses a row the
///   task's layout did not declare as its own output).
/// - **When a reader sees them.** A launcher fetches each row through the
///   reading task's [`Bufs::get`], which panics unless that task's layout
///   declared the read; a declared read is an edge, ordered after the
///   uploads, as [`ClassRows`](crate::ClassRows)' are.
/// - **Which plan.** A row names its table, so a `PackedTables` of another
///   plan or worker panics in [`Dag::add_gpu`] or at [`Bufs::get`].
///
/// What the launchers check, in every build, is the rest: each slice's
/// length against `nao` and the column count, and the counts within the
/// kernels' `i32` arguments.
#[derive(Clone, Copy, Debug)]
pub struct PackedTables {
    nao: usize,
    n_columns: usize,
    rows: RowId<Device, i32>,
    cols: RowId<Device, i32>,
    weights: RowId<Device, f64>,
}

/// Add a packed map's index tables to `dag` on GPU worker `worker`, and
/// return them on the device: the CPU task **`packed tables`**, which calls
/// `fill(rows, cols, weights)` on three pinned rows of `n_columns` each and
/// checks every index it left, and their uploads on the worker's lane 0
/// (`upload packed m`, `upload packed n`, `upload packed weight`).
///
/// `fill` is the map's `Packing::write_tables`, over a packing of `nao`
/// functions and `n_columns` columns: it writes each column's `m`, `n` and
/// weight in place, the pinned rows being as long as a row of B
/// (`docs/memory.md`, "The table").
///
/// # Panics
/// When the plan runs, if `fill` leaves an `m` or `n` outside `[0, nao)`,
/// naming the column.
#[requires(le!(1, nao) && fits_i32!(nao), "nao is a positive i32")]
#[requires(
    le!(1, n_columns) && fits_i32!(n_columns),
    "n_columns is a positive i32"
)]
#[track_caller]
pub fn add_packed_tables<'a, F>(
    dag: &mut Dag<'a>,
    worker: usize,
    nao: usize,
    n_columns: usize,
    fill: F,
) -> PackedTables
where
    F: Fn(&mut [i32], &mut [i32], &mut [f64]) + Send + Sync + 'a,
{
    let staged = dag.add_cpu(
        "packed tables",
        worker,
        Tables {
            nao,
            n_columns,
            fill,
        },
    );
    PackedTables {
        nao,
        n_columns,
        rows: dag.upload(worker, staged.rows, "packed m", n_columns),
        cols: dag.upload(worker, staged.cols, "packed n", n_columns),
        weights: dag.upload(worker, staged.weights, "packed weight", n_columns),
    }
}

/// The pinned rows of the `packed tables` task.
#[derive(Clone, Copy)]
struct Staged {
    rows: RowId<Pinned, i32>,
    cols: RowId<Pinned, i32>,
    weights: RowId<Pinned, f64>,
}

/// CPU: the tables filled into pinned rows, and every index checked.
struct Tables<F> {
    nao: usize,
    n_columns: usize,
    fill: F,
}

impl<F> CpuTask for Tables<F>
where
    F: Fn(&mut [i32], &mut [i32], &mut [f64]) + Send + Sync,
{
    type Rows = Staged;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Staged {
        Staged {
            rows: host.output("packed m", self.n_columns),
            cols: host.output("packed n", self.n_columns),
            weights: host.output("packed weight", self.n_columns),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("nao", self.nao), ("n_columns", self.n_columns)]
    }

    fn run(&self, rows: &Staged, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let (m, n) = (host.take_mut(rows.rows), host.take_mut(rows.cols));
        (self.fill)(m, n, host.take_mut(rows.weights));
        check_indices("packed tables", self.nao, m, n);
        Ok(())
    }
}

/// Every `m` and `n` inside `[0, nao)`: what the kernels' stores and reads
/// rest on. `what` names the caller in the message.
///
/// # Panics
/// At the first column outside, naming it and its `(m, n)`.
#[track_caller]
pub(crate) fn check_indices(what: &str, nao: usize, rows: &[i32], cols: &[i32]) {
    let inside = |i: i32| usize::try_from(i).is_ok_and(|i| i < nao);
    for (c, (&m, &n)) in rows.iter().zip(cols).enumerate() {
        assert!(
            inside(m) && inside(n),
            "{what}: column {c} stands at ({m}, {n}), outside a {nao} × {nao} matrix"
        );
    }
}

impl PackedTables {
    /// Declare, in a reading task's layout, that it reads the three rows.
    pub fn read(&self, device: &mut Layout<'_, Device>) {
        device.read(self.rows);
        device.read(self.cols);
        device.read(self.weights);
    }

    /// Packs the `nao × nao` density `d` into `dp`, a column each, on
    /// `stream`: `dp[c] = w_c d[m nao + n]` ([`fb_packed_pack`]), the map's
    /// `Packing::pack` times each column's weight, bitwise. `d` is read as
    /// stored. `device` is the reading task's, whose layout declared
    /// [`PackedTables::read`].
    ///
    /// # Errors
    /// The launch error, as [`fb_kernel_abi::launch_status`] maps it.
    ///
    /// # Panics
    /// If the reading task's layout did not declare the read
    /// ([`Bufs::get`]).
    #[requires(
        eq!(Some(d.len()), self.nao.checked_mul(self.nao)),
        "d is nao² doubles"
    )]
    #[requires(eq!(dp.len(), self.n_columns), "dp is a column each")]
    #[track_caller]
    pub fn pack(
        &self,
        stream: &Stream,
        device: &Bufs<'_, Device>,
        d: &impl DeviceSlice<f64>,
        dp: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let (rows, cols, weights) = (
            device.get(self.rows),
            device.get(self.cols),
            device.get(self.weights),
        );
        let dp = dp.as_mut_ptr();
        // SAFETY: rows, cols and weights are n_columns long, written by
        // add_packed_tables' uploads from pinned rows whose every m and n
        // its task checked inside [0, nao), so each read of d, nao² doubles
        // by the contracts, is inside it, and each store of dp, n_columns
        // doubles, too; nao and n_columns are positive i32s
        // (add_packed_tables' contracts), so the casts are exact; dp has no
        // other user until the enqueued work finishes (DeviceSliceMut's
        // promise), so apart from d; the call only enqueues.
        let status = unsafe {
            fb_packed_pack(
                stream.stream(),
                self.nao as i32,
                self.n_columns as i32,
                rows.as_ptr(),
                cols.as_ptr(),
                weights.as_ptr(),
                d.as_ptr(),
                dp,
            )
        };
        launched("fb_packed_pack", &[], status)
    }

    /// Gathers the packed batch `packed`, `b` aux functions of a column
    /// each, pair-fastest, into the square slab `slab`, `b` matrices of
    /// `nao × nao`, on `stream`: the slab zeroed, then
    /// [`fb_packed_gather`], so every element is written, zero where no
    /// kept pair covers it, `(m, n)` and `(n, m)` the same bits, each
    /// matrix the map's `Packing::unpack` of its function, bitwise. With
    /// `b = 1` it unpacks one packed vector. `device` is the reading
    /// task's, whose layout declared [`PackedTables::read`].
    ///
    /// # Errors
    /// The zero's or the launch's error, as [`fb_kernel_abi::launch_status`]
    /// maps the latter.
    ///
    /// # Panics
    /// If the reading task's layout did not declare the read
    /// ([`Bufs::get`]).
    #[requires(le!(1, b) && fits_i32!(b), "b is a positive i32")]
    #[requires(
        eq!(Some(packed.len()), b.checked_mul(self.n_columns)),
        "packed is b × n_columns doubles"
    )]
    #[requires(
        eq!(Some(slab.len()), self.nao.checked_mul(self.nao).and_then(|s| s.checked_mul(b))),
        "slab is b × nao² doubles"
    )]
    #[track_caller]
    pub fn gather(
        &self,
        stream: &Stream,
        device: &Bufs<'_, Device>,
        b: usize,
        packed: &impl DeviceSlice<f64>,
        slab: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let (rows, cols, weights) = (
            device.get(self.rows),
            device.get(self.cols),
            device.get(self.weights),
        );
        memset::zero(slab, stream)?;
        let slab = slab.as_mut_ptr();
        // SAFETY: as pack's for the tables, so every store at (m, n) and
        // (n, m) of a matrix is inside its nao² doubles and every slab one
        // of b, by the contracts, as every read of packed is inside its
        // b × n_columns; b is a positive i32 (the contracts), so the cast
        // is exact; slab has no other user until the enqueued work finishes
        // (DeviceSliceMut's promise), so apart from packed; the call only
        // enqueues, after the zero on the same stream.
        let status = unsafe {
            fb_packed_gather(
                stream.stream(),
                self.nao as i32,
                self.n_columns as i32,
                b as i32,
                rows.as_ptr(),
                cols.as_ptr(),
                weights.as_ptr(),
                packed.as_ptr(),
                slab,
            )
        };
        launched("fb_packed_gather", &[], status)
    }
}

#[cfg(test)]
mod tests {
    use super::check_indices;

    #[test]
    fn indices_inside_the_matrix_pass() {
        check_indices("packed tables", 3, &[0, 2, 1], &[2, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "column 1 stands at (3, 0), outside a 3 × 3 matrix")]
    fn a_row_past_the_matrix_panics() {
        check_indices("packed tables", 3, &[0, 3], &[0, 0]);
    }

    #[test]
    #[should_panic(expected = "column 0 stands at (0, -1), outside a 3 × 3 matrix")]
    fn a_negative_column_panics() {
        check_indices("packed tables", 3, &[0], &[-1]);
    }
}
