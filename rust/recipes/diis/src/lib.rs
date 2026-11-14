//! Pulay DIIS, direct inversion of the iterative subspace, on the GPU, as
//! tasks of a [`Dag`]. The C++ `src/recipes/diis` is the reference design,
//! and its README has the algorithm at length.
//!
//! DIIS accelerates a fixed-point iteration by replacing the current
//! iterate with a combination `F* = Σ_i c_i F_i` of recent ones, the
//! coefficients minimizing the norm of the same combination of their
//! residuals `e_i` subject to `Σ_i c_i = 1`: the bordered system
//!
//! ```text
//! [ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
//! [   -1^T      0 ] [ l ] = [ -1 ]
//! ```
//!
//! For the SCF, `F_i` is the Fock matrix and `e_i` the orthogonal-basis
//! commutator; this crate sees only opaque vectors of `len` doubles.
//!
//! **The history is rows a loop carries.** The ring of iterates and the
//! ring of residuals, each `len` × `cap` column-major with one pair per
//! column, and the `cap` × `cap` residual Gram live from one iteration of
//! a loop to the next ([`Dag::add_loop`]): [`add_diis`], before the loop,
//! adds the task that produces them, `diis history`, and
//! [`Diis::add_push`], in the loop's body, the task that writes them,
//! `diis push`, their last user there (`docs/memory.md`, "Nested plans").
//! So the rows are the parent's, live across the whole loop, and a replay
//! allocates nothing.
//!
//! **The cursor is host state.** How many pairs the ring holds and which
//! column is the oldest are two integers the push reads and advances when
//! it enqueues, held by the tasks and every clone of the [`Diis`] handle.
//! `diis history` empties it when it runs, so every run of the plan starts
//! a fresh subspace, and [`Diis::reset`] empties it between iterations,
//! from the loop's decision, when a caller restarts the subspace. Neither
//! touches the rows: the ring is written before it is read, column by
//! column, so what an earlier subspace left is never read.
//!
//! **One push is one stream task**, the C++ `push_and_extrapolate` in
//! order: the iterate and the residual copied into the ring's slot (the
//! next column while the ring grows, the oldest once it is full, so the
//! valid columns are always a prefix and every BLAS call sees one base and
//! one stride), the Gram's changed row and column by two gemvs, then with
//! two or more pairs the solve (`libfb_recipes_kernels.so`'s one-block
//! kernel, [`diis_solve`]) and the extrapolating gemv into the output. The
//! sum over the ring's rotated columns is order-invariant, so the result
//! agrees with an age-ordered solve to rounding, not bit for bit. A
//! singular subspace (a pivot below `1e-14`) makes the coefficients the
//! unit selector at the slot just written, so the output is the newest
//! iterate, bit for bit, and the flag the push copies into a pinned row says
//! so, for a host node to read: a loop's decision, which may log it and
//! reset.
//!
//! | C++ | Here |
//! |---|---|
//! | `kDiisMaxHistoryLimit` | [`DIIS_MAX_HISTORY_LIMIT`], asserted against the kernel library's `FB_DIIS_SOLVE_MAX_DIM` |
//! | `Diis(res, len, max_history)`, its five buffers allocated, `max_history` clamped to 1..64 | [`add_diis`], the history's three rows produced before the loop; the coefficients and the flag the push's scratch; `max_history` a contract, not a clamp |
//! | `diis_footprint(len, max_history)` | nothing: the rows are the layouts', and the plan's table is the footprint |
//! | `push_and_extrapolate(res, cur_fock, cur_residual, out_fock)` | [`Diis::add_push`]'s `diis push` task, one per iteration |
//! | `out_fock`, left untouched with fewer than two pairs, and allowed to alias `cur_fock` | the push's own output row, which it writes on every push, with the iterate itself when there are fewer than two pairs: a task writes only its own rows, so there is no alias, and what an untouched aliased output held is what that copy holds |
//! | `last_extrapolation_singular()`, one stream synchronize per push | the flag copied into the push's pinned output row, which a later CPU node reads after the executor waited, as the C++ synchronize did |
//! | `reset()` | [`Diis::reset`], and `diis history` at the start of every run |
//! | `size()`, `capacity()`, `length()` | nothing: the cursor is the tasks' |
//! | `res.dev_idx()` checked against the constructor's, and `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | `len` past `INT32_MAX`: `InvalidConfig` | a panic at planning; `len == 0` refused by a contract |
//! | a GPU failure resets the history | the same: a failed push empties the cursor, and the run fails with its error |
//! | `wwrMemcpyAsync` into a column of the ring | [`copy::d2d`] into a [`split_at_mut`] part of the ring's row |
//! | `wwr::gemv<double, int32_t>` with `kOne`, `kZero` | [`gemv`], the same modes, on the stream worker's handle, the Gram's `x` the incoming residual row (the copy in the ring holds the same values) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`fb_recipes_sys::check_library`] first, at planning |
//!
//! [`copy::d2d`]: fb_core_wwr::copy::d2d
//! [`split_at_mut`]: fb_core_wwr::device_slice::split_at_mut
//! [`gemv`]: fb_core_wwr::blas::gemv

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex, PoisonError};

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::gemv;
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use fb_recipes_sys::{FB_DIIS_SOLVE_MAX_DIM, diis_solve};
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

/// The largest history kept: the C++ `kDiisMaxHistoryLimit`. The solve
/// holds the bordered system of one more order in one block's statically
/// sized shared memory, the kernel library's `FB_DIIS_SOLVE_MAX_DIM`.
pub const DIIS_MAX_HISTORY_LIMIT: usize = 64;

const _: () = assert!(DIIS_MAX_HISTORY_LIMIT + 1 == FB_DIIS_SOLVE_MAX_DIM as usize);

/// A DIIS history in a plan: its three rows, the cursor its tasks share,
/// and what a push needs to be added. Made by [`add_diis`]; a clone is the
/// same history, so a loop's decision may hold one to [`Diis::reset`] it.
#[derive(Clone)]
pub struct Diis {
    worker: usize,
    name: String,
    len: usize,
    cap: usize,
    history: History,
    cursor: Arc<Mutex<Cursor>>,
}

/// The rows a loop carries: the two rings and the Gram.
#[derive(Clone, Copy)]
struct History {
    iterates: RowId<Device, f64>,
    residuals: RowId<Device, f64>,
    gram: RowId<Device, f64>,
}

/// How many pairs the ring holds, and its oldest column once it is full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cursor {
    size: usize,
    head: usize,
}

impl Cursor {
    /// Take the next push into a ring of `cap` columns: returns the column
    /// it writes and the pairs held after it. The next column while the
    /// ring grows, the oldest once it is full, so the valid columns are
    /// always `0..m`.
    fn push(&mut self, cap: usize) -> (usize, usize) {
        if self.size < cap {
            let slot = self.size;
            self.size += 1;
            (slot, self.size)
        } else {
            let slot = self.head;
            self.head = (self.head + 1) % cap;
            (slot, cap)
        }
    }
}

/// Lock the cursor; a panic elsewhere while it was held left it two
/// integers, which every value of is a cursor.
fn lock(cursor: &Mutex<Cursor>) -> std::sync::MutexGuard<'_, Cursor> {
    cursor.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Add a DIIS history for vectors of `len` doubles, at most `max_history`
/// pairs, to `dag` on GPU worker `worker`, and return its handle: the task
/// `diis history` (stream), which produces the rows `"{name} iterates"`
/// and `"{name} residuals"`, `len` × `max_history` doubles each, and
/// `"{name} gram"`, `max_history²` doubles, and empties the cursor when it
/// runs. Add it before the loop whose body pushes ([`Diis::add_push`]).
///
/// `len` is an element count, not a matrix order: `n²` for one `n` × `n`
/// Fock matrix.
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_recipes_sys::check_library`]): planning is a run's first call into
/// the library.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_recipes_kernels.so` is not the one the bindings describe.
///
/// # Panics
/// If `len` does not fit an `i32`, the BLAS's integer.
#[debug_requires(len > 0, "a vector of at least one element")]
#[debug_requires(
    (1..=DIIS_MAX_HISTORY_LIMIT).contains(&max_history),
    "a history of 1 to DIIS_MAX_HISTORY_LIMIT pairs"
)]
#[track_caller]
pub fn add_diis(
    dag: &mut Dag<'_>,
    worker: usize,
    name: &str,
    len: usize,
    max_history: usize,
) -> Result<Diis> {
    assert!(
        i32::try_from(len).is_ok(),
        "add_diis: len = {len} does not fit the BLAS's i32"
    );
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    let cursor = Arc::new(Mutex::new(Cursor::default()));
    let history = dag.add(
        "diis history",
        worker,
        Produce {
            name: name.to_owned(),
            len,
            cap: max_history,
            cursor: Arc::clone(&cursor),
        },
    );
    Ok(Diis {
        worker,
        name: name.to_owned(),
        len,
        cap: max_history,
        history,
        cursor,
    })
}

impl Diis {
    /// Add one push of the pair (`iterate`, `residual`), each an earlier
    /// task's output of `len` doubles, to `dag` in a loop's body, on the
    /// history's worker, and return `(out, singular)`: an output row of
    /// `len` doubles, named `name`, and a pinned output row of one `i32`,
    /// `"{name} singular"`.
    ///
    /// The task, `diis push` (stream), writes the history's three rows as
    /// their last user in the loop, and with its scratch
    /// `"{name} coefficients"` and `"{name} singular"` on the device writes
    /// `out`: `Σ_i c_i F_i` over the pairs held once there are two or more,
    /// else the iterate itself. `singular` is 1 when the subspace was
    /// singular and `out` is the newest iterate, else 0, written on every
    /// push; a CPU node reads it.
    ///
    /// # Panics
    /// As [`Dag::add`], if a row is of another worker's table.
    #[track_caller]
    pub fn add_push(
        &self,
        dag: &mut Dag<'_>,
        iterate: RowId<Device, f64>,
        residual: RowId<Device, f64>,
    ) -> (RowId<Device, f64>, RowId<Pinned, i32>) {
        let rows = dag.add(
            "diis push",
            self.worker,
            Push {
                diis: self.clone(),
                iterate,
                residual,
            },
        );
        (rows.out, rows.singular)
    }

    /// Drop every pair, so the next push starts a fresh subspace. The rows
    /// are kept. Call it between pushes, from a CPU node the executor runs
    /// after the push, such as the loop's decision.
    pub fn reset(&self) {
        *lock(&self.cursor) = Cursor::default();
    }
}

/// GPU, before the loop: the history's rows, and an empty cursor.
struct Produce {
    name: String,
    len: usize,
    cap: usize,
    cursor: Arc<Mutex<Cursor>>,
}

impl Task for Produce {
    type Rows = History;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> History {
        let ring = self.len * self.cap;
        History {
            iterates: device.output(&format!("{} iterates", self.name), ring),
            residuals: device.output(&format!("{} residuals", self.name), ring),
            gram: device.output(&format!("{} gram", self.name), self.cap * self.cap),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len), ("cap", self.cap)]
    }

    fn run(
        &self,
        _: &History,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        // Nothing to write: every column and Gram entry is written by the
        // push that brings it into the subspace, before any read of it.
        *lock(&self.cursor) = Cursor::default();
        Ok(())
    }
}

/// GPU, in the loop's body: one push and its extrapolation.
struct Push {
    diis: Diis,
    iterate: RowId<Device, f64>,
    residual: RowId<Device, f64>,
}

#[derive(Clone, Copy)]
struct PushRows {
    out: RowId<Device, f64>,
    coefficients: RowId<Device, f64>,
    flag: RowId<Device, i32>,
    singular: RowId<Pinned, i32>,
}

impl Task for Push {
    type Rows = PushRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> PushRows {
        let d = &self.diis;
        device.read(self.iterate);
        device.read(self.residual);
        device.write(d.history.iterates);
        device.write(d.history.residuals);
        device.write(d.history.gram);
        PushRows {
            coefficients: device.scratch(&format!("{} coefficients", d.name), d.cap),
            flag: device.scratch(&format!("{} singular", d.name), 1),
            out: device.output(&d.name, d.len),
            singular: host.output(&format!("{} singular", d.name), 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.diis.len), ("cap", self.diis.cap)]
    }

    fn run(
        &self,
        rows: &PushRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut cursor = lock(&self.diis.cursor);
        let (slot, m) = cursor.push(self.diis.cap);
        let status = self.push(slot, m, rows, device, host, worker);
        // A failure part-way may leave a column or a Gram entry
        // half-written, so it drops the whole subspace.
        if status.is_err() {
            *cursor = Cursor::default();
        }
        status
    }
}

impl Push {
    /// Enqueue the push into column `slot`, `m` pairs held after it.
    fn push(
        &self,
        slot: usize,
        m: usize,
        rows: &PushRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let (len, cap, history) = (self.diis.len, self.diis.cap, self.diis.history);
        let iterate = device.get(self.iterate);
        let residual = device.get(self.residual);
        let mut iterates = device.take_mut(history.iterates);
        let mut residuals = device.take_mut(history.residuals);
        let mut gram = device.take_mut(history.gram);
        let mut out = device.take_mut(rows.out);
        let mut coefficients = device.take_mut(rows.coefficients);
        let mut flag = device.take_mut(rows.flag);

        // The pair into column `slot` of each ring.
        for (ring, pair) in [(&mut iterates, iterate), (&mut residuals, residual)] {
            let (_, mut tail) = split_at_mut(ring, slot * len);
            let (mut column, _) = split_at_mut(&mut tail, len);
            copy::d2d(&mut column, &pair, worker)?;
        }

        // The Gram's changed products, <e_j, e_slot>, straight into the
        // upper triangle the solve reads, at min(j, slot) cap + max(j,
        // slot): j = 0..=slot down column `slot` (from gram + slot, stride
        // cap), j = slot..m along row `slot` (from gram + slot cap + slot,
        // stride 1). Both write the diagonal. Every push rewrites them, so
        // entry (a, c) is always the push of column max(a, c)'s.
        // add_diis checked that len fits an i32; cap is at most 64.
        let blas = worker.blas();
        let (len32, cap32) = (len as i32, cap as i32);
        {
            let (_, mut column) = split_at_mut(&mut gram, slot);
            gemv(
                blas,
                WWR_BLAS_OP_T,
                len32,
                slot as i32 + 1,
                1.0,
                &residuals,
                len32,
                &residual,
                1,
                0.0,
                &mut column,
                cap32,
            )?;
        }
        if m > slot + 1 {
            let (_, newer) = split_at_mut(&mut residuals, slot * len);
            let (_, mut row) = split_at_mut(&mut gram, slot * cap + slot);
            gemv(
                blas,
                WWR_BLAS_OP_T,
                len32,
                (m - slot) as i32,
                1.0,
                &newer,
                len32,
                &residual,
                1,
                0.0,
                &mut row,
                1,
            )?;
        }

        if m < 2 {
            // One pair: nothing to extrapolate, and the output is the
            // iterate, which is what the C++ in-place call left in it.
            memcpy_d2d(&out, iterate, worker)?;
            memset::zero(&mut flag, worker)?;
        } else {
            diis_solve(worker, m, cap, slot, &gram, &mut coefficients, &mut flag)?;
            // F* = F_hist c over the ring's columns 0..m, in their rotated
            // order: the sum does not depend on it.
            gemv(
                blas,
                WWR_BLAS_OP_N,
                len32,
                m as i32,
                1.0,
                &iterates,
                len32,
                &coefficients,
                1,
                0.0,
                &mut out,
                1,
            )?;
        }
        memcpy_d2p(host, rows.singular, &flag, worker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier 3: eleven pushes into a ring of four write columns 0 to 3 while
    /// it grows, then the oldest, in turn; the pairs held climb to four and
    /// stay. Emptied, the cursor starts over at column 0.
    #[test]
    fn the_cursor_grows_then_overwrites_the_oldest() {
        let mut cursor = Cursor::default();
        let pushes: Vec<(usize, usize)> = (0..11).map(|_| cursor.push(4)).collect();
        assert_eq!(
            pushes,
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 4),
                (1, 4),
                (2, 4),
                (3, 4),
                (0, 4),
                (1, 4),
                (2, 4)
            ]
        );
        cursor = Cursor::default();
        assert_eq!(cursor.push(4), (0, 1));
    }
}
