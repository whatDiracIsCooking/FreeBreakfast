//! The density-fitting factor on the GPU, as tasks of a [`Dag`]:
//!
//! ```text
//! M = L Lᵀ             (fb-recipes-cholesky, lower)
//! B = L⁻¹ (P|mn)       (fb-recipes-sys's trsm: left, lower, no transpose)
//! ```
//!
//! so that `(mn|ls) ≈ Σ_Q B_Q,mn B_Q,ls = (mn|P) M⁻¹ (P|ls)`, and J and K
//! are products over B alone. The C++ `src/integrals/eri/density_fitting/fit`
//! is the reference design; its README has the layout and why the factor is
//! the tree's own blocked Cholesky and not `potrf`.
//!
//! [`add_batch_fit`] adds, for a factor `(L, checked)` that
//! [`add_cholesky`] made once over the metric M (**`cholesky`** on the
//! stream and **`cholesky check`** on the CPU, a metric that is not
//! positive definite failing the run with `Numerical`, naming the 1-based
//! leading minor) and a tensor row of `columns` columns:
//!
//! - **`density fit`**, on the stream: the tensor copied into the output
//!   row B, then `B ← L⁻¹ B` by one [`trsm`] over L's lower triangle. It
//!   reads the check's `checked` row beside L, so it runs after the check
//!   and only when the check passed: on a failed factorization it, and
//!   every reader of B, is skipped.
//!
//! The columns are a packed batch of shell pairs'
//! (`fb_integrals_eri_density_fitting_three_center::add_packed_three_center`),
//! each solved apart from every other batch's, so a B built batch by batch
//! (`add_packed_b`) is the one-batch B column for column.
//!
//! **The layout** keeps the C++ one's aux index fastest: the tensor and B
//! are `naux` × `columns`, column-major at leading dimension `naux`,
//! element `(P, c)` at `P + naux · c`. The C++ B is the square
//! `naux` × `nao²`; here a batch's columns are its kept pairs' spherical
//! ones, and the build loop transposes each batch into B's pair-fastest
//! heap row.
//!
//! **Not in place.** The C++ call overwrites the metric with L and the
//! tensor with B. A task writes only its own rows, so L and B are output
//! rows of their own, each filled by a device copy first; the metric's and
//! the tensor's rows end at their last read, and the assigner gives their
//! bytes to whatever starts after (`docs/memory.md`, "Lifetimes").
//!
//! | C++ | Here |
//! |---|---|
//! | `density_fit(res, nao, naux, metric, b)`, the metric overwritten by L and the square `b` by B | [`add_cholesky`] once, then [`add_batch_fit`] per batch of kept pairs, each returning its own output row; no square fit (#1721) |
//! | `cholesky(res, naux, metric)` and its flag, read by one stream sync | [`add_cholesky`]'s `cholesky` and `cholesky check`, the sync the edge between them |
//! | the trsm queued after the sync | the `density fit` task, which reads `checked` and so runs after the check |
//! | `trsm<double>(res, LEFT, LOWER, N, NON_UNIT, naux, nao², 1, metric, naux, b, naux)` (`src/recipes/trsm`) | [`trsm`], the tree's own blocked solve and never the library's, the same modes over the batch's `columns`, on the stream worker's handle |
//! | `nao == 0` or `naux == 0`: an empty success; a negative count: `InvalidConfig` | contracts: both counts are at least 1 |
//! | `nao²` past `int32_t`: `InvalidConfig` | an always-on contract on `columns`, since the BLAS's column count is an `i32` |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | `lindep > 0`, the fit over M's eigenmodes above it (#630) | not ported: no basis the tree carries needs it, and the C++ Eri step never sets it |
//! | `density_fit_footprint(…)`, nothing on the Cholesky path | the tasks' layouts: L, B, and the factor's scratch, read off the plan's table |
//!
//! [`trsm`]: fb_recipes_sys::trsm
//! [`add_cholesky`]: fb_recipes_cholesky::add_cholesky

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::Status;
use fb_core_log::fits_i32;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_recipes_sys::trsm;
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode};

/// Add the fit of a batch of `batch` columns of a three-center tensor of
/// `columns`, `tensor`, against a metric's factor L, to `dag` on GPU
/// worker `worker`, and return B = L⁻¹ `tensor`: an output row of
/// `naux · batch` doubles named `"B"`, in the tensor's layout.
///
/// Batch `k` is the `k`-th `batch` of the `columns`, the last what is
/// left: outside a loop the task fits the first, and in a counted loop of
/// `columns.div_ceil(batch)` trips batch `k` in trip `k`, its
/// `batch.min(columns - k · batch)` columns, from its views' iteration,
/// the front of both rows, the rest of B left. One batch of every column
/// is `batch == columns`.
///
/// `tensor` is an earlier task's output of `naux · batch` doubles,
/// column-major at leading dimension `naux`, element `(P, c)` at
/// `P + naux · c`, left as it was: a packed batch of shell pairs'
/// spherical columns
/// (`fb_integrals_eri_density_fitting_three_center::add_packed_three_center`'s).
/// Each column is
/// solved alone, so a batch's B is its columns of the whole tensor's. B
/// keeps the aux index fastest; a packed B's heap row is pair-fastest
/// (`fb_integrals_eri_density_fitting_pair_map`'s crate doc), so a
/// batch's `(Q, c)` goes to `first + c + n_columns · Q` there, `first` the
/// batch's first column: that copy is a transpose, the build loop's.
/// `factor` is [`add_cholesky`](fb_recipes_cholesky::add_cholesky)'s
/// `(L, checked)` over the `naux` × `naux`
/// metric, added once for every batch fitted against it. The task added
/// is `density fit` (stream), which reads `tensor`, L and `checked`, so it
/// runs after the check and only when the check passed.
///
/// # Panics
/// If a row is of another worker's table ([`Dag::add_gpu`]).
#[debug_requires(naux > 0, "a fitting set of at least one function")]
#[debug_requires(batch > 0, "a batch of at least one column")]
#[debug_requires(batch <= columns, "a batch of at most every column")]
#[requires(fits_i32!(naux), "naux fits the BLAS's i32")]
#[requires(fits_i32!(batch), "batch fits the BLAS's i32")]
#[track_caller]
pub fn add_batch_fit(
    dag: &mut Dag<'_>,
    worker: usize,
    naux: usize,
    columns: usize,
    batch: usize,
    factor: (RowId<Device, f64>, RowId<Pinned, i32>),
    tensor: RowId<Device, f64>,
) -> RowId<Device, f64> {
    let (l, checked) = factor;
    dag.add_gpu(
        "density fit",
        worker,
        Fit {
            tensor,
            l,
            checked,
            columns,
            batch,
            naux,
        },
    )
}

/// GPU: the tensor copied into B, then `B ← L⁻¹ B`.
struct Fit {
    tensor: RowId<Device, f64>,
    l: RowId<Device, f64>,
    /// Read only to order the task after a check that passed.
    checked: RowId<Pinned, i32>,
    /// Every batch's columns, and the first's.
    columns: usize,
    batch: usize,
    naux: usize,
}

impl GpuTask for Fit {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.tensor);
        device.read(self.l);
        host.read(self.checked);
        device.output("B", self.naux * self.batch)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("columns", self.batch), ("naux", self.naux)]
    }

    fn run(
        &self,
        b: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // This trip's columns: the batch's, or what is left for the last.
        let count = self
            .batch
            .min(self.columns - device.iteration() * self.batch);
        let mut b = device.take_mut(*b);
        memcpy_d2d(&b, device.get(self.tensor), worker)?;
        // add_batch_fit checked that naux and batch fit an i32, and a
        // trip's columns are at most the batch; the handle is the worker's,
        // bound to its stream, so the solve follows the copy.
        let (naux, columns) = (self.naux as i32, count as i32);
        trsm(
            worker.blas(),
            wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            wwr_blas_operation::WWR_BLAS_OP_N,
            wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
            naux,
            columns,
            1.0,
            &device.get(self.l),
            naux,
            &mut b,
            naux,
        )
    }
}
