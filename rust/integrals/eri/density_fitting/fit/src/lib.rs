//! The density-fitting factor on the GPU, as tasks of a [`Dag`]:
//!
//! ```text
//! M = L Lᵀ             (fb-recipes-cholesky, lower)
//! B = L⁻¹ (P|mn)       (one trsm: left, lower, no transpose)
//! ```
//!
//! so that `(mn|ls) ≈ Σ_Q B_Q,mn B_Q,ls = (mn|P) M⁻¹ (P|ls)`, and J and K
//! are products over B alone. The C++ `src/integrals/eri/density_fitting/fit`
//! is the reference design; its README has the layout and why the factor is
//! the tree's own blocked Cholesky and not `potrf`.
//!
//! [`add_density_fit`] adds, for a metric row M and a tensor row (P|mn):
//!
//! - [`add_cholesky`]'s two tasks over M: **`cholesky`** on the stream, L
//!   an output row, and **`cholesky check`** on the CPU, a metric that is not
//!   positive definite failing the run with `Numerical`, naming the 1-based
//!   leading minor.
//! - **`density fit`**, on the stream: (P|mn) copied into the output row B,
//!   then `B ← L⁻¹ B` by one [`trsm`] over L's lower triangle. It reads the
//!   check's `checked` row beside L, so it runs after the check and only
//!   when the check passed: on a failed factorization it, and every reader
//!   of B, is skipped.
//!
//! **The layout** is the C++ one. The tensor and B are `naux` × `nao²`,
//! column-major at leading dimension `naux`: element `(P, n + nao m)` at
//! `P + naux (n + nao m)`, so P is fastest, then n, then m. B's `[Q][n][m]`
//! is what J (a gemv over `vec(D)`) and K (the `naux` × `nao` block `B_m`
//! at `naux · nao · m`) read. (mn|P) is symmetric in m and n, so which of
//! the two is slower is the producer's choice, and the fit keeps it.
//!
//! **Not in place.** The C++ call overwrites the metric with L and the
//! tensor with B. A task writes only its own rows, so L and B are output
//! rows of their own, each filled by a device copy first; the metric's and
//! the tensor's rows end at their last read, and the assigner gives their
//! bytes to whatever starts after.
//!
//! | C++ | Here |
//! |---|---|
//! | `density_fit(res, nao, naux, metric, b)`, the metric overwritten by L and `b` by B | [`add_density_fit`], returning B's and L's own output rows |
//! | `cholesky(res, naux, metric)` and its flag, read by one stream sync | [`add_cholesky`]'s `cholesky` and `cholesky check`, the sync the edge between them |
//! | the trsm queued after the sync | the `density fit` task, which reads `checked` and so runs after the check |
//! | `wwr::trsm<double, int32_t>(…, LEFT, LOWER, N, NON_UNIT, naux, nao², 1, metric, naux, b, naux)` | [`trsm`], the same modes, on the stream worker's handle |
//! | `nao == 0` or `naux == 0`: an empty success; a negative count: `InvalidConfig` | refused at planning: both counts are at least 1 |
//! | `nao²` past `int32_t`: `InvalidConfig` | an always-on assert, since the BLAS's column count is an `i32` |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | `lindep > 0`, the fit over M's eigenmodes above it (#630) | not ported: no basis the tree carries needs it, and the C++ Eri step never sets it |
//! | `density_fit_footprint(…)`, nothing on the Cholesky path | the tasks' layouts: L, B, and the factor's scratch, read off the plan's table |
//!
//! [`trsm`]: fb_core_wwr::blas::trsm

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::trsm;
use fb_recipes_cholesky::add_cholesky;
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode};

/// Add the density-fitting factor of the `naux` × `naux` Coulomb metric
/// `metric` and the `naux` × `nao²` three-center tensor `tensor` to `dag`
/// on GPU worker `worker`, and return `(B, L)`: B = L⁻¹ (P|mn), an output
/// row of `naux · nao²` doubles named `"B"`, in the tensor's layout; and L,
/// with M = L Lᵀ, an output row of `naux²` doubles named `"L"`, column-major,
/// L in its lower triangle and M's strict upper triangle above it.
///
/// `metric` is an earlier task's output of `naux²` doubles, column-major,
/// of which only the lower triangle is read; `tensor` one of `naux · nao²`
/// doubles, column-major at leading dimension `naux` (P fastest, then n,
/// then m). Both are left as they were. The tasks added are
/// [`add_cholesky`]'s over `metric`, then `density fit` (stream), which
/// reads L and the check's `checked` row, so a metric that is not positive
/// definite fails the run with the check's `Numerical`, naming the leading
/// minor, and neither `density fit` nor any reader of B runs.
///
/// # Errors
/// As [`add_cholesky`]: a `GpuRuntime` error if the loaded
/// `libfb_recipes_kernels.so` is not the one the bindings describe, before
/// anything is added.
///
/// # Panics
/// If `naux` or `nao²` does not fit an `i32`, the BLAS's integer, or a row
/// is of another worker's table ([`Dag::add`]).
#[debug_requires(naux > 0, "a fitting set of at least one function")]
#[debug_requires(nao > 0, "an orbital basis of at least one function")]
#[track_caller]
pub fn add_density_fit(
    dag: &mut Dag<'_>,
    worker: usize,
    nao: usize,
    naux: usize,
    metric: RowId<Device, f64>,
    tensor: RowId<Device, f64>,
) -> Result<(RowId<Device, f64>, RowId<Device, f64>)> {
    assert!(
        i32::try_from(naux).is_ok(),
        "add_density_fit: naux = {naux} does not fit the BLAS's i32"
    );
    assert!(
        nao.checked_mul(nao)
            .is_some_and(|pairs| i32::try_from(pairs).is_ok()),
        "add_density_fit: nao² = {nao}² does not fit the BLAS's i32"
    );
    let (l, checked) = add_cholesky(dag, worker, "L", naux, metric)?;
    let b = dag.add(
        "density fit",
        worker,
        Fit {
            tensor,
            l,
            checked,
            nao,
            naux,
        },
    );
    Ok((b, l))
}

/// GPU: (P|mn) copied into B, then `B ← L⁻¹ B`.
struct Fit {
    tensor: RowId<Device, f64>,
    l: RowId<Device, f64>,
    /// Read only to order the task after a check that passed.
    checked: RowId<Pinned, i32>,
    nao: usize,
    naux: usize,
}

impl Task for Fit {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.tensor);
        device.read(self.l);
        host.read(self.checked);
        device.output("B", self.naux * self.nao * self.nao)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("nao", self.nao), ("naux", self.naux)]
    }

    fn run(
        &self,
        b: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut b = device.take_mut(*b);
        memcpy_d2d(&b, device.get(self.tensor), worker)?;
        // add_density_fit checked that naux and nao² fit an i32; the handle
        // is the worker's, bound to its stream, so the solve follows the copy.
        let (naux, pairs) = (self.naux as i32, (self.nao * self.nao) as i32);
        trsm(
            worker.blas(),
            wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            wwr_blas_operation::WWR_BLAS_OP_N,
            wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
            naux,
            pairs,
            1.0,
            &device.get(self.l),
            naux,
            &mut b,
            naux,
        )
    }
}
