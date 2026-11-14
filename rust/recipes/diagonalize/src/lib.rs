//! The symmetric eigendecomposition `A = U diag(w) Uᵀ` on the GPU, as tasks
//! of a [`Dag`]: the dense solver's legacy `syevd` through
//! [`fb_core_wwr::solver::syevd`], on the stream worker's solver handle.
//! The C++ `src/recipes/diagonalize` is the reference design.
//!
//! [`add_diagonalize`] adds, for one symmetric matrix row:
//!
//! - **`syevd`**, on the worker's stream: A copied into the output row U,
//!   then `syevd` over it (the lower triangle read, the eigenvectors written
//!   over it, the eigenvalues into the output row w in ascending order),
//!   its work a scratch row of exactly `lwork` doubles and its flag a
//!   device scratch row, copied into a pinned row as the task's last work.
//! - **`syevd check`**, on the CPU: the flag read from that pinned row, after
//!   the executor waited for `syevd`. Non-zero fails the run.
//!
//! Two tasks because the C++ call synchronizes to read the flag, and a task
//! has no host wait inside it: the wait is the edge from `syevd` to the
//! check.
//!
//! **`lwork` is a dimension.** The vendor's work query needs a handle, so it
//! cannot run in a device-free planner; as the C++ `diagonalize_footprint`
//! takes `lwork` rather than a handle, the layout takes it as a number. A
//! run asks the device first, [`syevd_lwork`] on a solver handle of the
//! device it will run on, and a plan test passes any number
//! (`docs/gpu_layer.md`, "Scratch: the workspace query").
//!
//! | C++ | Here |
//! |---|---|
//! | `diagonalize<T>(res, n, a, w)`, A overwritten by U | [`add_diagonalize`]'s `syevd` task, A copied into its own output row U first, since a task writes only its own rows |
//! | `res.device_workspace()`, grown to `syevd_lwork` at the call | the task's `syevd work` scratch row, `lwork` doubles, laid out at planning |
//! | `res.info()` and `res.fetch_info()`, one stream sync | the `syevd info` device scratch row, its pinned copy, and the `syevd check` task the executor waits for `syevd` before |
//! | `info < 0`: `GpuRuntime`; `info > 0`: `Numerical` | the same, from `syevd check` |
//! | `syevd_lwork<T>(res, n)` | [`syevd_lwork`], on a [`SolverHandle`]: [`solver::syevd_lwork`] at this crate's modes |
//! | `wwr::syevd<T>`, raw pointers | [`solver::syevd`], over the task's rows, each extent checked |
//! | `diagonalize_footprint<T>(lwork)` | the layout, read off by the planner |
//! | `res.solver()`, the `DeviceResources`' handle | the stream worker's handle, [`DeviceStream::solver`], made with the backend and bound to the worker's stream |
//! | `float` and `double` | `f64` only, until a caller needs `f32` |
//! | `n <= 0`: an empty success | refused at planning: `n` is at least 1 |

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, HostKind, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::solver::{self, SolverHandle};
use wwr_sys::{wwr_solver_eig_mode, wwr_solver_fill_mode};

/// syevd's eigenvector mode and triangle, shared by the query and the call
/// so the two cannot disagree.
const EIG_MODE: wwr_solver_eig_mode = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
const FILL_MODE: wwr_solver_fill_mode = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;

/// syevd's work for order `n`, in doubles: the solver's own query on
/// `solver`, a host computation, with the eigenvector mode and triangle
/// [`add_diagonalize`]'s call uses. 0 for `n` 0.
///
/// Call it before planning, on a handle of the device the plan will run on,
/// and hand the number to [`add_diagonalize`]: the query needs a handle and
/// a planner has none.
///
/// # Errors
/// The query's `GpuRuntime` error, naming `syevd_lwork` and `n`.
///
/// # Panics
/// If `n` does not fit an `i32`, the legacy API's integer.
pub fn syevd_lwork(solver: &SolverHandle, n: usize) -> Result<usize> {
    if n == 0 {
        return Ok(0);
    }
    let n = i32::try_from(n).expect("syevd_lwork: n fits the solver's i32");
    solver::syevd_lwork::<f64>(solver, EIG_MODE, FILL_MODE, n, n)
}

/// Add the eigendecomposition of the symmetric `n` × `n` matrix `a` to `dag`
/// on GPU worker `worker`, and return `(U, w)`: an output row of `n²`
/// doubles, column-major, column `j` the eigenvector of `w[j]`, named
/// `"{name} U"`, and an output row of the `n` eigenvalues in ascending
/// order, named `"{name} w"`.
///
/// `a` is an earlier task's output of `n²` doubles, column-major; only its
/// lower triangle is read, and it is left as it was. `lwork` is
/// [`syevd_lwork`] at `n` on the device the plan will run on; the work row
/// is exactly that long. The tasks added are `syevd` (stream) and
/// `syevd check` (CPU), which fails the run when the solver's flag is
/// non-zero: a positive flag (no convergence) as `Numerical`, a negative one
/// (an argument refused, such as an `lwork` too small) as `GpuRuntime`.
///
/// A task reading U or w is ordered after `syevd`, not after the check: on
/// a failed flag it may have run on what syevd left, and the run's result
/// is the check's error either way.
///
/// Requires, checked in debug builds: `n` is at least 1.
///
/// # Panics
/// If `n²` or `lwork` does not fit an `i32`, or `a` is a row of another
/// worker's table ([`Dag::add`]).
#[debug_requires(n > 0, "a matrix of at least one row")]
#[track_caller]
pub fn add_diagonalize(
    dag: &mut Dag<'_>,
    worker: usize,
    name: &str,
    n: usize,
    lwork: usize,
    a: RowId<Device, f64>,
) -> (RowId<Device, f64>, RowId<Device, f64>) {
    assert!(
        n.checked_mul(n).is_some_and(|n2| i32::try_from(n2).is_ok()),
        "add_diagonalize: n = {n}: n² does not fit the solver's i32"
    );
    assert!(
        i32::try_from(lwork).is_ok(),
        "add_diagonalize: lwork = {lwork} does not fit the solver's i32"
    );
    let rows = dag.add(
        "syevd",
        worker,
        Syevd {
            a,
            n,
            lwork,
            name: name.to_owned(),
        },
    );
    dag.add("syevd check", worker, Check { info: rows.info });
    (rows.u, rows.w)
}

/// GPU: A copied into U, then syevd over U, then the flag into a pinned row.
struct Syevd {
    a: RowId<Device, f64>,
    n: usize,
    lwork: usize,
    /// The outputs' row-name prefix.
    name: String,
}

#[derive(Clone, Copy)]
struct SyevdRows {
    u: RowId<Device, f64>,
    w: RowId<Device, f64>,
    work: RowId<Device, f64>,
    flag: RowId<Device, i32>,
    info: RowId<Pinned, i32>,
}

impl Task for Syevd {
    type Rows = SyevdRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> SyevdRows {
        device.read(self.a);
        SyevdRows {
            u: device.output(&format!("{} U", self.name), self.n * self.n),
            w: device.output(&format!("{} w", self.name), self.n),
            work: device.scratch("syevd work", self.lwork),
            flag: device.scratch("syevd info", 1),
            info: host.output("syevd info", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("lwork", self.lwork)]
    }

    fn run(
        &self,
        rows: &SyevdRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut u = device.take_mut(rows.u);
        memcpy_d2d(&u, device.get(self.a), worker)?;
        let mut w = device.take_mut(rows.w);
        let (mut work, mut flag) = (device.take_mut(rows.work), device.take_mut(rows.flag));
        // The planner checked n² fits an i32. The handle is the worker's,
        // bound to its stream, so syevd enqueues there, after the copy into
        // U; it checks the work row against its own query.
        solver::syevd(
            worker.solver(),
            EIG_MODE,
            FILL_MODE,
            self.n as i32,
            &mut u,
            self.n as i32,
            &mut w,
            &mut work,
            &mut flag,
        )?;
        memcpy_d2p(host, rows.info, &flag, worker)
    }
}

/// CPU: syevd's flag read from its pinned row, after the executor waited
/// for the copy. Generic over the host kind and stream, so the tier-3 test
/// hands it a flag from a fixture on `Heap` rows and `HostOnly` workers.
struct Check<H: HostKind> {
    info: RowId<H, i32>,
}

impl<H: HostKind, S> Task<H, S> for Check<H> {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) {
        host.read(self.info);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Device>, host: &Bufs<'_, H>, _: Lane<'_, S>) -> Status {
        // info < 0 names a refused argument (a bug in the caller, not in A);
        // info > 0 is non-convergence, which is A's doing.
        match host.get(self.info)[0] {
            0 => Ok(()),
            info if info < 0 => Err(Error::gpu(format!(
                "syevd rejected argument {} (info = {info})",
                -info
            ))),
            info => Err(Error::numerical(format!(
                "syevd did not converge (info = {info})"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fb_core_arena::Heap;
    use fb_core_error_handling::ErrorKind;
    use fb_core_thread_pool::HostOnly;

    /// CPU: a fixture's flag written into a host row.
    struct Flag(i32);

    impl Task<Heap, ()> for Flag {
        type Rows = RowId<Heap, i32>;
        const GPU: bool = false;

        fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Heap>) -> Self::Rows {
            host.output("syevd info", 1)
        }

        fn run(
            &self,
            info: &Self::Rows,
            _: &Bufs<'_, Device>,
            host: &Bufs<'_, Heap>,
            _: Lane<'_, ()>,
        ) -> Status {
            host.take_mut(*info)[0] = self.0;
            Ok(())
        }
    }

    /// The check of the flag `info`, run once on `Heap` rows and no device.
    fn check(info: i32) -> Status {
        let mut dag: Dag<'static, Heap, ()> = Dag::new();
        let row = dag.add("flag", 0, Flag(info));
        dag.add("syevd check", 0, Check { info: row });
        dag.plan()
            .instantiate(&[0])
            .run(&HostOnly::new(0), 1)
            .result
    }

    /// Tier 3: a zero flag passes; a positive one is `Numerical` and a
    /// negative one `GpuRuntime`, each naming the flag and the check.
    #[test]
    fn the_check_reads_the_flag() {
        check(0).unwrap();

        let e = check(3).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Numerical);
        assert_eq!(e.message, "syevd did not converge (info = 3)");
        assert_eq!(e.context, ["syevd check@cpu0"]);

        let e = check(-9).unwrap_err();
        assert_eq!(e.kind, ErrorKind::GpuRuntime);
        assert_eq!(e.message, "syevd rejected argument 9 (info = -9)");
    }
}
