//! The Roothaan step on a stream: F transformed by X, diagonalized, and its
//! eigenvectors transformed back.

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_recipes_diagonalize::add_diagonalize;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

/// Add the solution of `F C = S C ε` through `x = S^(-1/2)` to `dag` on GPU
/// worker `worker`, and return `(ε, C)`: an output row of the `n` orbital
/// energies in ascending order, named `"F' w"`, and an output row of `n²`
/// doubles, column-major, column `j` the orbital of `ε[j]`, named `"C"`,
/// S-orthonormal (`Cᵀ S C = I`).
///
/// `f` and `x` are earlier tasks' outputs of `n²` doubles, column-major and
/// symmetric up to rounding, both left as they were; X is
/// [`add_orthogonalize`](fb_recipes_orthogonalize::add_orthogonalize)'s.
/// `lwork` is [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) at `n` on
/// the device the plan will run on. The tasks added are `fock transform`
/// (stream: `F' = X F X` into the output row `"F'"`, through the scratch
/// row `"F X"`), [`add_diagonalize`]'s `syevd` and `syevd check` on F' (C'
/// its row `"F' U"`), and `back transform` (stream: `C = X C'`).
///
/// Requires, checked in debug builds: `n` is at least 1.
///
/// # Panics
/// As [`add_diagonalize`]: if `n²` or `lwork` does not fit an `i32`; or if
/// `f` or `x` is a row of another worker's table ([`Dag::add`]).
#[debug_requires(n > 0, "a matrix of at least one row")]
#[track_caller]
pub fn add_solve_fock(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    lwork: usize,
    f: RowId<Device, f64>,
    x: RowId<Device, f64>,
) -> (RowId<Device, f64>, RowId<Device, f64>) {
    assert!(
        n.checked_mul(n).is_some_and(|n2| i32::try_from(n2).is_ok()),
        "add_solve_fock: n = {n}: n² does not fit BLAS's i32"
    );
    let f_prime = dag
        .add("fock transform", worker, FockTransform { f, x, n })
        .f_prime;
    let (c_prime, eps) = add_diagonalize(dag, worker, "F'", n, lwork, f_prime);
    let c = dag.add("back transform", worker, BackTransform { x, c_prime, n });
    (eps, c)
}

/// `c = op_a(a) b` for `n` × `n` column-major device matrices, on the
/// worker's BLAS handle.
///
/// # Safety
/// `a`, `b` and `c` are device rows of `n²` doubles, `c` apart from both,
/// and `handle` is bound to the task's stream.
unsafe fn gemm(
    handle: wwr_sys::wwr_blas_handle,
    op_a: wwr_sys::wwr_blas_operation,
    n: i32,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
) -> Status {
    let (one, zero) = (1.0_f64, 0.0_f64);
    // SAFETY: the caller's; the scalars are host memory, the handle's
    // default pointer mode.
    gpu_status(unsafe {
        wwr_sys::wwr_blas_dgemm(
            handle,
            op_a,
            WWR_BLAS_OP_N,
            n,
            n,
            n,
            &one,
            a,
            n,
            b,
            n,
            &zero,
            c,
            n,
        )
    })
}

/// GPU: `F X` into scratch, then `F' = Xᵀ (F X)`.
struct FockTransform {
    f: RowId<Device, f64>,
    x: RowId<Device, f64>,
    n: usize,
}

#[derive(Clone, Copy)]
struct FockTransformRows {
    fx: RowId<Device, f64>,
    f_prime: RowId<Device, f64>,
}

impl Task for FockTransform {
    type Rows = FockTransformRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        _: &mut Layout<'_, Pinned>,
    ) -> FockTransformRows {
        device.read(self.f);
        device.read(self.x);
        FockTransformRows {
            fx: device.scratch("F X", self.n * self.n),
            f_prime: device.output("F'", self.n * self.n),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &FockTransformRows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        // The worker's handle, bound to its stream when the backend was made.
        let handle = worker.blas().get();
        // add_solve_fock checked that n² fits an i32, so n does too.
        let n = self.n as i32;
        let (f, x) = (device.get(self.f), device.get(self.x));
        let (fx, f_prime) = (device.take_mut(rows.fx), device.take_mut(rows.f_prime));
        // SAFETY: four n × n device rows, the outputs this task's own; the
        // second gemm follows the first on the handle's stream.
        unsafe {
            gemm(
                handle,
                WWR_BLAS_OP_N,
                n,
                f.as_ptr(),
                x.as_ptr(),
                fx.as_mut_ptr(),
            )?;
            gemm(
                handle,
                WWR_BLAS_OP_T,
                n,
                x.as_ptr(),
                fx.as_ptr(),
                f_prime.as_mut_ptr(),
            )
        }
    }
}

/// GPU: `C = X C'`.
struct BackTransform {
    x: RowId<Device, f64>,
    c_prime: RowId<Device, f64>,
    n: usize,
}

impl Task for BackTransform {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.x);
        device.read(self.c_prime);
        device.output("C", self.n * self.n)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        c: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let (x, c_prime, c) = (
            device.get(self.x),
            device.get(self.c_prime),
            device.take_mut(*c),
        );
        // SAFETY: three n × n device rows, C this task's own; the handle is
        // the worker's, bound to its stream.
        unsafe {
            gemm(
                worker.blas().get(),
                WWR_BLAS_OP_N,
                self.n as i32,
                x.as_ptr(),
                c_prime.as_ptr(),
                c.as_mut_ptr(),
            )
        }
    }
}
