//! The symmetric (Löwdin) orthogonalizer `X = S^(-1/2) = U Λ^(-1/2) Uᵀ` on
//! the GPU, as tasks of a [`Dag`], where `S = U Λ Uᵀ` is the symmetric
//! eigendecomposition. The C++ `src/recipes/orthogonalize` is the reference
//! design, its README the reasoning.
//!
//! [`add_orthogonalize`] adds, for one symmetric matrix row S:
//!
//! - **`syevd`** and **`syevd check`**, [`add_diagonalize`]'s two tasks:
//!   S copied into U, then U and the ascending eigenvalues w, with S left
//!   as it was.
//! - **`inverse root`**, on the worker's stream: w copied into a scratch
//!   row Λ, mapped in place to `Λ^(-1/2)` by `libfb_recipes_kernels.so`'s
//!   [`fb_inverse_sqrt`] (an eigenvalue at or below the floor becomes 0,
//!   which drops its mode), then `M = U diag(Λ^(-1/2))` by a BLAS `dgmm`
//!   into scratch M and `X = M Uᵀ` by a `gemm` into the output row X, both
//!   on the worker's handle, through [`fb_recipes_blas::dgmm`] and
//!   [`fb_recipes_blas::gemm`].
//!
//! **Conditioning.** As in the C++: the floor test comes before the square
//! root, so X is free of NaN and Inf for a zero or negative eigenvalue, and
//! `X S X = I` only when every eigenvalue clears the floor; for a singular
//! S, `X S X` is the projector onto the kept modes. X stays square: this is
//! not canonical orthogonalization.
//!
//! | C++ | Here |
//! |---|---|
//! | `orthogonalize<T>(res, n, s, x, floor)` | [`add_orthogonalize`]`(dag, worker, name, n, lwork, s, floor)`, X its output row |
//! | `kOrthogonalizeEigenvalueFloor<T>`, 1e-10 | [`ORTHOGONALIZE_EIGENVALUE_FLOOR`] |
//! | S copied into U, then `diagonalize(res, n, U, w)`, which syncs to read syevd's flag | [`add_diagonalize`]'s `syevd` (the copy into U its first work) and `syevd check` |
//! | U and the eigenvalues, pool buffers of the call's own; M, the device workspace's | U and w, `syevd`'s output rows, read by `inverse root`; Λ and M, its scratch rows |
//! | `inverse_sqrt<T>(stream, n, w, floor)` over w itself | Λ, a copy of w, through [`fb_inverse_sqrt`]: a task writes only its own rows |
//! | `gmmwich<WWRBLAS_OP_T>(res.blas(), n, U, Λ, M, X)`: a `dgmm` and a `gemm` | the same two calls, on the worker's handle, [`DeviceStream::blas`] |
//! | `orthogonalize_footprint<T>(n, syevd_lwork)` | the layout, read off by the planner, `lwork` handed in ([`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork)) |
//! | the extreme eigenvalues copied back and cond(S) logged, a second sync | not ported: a task has no host wait inside it, and nothing reads the log yet |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | `float` and `double` | `f64` only, until a caller needs `f32` |
//! | `n <= 0`: an empty success | refused at planning: `n` is at least 1 |
//! | nothing: the C++ tree compiles its kernels into the binary | [`fb_recipes_sys::check_library`] first, at planning |
//! | `orthonormalize`, `orthonormalize_footprint` | not ported: nothing calls them yet |

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::gpu_status_at;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_recipes_blas::{N, Right, T, dgmm, gemm};
use fb_recipes_diagonalize::add_diagonalize;
use fb_recipes_sys::fb_inverse_sqrt;

/// The default eigenvalue floor, the C++ `kOrthogonalizeEigenvalueFloor`: an
/// eigenvalue at or below it contributes 0 to X rather than `λ^(-1/2)`.
pub const ORTHOGONALIZE_EIGENVALUE_FLOOR: f64 = 1e-10;

/// Add `X = S^(-1/2)` of the symmetric `n` × `n` matrix `s` to `dag` on GPU
/// worker `worker`, and return X: an output row of `n²` doubles,
/// column-major, named `name`.
///
/// `s` is an earlier task's output of `n²` doubles, column-major; only its
/// lower triangle is read, and it is left as it was. `lwork` is
/// [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) at `n` on the device
/// the plan will run on. An eigenvalue at or below `floor`
/// ([`ORTHOGONALIZE_EIGENVALUE_FLOOR`] unless the caller has its own) drops
/// its mode. The tasks added are [`add_diagonalize`]'s, U and w named
/// `"{name} U"` and `"{name} w"`, then `inverse root` (stream), its scratch
/// `"{name} Lambda"` and `"{name} M"`.
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
/// As [`add_diagonalize`]: if `n` is 0, or `n²` or `lwork` does not fit an
/// `i32`, or `s` is a row of another worker's table.
#[track_caller]
pub fn add_orthogonalize(
    dag: &mut Dag<'_>,
    worker: usize,
    name: &str,
    n: usize,
    lwork: usize,
    s: RowId<Device, f64>,
    floor: f64,
) -> Result<RowId<Device, f64>> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    let (u, w) = add_diagonalize(dag, worker, name, n, lwork, s);
    Ok(dag
        .add(
            "inverse root",
            worker,
            InverseRoot {
                u,
                w,
                n,
                floor,
                name: name.to_owned(),
            },
        )
        .x)
}

/// GPU: w copied into Λ, Λ mapped to `Λ^(-1/2)`, then `M = U diag(Λ)` and
/// `X = M Uᵀ`.
struct InverseRoot {
    u: RowId<Device, f64>,
    w: RowId<Device, f64>,
    n: usize,
    floor: f64,
    /// X's row name, and the scratch rows' prefix.
    name: String,
}

#[derive(Clone, Copy)]
struct InverseRootRows {
    lambda: RowId<Device, f64>,
    m: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

impl Task for InverseRoot {
    type Rows = InverseRootRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        _: &mut Layout<'_, Pinned>,
    ) -> InverseRootRows {
        device.read(self.u);
        device.read(self.w);
        InverseRootRows {
            lambda: device.scratch(&format!("{} Lambda", self.name), self.n),
            m: device.scratch(&format!("{} M", self.name), self.n * self.n),
            x: device.output(&self.name, self.n * self.n),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &InverseRootRows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let stream = worker.stream();
        let lambda = device.take_mut(rows.lambda);
        memcpy_d2d(&lambda, device.get(self.w), worker)?;
        // add_diagonalize checked that n² fits an i32, so n does too.
        let n = self.n as i32;
        // SAFETY: Λ is a device row of n doubles, on the task's stream,
        // after the copy into it.
        let status = unsafe { fb_inverse_sqrt(stream, n as u32, lambda.as_mut_ptr(), self.floor) };
        fb_kernel_abi::launch_status("fb_inverse_sqrt", status, |code, at| {
            Some(gpu_status_at(wwr_status_from_code(code)?, at))
        })?;

        // The worker's handle, bound to `stream` when the backend was made.
        let handle = worker.blas();
        let (u, mut m, mut x) = (
            device.get(self.u),
            device.take_mut(rows.m),
            device.take_mut(rows.x),
        );
        // M = U diag(Λ): column j of U scaled by Λ[j].
        dgmm::<Right, f64>(handle, n, n, &u, &lambda, &mut m)?;
        // X = M Uᵀ, symmetric by construction.
        gemm::<N, T, f64>(handle, n, n, n, 1.0, &m, &u, 0.0, &mut x)
    }
}
