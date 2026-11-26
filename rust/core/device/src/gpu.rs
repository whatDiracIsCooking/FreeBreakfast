//! One GPU's library resources, [`GpuResources`]: its stream, the BLAS and
//! dense-solver handles made on it, the handle workspace, the checked BLAS
//! and solver calls, and the tree's `trsm` and Cholesky recipes over them.

use std::ffi::c_void;
use std::panic::Location;
use std::sync::Mutex;

use contracts::requires;
use fb_core_error_handling::{Error, Result, Status, check_or_abort};
use fb_core_log::{eq, fits_i32, le};
use fb_kernel_abi::LaunchStatus;
use fb_recipes_sys::{FB_CHOLESKY_TILE, FB_TRSM_TILE, fb_cholesky_diagonal, fb_trsm_diagonal_f64};
use wwr_sys::{
    wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_handle, wwr_blas_operation,
    wwr_blas_side_mode, wwr_blas_status, wwr_solver_eig_mode, wwr_solver_fill_mode,
    wwr_solver_handle, wwr_solver_status,
};

use crate::slice::{DeviceSlice, DeviceSliceMut, part, part_mut, split_at_mut};
use crate::stream::Stream;
use crate::{ScopedDevice, log_destroy};

/// The handle workspace's size: the device memory every BLAS call on a GPU
/// draws on, bound once ([`GpuResources::bind_workspace`]).
pub const HANDLE_WORKSPACE_BYTES: usize = 128 << 20;

/// The order of [`GpuResources::cholesky`]'s tile, whose scratch is its
/// square.
pub const CHOLESKY_TILE: usize = FB_CHOLESKY_TILE as usize;

const OP_N: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
const OP_T: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
const EIG_VECTORS: wwr_solver_eig_mode = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
const EIG_LOWER: wwr_solver_fill_mode = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;

/// A BLAS call's status as a [`Status`] naming `what`.
#[track_caller]
fn blas_status(what: &str, code: wwr_blas_status) -> Status {
    match code {
        wwr_blas_status::WWR_BLAS_STATUS_SUCCESS => Ok(()),
        _ => Err(Error::gpu(format!("{what}: {code:?}"))),
    }
}

/// A dense-solver call's status as a [`Status`] naming `what`.
#[track_caller]
fn solver_status(what: &str, code: wwr_solver_status) -> Status {
    match code {
        wwr_solver_status::WWR_SOLVER_STATUS_SUCCESS => Ok(()),
        _ => Err(Error::gpu(format!("{what}: {code:?}"))),
    }
}

/// A recipe kernel's launch status as a [`Status`] naming `what`.
#[track_caller]
fn launched(what: &str, status: LaunchStatus) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, _| {
        Some(Err(Error::gpu(format!(
            "runtime error {code} at the launch"
        ))))
    })
}

/// The elements a column-major `rows × cols` matrix at leading dimension
/// `ld` spans, 0 for an empty one, widened so two `i32`s cannot overflow.
fn extent(rows: i32, cols: i32, ld: i32) -> u64 {
    if rows <= 0 || cols <= 0 {
        return 0;
    }
    ld as u64 * (cols as u64 - 1) + rows as u64
}

/// The order of a triangular solve's A: m on the left, n on the right.
fn order(side: wwr_blas_side_mode, m: i32, n: i32) -> i32 {
    match side {
        wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT => m,
        wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT => n,
    }
}

/// Panic unless a column-major `rows × cols` operand at leading dimension
/// `ld` fits its slice of `len` elements, with `ld` at least `max(1, rows)`.
#[track_caller]
fn check_matrix(routine: &str, name: &str, len: usize, rows: i32, cols: i32, ld: i32) {
    assert!(
        rows >= 0 && cols >= 0 && ld >= rows.max(1),
        "{routine}: {name} is {rows} × {cols} at leading dimension {ld}, which is no matrix"
    );
    let need = extent(rows, cols, ld);
    assert!(
        need <= len as u64,
        "{routine}: {name} is {rows} × {cols} at leading dimension {ld}, {need} elements, \
         but its slice holds {len}"
    );
}

/// Panic unless `batch` operands of [`check_matrix`]'s shape, the `i`-th
/// `i · stride` elements past the first, fit `len` elements; `stride` is
/// non-negative, and at least one operand's extent when `apart`, an operand
/// the call writes.
#[allow(
    clippy::too_many_arguments,
    reason = "one operand's shape and its batch"
)]
#[track_caller]
fn check_batch(
    routine: &str,
    name: &str,
    len: usize,
    rows: i32,
    cols: i32,
    ld: i32,
    stride: i64,
    batch: i32,
    apart: bool,
) {
    assert!(
        rows >= 0 && cols >= 0 && ld >= rows.max(1) && batch >= 0,
        "{routine}: {name} is {batch} × ({rows} × {cols}) at leading dimension {ld}, which is no batch"
    );
    let one = extent(rows, cols, ld);
    if batch == 0 || one == 0 {
        return;
    }
    let stride_ok = stride >= 0 && (!apart || stride as u64 >= one);
    assert!(
        stride_ok,
        "{routine}: {name}'s stride {stride} is negative or, for an operand written, under \
         one operand's {one} elements"
    );
    let need = (batch as u64 - 1)
        .checked_mul(stride as u64)
        .and_then(|s| s.checked_add(one));
    assert!(
        need.is_some_and(|need| need <= len as u64),
        "{routine}: {name} is {batch} operands at stride {stride}, {need:?} elements, but its \
         slice holds {len}"
    );
}

/// Panic unless a solver call's work slice of `len` elements holds the
/// size query's `need`: the library does not check it, and writes past a
/// short one.
#[track_caller]
fn check_work(routine: &str, n: usize, len: usize, need: usize) {
    assert!(
        len >= need,
        "{routine} at n = {n}: the work buffer holds {len} elements, short of the size \
         query's {need}: the call is missing from the step's scratch"
    );
}

/// One GPU's library resources: its stream, a BLAS and a dense-solver
/// handle made on that stream, and the handle workspace. Made inside a
/// [`Mutex`], and reached only through its guard, which a step holds for
/// its whole `run`; the step wrapper syncs the stream before the guard
/// drops, so no handle is ever handed on with work in flight
/// (`docs/gpu_layer.md`, "The v0.0.4 runtime").
///
/// - **A handle keeps its stream.** `SetStream` is called once, in
///   [`GpuResources::new`], and the workspace is bound once
///   ([`GpuResources::bind_workspace`]), so no handle manages memory of its
///   own.
/// - **A call takes slices and checks them before it enqueues.** Each BLAS
///   and solver method holds its operands against the call's own extents,
///   and a solver call its work slice against the size query repeated at
///   the same arguments, panicking at the caller's line with nothing
///   enqueued.
/// - **The recipes take the guard.** [`GpuResources::trsm`] and
///   [`GpuResources::cholesky`] put their diagonal kernels and their BLAS
///   calls on the one stream the guard has. A vendor `trsm`, `potrs` or
///   `getrs` is not bound: each takes workspace growing with its
///   right-hand sides, past the fixed one (`docs/gpu_layer.md`,
///   "Layering").
///
/// Every method makes the GPU's device current for the call, as a
/// [`Stream`]'s do.
///
/// ```no_run
/// use fb_core_device::GpuResources;
///
/// let gpu0 = GpuResources::new(0, 0);
/// let mut held = gpu0.lock().unwrap();
/// let lwork = held.syevd_lwork(64).unwrap();
/// assert!(lwork >= 1);
/// ```
///
/// What the guard lends ends with it: nothing reached through it outlives
/// the lock.
///
/// ```compile_fail,E0597
/// use fb_core_device::{GpuResources, Stream};
///
/// let gpu0 = GpuResources::new(0, 0);
/// let stream: &Stream = {
///     let held = gpu0.lock().unwrap();
///     held.stream()
/// };
/// stream.sync().unwrap();
/// ```
pub struct GpuResources {
    gpu: usize,
    blas: wwr_blas_handle,
    solver: wwr_solver_handle,
    workspace: Option<Box<dyn DeviceSliceMut<u8> + Send>>,
    stream: Stream,
}

// SAFETY: the handles are runtime objects not tied to the thread that made
// them; every call names its handle and makes the device current first, and
// `&mut self` on every call that uses one, through the guard, keeps them to
// one thread at a time. The workspace is `Send` by its bound. Not `Sync`:
// the raw handles keep it out.
unsafe impl Send for GpuResources {}

// The above, checked at every build: what a `Mutex` needs to be shared.
const _: () = {
    const fn shared_behind_a_mutex<T: Send + Sync>() {}
    shared_behind_a_mutex::<Mutex<GpuResources>>();
};

impl GpuResources {
    /// GPU `gpu` of the run, on device `physical`, inside its [`Mutex`]: a
    /// stream, and a BLAS and a dense-solver handle given that stream once,
    /// for their life. Aborts at the caller's line if the runtime or a
    /// library refuses, as every create does.
    #[track_caller]
    pub fn new(gpu: usize, physical: i32) -> Mutex<GpuResources> {
        let stream = Stream::new(physical);
        let _current = ScopedDevice::new(physical);
        let mut blas: wwr_blas_handle = std::ptr::null_mut();
        // SAFETY: writes one handle through a valid local, with the device
        // current.
        check_or_abort(blas_status("blas create", unsafe {
            wwr_sys::wwr_blas_create(&mut blas)
        }));
        // SAFETY: a live handle and a live stream of its device.
        check_or_abort(blas_status("blas set_stream", unsafe {
            wwr_sys::wwr_blas_set_stream(blas, stream.raw())
        }));
        let mut solver: wwr_solver_handle = std::ptr::null_mut();
        // SAFETY: as the BLAS create.
        check_or_abort(solver_status("solver create", unsafe {
            wwr_sys::wwr_solver_dn_create(&mut solver)
        }));
        // SAFETY: as the BLAS set_stream.
        check_or_abort(solver_status("solver set_stream", unsafe {
            wwr_sys::wwr_solver_dn_set_stream(solver, stream.raw())
        }));
        Mutex::new(GpuResources {
            gpu,
            blas,
            solver,
            workspace: None,
            stream,
        })
    }

    /// The GPU's index in the run.
    pub fn gpu(&self) -> usize {
        self.gpu
    }

    /// The device it is on.
    pub fn physical(&self) -> i32 {
        self.stream.physical()
    }

    /// The GPU's stream: the library calls go there, and the holder's
    /// kernels and copies when it wants one order with no event.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    fn current(&self) -> ScopedDevice {
        ScopedDevice::new(self.stream.physical())
    }

    /// Hand the BLAS handle its workspace, for the handle's life: the
    /// budget's [`HANDLE_WORKSPACE_BYTES`] on this GPU, kept here until the
    /// handle is destroyed, since the library keeps the pointer for every
    /// later call. Unbound, the library takes memory of its own, outside
    /// the budget.
    ///
    /// # Errors
    /// The library's refusal; the workspace is then dropped unbound.
    #[requires(self.workspace.is_none(), "the workspace is bound once")]
    #[requires(
        eq!(workspace.len(), HANDLE_WORKSPACE_BYTES),
        "the workspace is HANDLE_WORKSPACE_BYTES"
    )]
    #[track_caller]
    pub fn bind_workspace(
        &mut self,
        mut workspace: impl DeviceSliceMut<u8> + Send + 'static,
    ) -> Status {
        let _current = self.current();
        let ptr = workspace.as_mut_ptr().cast::<c_void>();
        // SAFETY: `workspace` is its length of this GPU's device memory with
        // no other user for as long as it is borrowed, and it is kept below
        // until the drop, which syncs the stream and destroys the handle
        // before releasing it; nothing is in flight on the handle, which
        // `&mut self` holds.
        blas_status("blas set_workspace", unsafe {
            wwr_sys::wwr_blas_set_workspace(self.blas, ptr, workspace.len())
        })?;
        self.workspace = Some(Box::new(workspace));
        Ok(())
    }

    /// `C = alpha op_a(A) op_b(B) + beta C`, column-major, `op_a(A)` m × k,
    /// `op_b(B)` k × n, C m × n, each at its leading dimension; A is stored
    /// m × k under `WWR_BLAS_OP_N` and k × m transposed, B likewise. With
    /// `beta` zero, C need not be initialised.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand is no matrix (a negative dimension, or a leading
    /// dimension under its rows or 1) or its slice is shorter than its
    /// extent, naming it; before anything is enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemm(
        &mut self,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        b: &impl DeviceSlice<f64>,
        ldb: i32,
        beta: f64,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
    ) -> Status {
        let stored = |op, rows, cols| {
            if op == OP_N {
                (rows, cols)
            } else {
                (cols, rows)
            }
        };
        let (ra, ca) = stored(op_a, m, k);
        let (rb, cb) = stored(op_b, k, n);
        check_matrix("gemm", "A", a.len(), ra, ca, lda);
        check_matrix("gemm", "B", b.len(), rb, cb, ldb);
        check_matrix("gemm", "C", c.len(), m, n, ldc);
        let c = c.as_mut_ptr();
        // SAFETY: each operand covers its extent, checked above, and stays
        // so until the enqueued work finishes (the slices' promise); C is
        // its memory's only user, so apart from A and B.
        unsafe {
            self.gemm_raw(
                op_a,
                op_b,
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c,
                ldc,
            )
        }
    }

    /// [`Self::gemm`] over pointers, for the recipes' blocks of one matrix.
    ///
    /// # Safety
    /// Each pointer is device memory of this GPU covering its operand at
    /// its leading dimension until the work enqueued here finishes, and no
    /// element of C is an element of A or B.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    unsafe fn gemm_raw(
        &mut self,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    ) -> Status {
        let _current = self.current();
        // SAFETY: the caller's; the handle is live and on the stream, the
        // scalars host values in its default pointer mode.
        blas_status("gemm", unsafe {
            wwr_sys::wwr_blas_dgemm(
                self.blas, op_a, op_b, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
            )
        })
    }

    /// `batch` gemms at fixed strides: `C_i = alpha op_a(A_i) op_b(B_i) +
    /// beta C_i`, the `i`-th operand of each `i · stride` elements past its
    /// first, shaped as [`Self::gemm`]'s.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand's batch is no batch (as [`Self::gemm`]'s, or a negative
    /// count), a stride is negative, C's is under one C, or a slice is
    /// shorter than its batch's extent, naming it; before anything is
    /// enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemm_strided_batched(
        &mut self,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        stride_a: i64,
        b: &impl DeviceSlice<f64>,
        ldb: i32,
        stride_b: i64,
        beta: f64,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
        stride_c: i64,
        batch: i32,
    ) -> Status {
        let stored = |op, rows, cols| {
            if op == OP_N {
                (rows, cols)
            } else {
                (cols, rows)
            }
        };
        let (ra, ca) = stored(op_a, m, k);
        let (rb, cb) = stored(op_b, k, n);
        let routine = "gemm_strided_batched";
        check_batch(routine, "A", a.len(), ra, ca, lda, stride_a, batch, false);
        check_batch(routine, "B", b.len(), rb, cb, ldb, stride_b, batch, false);
        check_batch(routine, "C", c.len(), m, n, ldc, stride_c, batch, true);
        let _current = self.current();
        let c = c.as_mut_ptr();
        // SAFETY: each batch covers its extent, checked above, the C
        // operands apart from one another by their stride, and stays so
        // until the enqueued work finishes; C is its memory's only user.
        blas_status(routine, unsafe {
            wwr_sys::wwr_blas_dgemm_strided_batched(
                self.blas,
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a.as_ptr(),
                lda,
                stride_a,
                b.as_ptr(),
                ldb,
                stride_b,
                &beta,
                c,
                ldc,
                stride_c,
                batch,
            )
        })
    }

    /// `y = alpha op(A) x + beta y`, A m × n at `lda`, `x` and `y`
    /// contiguous: n and m long under `WWR_BLAS_OP_N`, m and n transposed.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand is no matrix or its slice is shorter than its extent,
    /// naming it; before anything is enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemv(
        &mut self,
        op: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        x: &impl DeviceSlice<f64>,
        beta: f64,
        y: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let (nx, ny) = if op == OP_N { (n, m) } else { (m, n) };
        check_matrix("gemv", "A", a.len(), m, n, lda);
        check_matrix("gemv", "x", x.len(), nx, 1, nx.max(1));
        check_matrix("gemv", "y", y.len(), ny, 1, ny.max(1));
        let _current = self.current();
        let y = y.as_mut_ptr();
        // SAFETY: as `gemm`'s, the vectors at stride 1.
        blas_status("gemv", unsafe {
            wwr_sys::wwr_blas_dgemv(
                self.blas,
                op,
                m,
                n,
                &alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                1,
                &beta,
                y,
                1,
            )
        })
    }

    /// `C = alpha op_a(A) + beta op_b(B)`, C m × n at `ldc`; A is stored
    /// m × n under `WWR_BLAS_OP_N` and n × m transposed, B likewise.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand is no matrix or its slice is shorter than its extent,
    /// naming it; before anything is enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn geam(
        &mut self,
        op_a: wwr_blas_operation,
        op_b: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        beta: f64,
        b: &impl DeviceSlice<f64>,
        ldb: i32,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
    ) -> Status {
        let stored = |op, rows, cols| {
            if op == OP_N {
                (rows, cols)
            } else {
                (cols, rows)
            }
        };
        let (ra, ca) = stored(op_a, m, n);
        let (rb, cb) = stored(op_b, m, n);
        check_matrix("geam", "A", a.len(), ra, ca, lda);
        check_matrix("geam", "B", b.len(), rb, cb, ldb);
        check_matrix("geam", "C", c.len(), m, n, ldc);
        let _current = self.current();
        let c = c.as_mut_ptr();
        // SAFETY: as `gemm`'s.
        blas_status("geam", unsafe {
            wwr_sys::wwr_blas_dgeam(
                self.blas,
                op_a,
                op_b,
                m,
                n,
                &alpha,
                a.as_ptr(),
                lda,
                &beta,
                b.as_ptr(),
                ldb,
                c,
                ldc,
            )
        })
    }

    /// `C = diag(x) A` (`side` `WWR_BLAS_SIDE_LEFT`) or `C = A diag(x)`
    /// (`_RIGHT`), A and C m × n at their leading dimensions, `x`
    /// contiguous, m or n long.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand is no matrix or its slice is shorter than its extent,
    /// naming it; before anything is enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn dgmm(
        &mut self,
        side: wwr_blas_side_mode,
        m: i32,
        n: i32,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        x: &impl DeviceSlice<f64>,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
    ) -> Status {
        let nx = order(side, m, n);
        check_matrix("dgmm", "A", a.len(), m, n, lda);
        check_matrix("dgmm", "x", x.len(), nx, 1, nx.max(1));
        check_matrix("dgmm", "C", c.len(), m, n, ldc);
        let _current = self.current();
        let c = c.as_mut_ptr();
        // SAFETY: as `gemm`'s, `x` at stride 1.
        blas_status("dgmm", unsafe {
            wwr_sys::wwr_blas_ddgmm(
                self.blas,
                side,
                m,
                n,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                1,
                c,
                ldc,
            )
        })
    }

    /// `C = alpha op(A) op(A)ᵀ + beta C` in C's `fill` triangle, C n × n
    /// at `ldc`; A is stored n × k under `WWR_BLAS_OP_N` and k × n
    /// transposed. The other triangle of C is left as it was.
    ///
    /// # Errors
    /// The library's refusal.
    ///
    /// # Panics
    /// If an operand is no matrix or its slice is shorter than its extent,
    /// naming it; before anything is enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn syrk(
        &mut self,
        fill: wwr_blas_fill_mode,
        op: wwr_blas_operation,
        n: i32,
        k: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        beta: f64,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
    ) -> Status {
        let (ra, ca) = if op == OP_N { (n, k) } else { (k, n) };
        check_matrix("syrk", "A", a.len(), ra, ca, lda);
        check_matrix("syrk", "C", c.len(), n, n, ldc);
        let _current = self.current();
        let c = c.as_mut_ptr();
        // SAFETY: as `gemm`'s.
        blas_status("syrk", unsafe {
            wwr_sys::wwr_blas_dsyrk(
                self.blas,
                fill,
                op,
                n,
                k,
                &alpha,
                a.as_ptr(),
                lda,
                &beta,
                c,
                ldc,
            )
        })
    }

    /// [`Self::syevd`]'s work at order `n`, in doubles: the solver's size
    /// query on this GPU's handle, what a step declares as scratch for each
    /// eigensolve it makes. Never 0: a query of 0 still gets a one-element
    /// slice, since a null work buffer puts hipSOLVER's handle in
    /// library-managed mode.
    ///
    /// # Errors
    /// The query's refusal.
    #[requires(fits_i32!(n.checked_mul(n)), "n² fits the solver's i32")]
    #[track_caller]
    pub fn syevd_lwork(&self, n: usize) -> Result<usize> {
        let _current = self.current();
        let n = n as i32;
        let mut lwork = 0;
        // SAFETY: a size query on a live handle: it reads the dimensions and
        // neither array, and writes one integer through a valid local.
        solver_status("syevd_lwork", unsafe {
            wwr_sys::wwr_solver_dn_dsyevd_buffer_size(
                self.solver,
                EIG_VECTORS,
                EIG_LOWER,
                n,
                std::ptr::null(),
                n.max(1),
                std::ptr::null(),
                &mut lwork,
            )
        })?;
        Ok(usize::try_from(lwork).unwrap_or(0).max(1))
    }

    /// `A = V diag(w) Vᵀ` in place on the GPU's stream, A n × n at leading
    /// dimension n, its lower triangle read: the eigenvalues into `w`
    /// ascending, the eigenvectors over A, column `j` that of `w[j]`, and
    /// the solver's flag into `info[0]` on the device, 0 on success. `work`
    /// is a slice of the caller's scratch, declared from
    /// [`Self::syevd_lwork`] at the same `n`, which the call asks again.
    ///
    /// # Errors
    /// The query's or the solver's refusal.
    ///
    /// # Panics
    /// If `work` is shorter than the size query at `n`, at the caller's
    /// line and before anything is enqueued.
    #[requires(
        fits_i32!(n.checked_mul(n)) && le!(n * n, a.len()),
        "a holds the n × n matrix, n² within the solver's i32"
    )]
    #[requires(le!(n, w.len()), "w holds the n eigenvalues")]
    #[requires(le!(1, info.len()), "info holds the flag")]
    #[track_caller]
    pub fn syevd(
        &mut self,
        n: usize,
        a: &mut impl DeviceSliceMut<f64>,
        w: &mut impl DeviceSliceMut<f64>,
        work: &mut impl DeviceSliceMut<f64>,
        info: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        let need = self.syevd_lwork(n)?;
        check_work("syevd", n, work.len(), need);
        let _current = self.current();
        let lwork = i32::try_from(need).unwrap_or(i32::MAX);
        let n = n as i32;
        let (a, w, work, info) = (
            a.as_mut_ptr(),
            w.as_mut_ptr(),
            work.as_mut_ptr(),
            info.as_mut_ptr(),
        );
        // SAFETY: A, w and info cover their extents by the contracts, the
        // work at least the query's lwork, checked above, each until the
        // enqueued work finishes and each its memory's only user, so the
        // four are apart; the handle is live and on the stream.
        solver_status("syevd", unsafe {
            wwr_sys::wwr_solver_dn_dsyevd(
                self.solver,
                EIG_VECTORS,
                EIG_LOWER,
                n,
                a,
                n.max(1),
                w,
                work,
                lwork,
                info,
            )
        })
    }

    /// Each element of `data` replaced by `1 / sqrt(x)` in place on the
    /// GPU's stream, or by 0 at or below `floor`: the recipes'
    /// `fb_inverse_sqrt` kernel, the floor tested before the root, so no
    /// Inf comes out and no NaN that was not one already. An empty `data`
    /// enqueues nothing.
    ///
    /// # Errors
    /// The launch's failure; a fault inside the kernel surfaces at the next
    /// sync.
    #[requires(le!(data.len(), u32::MAX as usize), "data's length fits the launch's u32")]
    #[track_caller]
    pub fn inverse_sqrt(&mut self, data: &mut impl DeviceSliceMut<f64>, floor: f64) -> Status {
        let _current = self.current();
        let count = data.len() as u32;
        let data = data.as_mut_ptr();
        // SAFETY: `data` is `count` doubles of device memory with no other
        // user until the work enqueued here finishes (DeviceSliceMut's
        // promise), on the guard's live stream; the floor's scale is null,
        // which the kernel reads as none.
        let status = unsafe {
            fb_recipes_sys::fb_inverse_sqrt(self.stream.raw(), count, data, floor, std::ptr::null())
        };
        launched("fb_inverse_sqrt", status)
    }

    /// BLAS's triangular solve on the GPU's stream, the tree's own and never
    /// the library's (`docs/gpu_layer.md`, "Layering"): `B ← alpha
    /// op(A)⁻¹ B` (`side` `WWR_BLAS_SIDE_LEFT`) or `B ← alpha B op(A)⁻¹`
    /// (`_RIGHT`), in place. B is m × n at `ldb`; A is triangular of order
    /// k, m on the left and n on the right, at `lda`, its `fill` triangle
    /// alone read, and its diagonal taken as 1 and not read under
    /// `WWR_BLAS_DIAG_UNIT`. With `alpha` zero B becomes zero and neither A
    /// nor B is read; a singular A is not detected, and gives Inf or NaN.
    ///
    /// Blocked, [`FB_TRSM_TILE`] rows of `op(A)` (on the right, of its
    /// transpose) at a time, from the top for a lower one and from the
    /// bottom for an upper one: each diagonal block by the recipes'
    /// shared-memory kernel, then one gemm that takes that block's solution
    /// out of the rows still to come, `alpha` applied once, by the first of
    /// each. Kernels and gemms land on the one stream the guard has, and
    /// take no memory beyond the handle workspace. The C++
    /// `src/recipes/trsm` is the reference design.
    ///
    /// The guard is the only way in: a stream of a step's own is no
    /// receiver.
    ///
    /// ```compile_fail,E0308
    /// use fb_core_device::{DeviceSlice, DeviceSliceMut, GpuResources, Stream};
    /// use wwr_sys::{
    ///     wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode,
    /// };
    ///
    /// fn fit(
    ///     own: &mut Stream,
    ///     l: &impl DeviceSlice<f64>,
    ///     b: &mut impl DeviceSliceMut<f64>,
    ///     n: i32,
    /// ) {
    ///     GpuResources::trsm(
    ///         own,
    ///         wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
    ///         wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
    ///         wwr_blas_operation::WWR_BLAS_OP_N,
    ///         wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
    ///         n,
    ///         n,
    ///         1.0,
    ///         l,
    ///         n,
    ///         b,
    ///         n,
    ///     )
    ///     .unwrap();
    /// }
    /// ```
    ///
    /// # Errors
    /// The first launch's or gemm's failure; what was enqueued before it
    /// stays enqueued.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[requires(le!(0, m) && le!(0, n), "m and n are not negative")]
    #[requires(
        le!(order(side, m, n).max(1), lda),
        "A's leading dimension is at least max(1, k)"
    )]
    #[requires(le!(m.max(1), ldb), "B's leading dimension is at least max(1, m)")]
    #[requires(
        le!(extent(order(side, m, n), order(side, m, n), lda), a.len() as u64),
        "a holds the k × k triangle at lda"
    )]
    #[requires(
        le!(extent(m, n, ldb), b.len() as u64),
        "b holds the m × n matrix at ldb"
    )]
    #[track_caller]
    pub fn trsm(
        &mut self,
        side: wwr_blas_side_mode,
        fill: wwr_blas_fill_mode,
        op: wwr_blas_operation,
        diag: wwr_blas_diag_type,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        b: &mut impl DeviceSliceMut<f64>,
        ldb: i32,
    ) -> Status {
        if m == 0 || n == 0 {
            return Ok(());
        }
        let left = side == wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
        let trans = op != OP_N;
        let zero = alpha == 0.0;
        let lower_fill = fill == wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
        // Every case in one frame, M Y = alpha C: on the left M is op(A) and
        // C is B, on the right M is op(A)ᵀ and C is Bᵀ. op(A) is lower for
        // (lower, N) and (upper, T).
        let lower = if left {
            lower_fill != trans
        } else {
            lower_fill == trans
        };
        // M(p, q) is A(q, p) where M is op(A)ᵀ on the right or Aᵀ on the left.
        let a_trans = if left { trans } else { !trans };
        let unit = diag == wwr_blas_diag_type::WWR_BLAS_DIAG_UNIT;
        let a_op = if trans { OP_T } else { OP_N };
        // The contracts hold every count non-negative, so each is its own
        // usize.
        let (block, order, la, lb) = (
            FB_TRSM_TILE as usize,
            order(side, m, n) as usize,
            lda as usize,
            ldb as usize,
        );
        let ncols = if left { n } else { m };
        let blocks = order.div_ceil(block);
        let (pa, pb) = (a.as_ptr(), b.as_mut_ptr());
        for step in 0..blocks {
            let k0 = block * if lower { step } else { blocks - 1 - step };
            let jb = block.min(order - k0);
            // alpha is applied once per row of C: the first diagonal block
            // takes it, and the first gemm scales every other row by its
            // beta. alpha = 0 keeps s = 0 on every block, which zeroes it
            // unread.
            let first = step == 0;
            let scale = if first || zero { alpha } else { 1.0 };
            {
                let _current = self.current();
                // `wrapping_add`, as in `part`: the block at (k0, k0) of A
                // and row k0 of C (row k0 of B on the left, column k0 on the
                // right) are inside the extents the contracts checked.
                let (triangle, tile) = (
                    pa.wrapping_add(k0 + k0 * la),
                    pb.wrapping_add(if left { k0 } else { k0 * lb }),
                );
                // SAFETY: the kernel reads the jb × jb triangle at lda and
                // writes jb rows of C's ncols at ldb (transposed on the
                // right, where ldb >= m = ncols), all inside A's and B's
                // checked extents, which stay valid with B's only user this
                // call until the enqueued work finishes. jb is 1 to the tile;
                // ncols, lda and ldb are the caller's i32s.
                let status = unsafe {
                    fb_trsm_diagonal_f64(
                        self.stream.raw(),
                        jb as i32,
                        ncols,
                        scale,
                        triangle,
                        lda,
                        a_trans,
                        !lower,
                        unit,
                        tile,
                        ldb,
                        !left,
                    )
                };
                launched("fb_trsm_diagonal_f64", status)?;
            }
            // alpha = 0 zeroed this block without reading it, and zeroes the
            // rest block by block the same way: no gemm, which would read A.
            if zero {
                continue;
            }
            // The rows still to come: below the block for a lower M, above
            // it for an upper one.
            let r0 = if lower { k0 + jb } else { 0 };
            let rows = if lower { order - r0 } else { k0 };
            if rows == 0 {
                continue;
            }
            // M(rest, block) in A's storage. On the left it is
            // op(A)(rest, block): A at (rest, block) under N, at (block,
            // rest) transposed under T. On the right it is op(A)(block,
            // rest)ᵀ, so the gemm multiplies by op(A)(block, rest): A at
            // (block, rest) under N, at (rest, block) transposed under T.
            let off = pa.wrapping_add(if left != trans {
                r0 + k0 * la
            } else {
                k0 + r0 * la
            });
            let beta = if first { alpha } else { 1.0 };
            // Each count is at most k, an i32.
            let (jb, rows) = (jb as i32, rows as i32);
            let status = if left {
                // B(rest, :) = beta B(rest, :) - op(A)(rest, block) B(block, :)
                let (solved, rest) = (pb.wrapping_add(k0), pb.wrapping_add(r0));
                // SAFETY: A's block and B's two sets of rows are inside the
                // checked extents; the rows solved and the rows updated are
                // apart, rows of the same columns.
                unsafe {
                    self.gemm_raw(
                        a_op, OP_N, rows, n, jb, -1.0, off, lda, solved, ldb, beta, rest, ldb,
                    )
                }
            } else {
                // B(:, rest) = beta B(:, rest) - B(:, block) op(A)(block, rest)
                let (solved, rest) = (pb.wrapping_add(k0 * lb), pb.wrapping_add(r0 * lb));
                // SAFETY: as on the left, the two sets of B's columns apart.
                unsafe {
                    self.gemm_raw(
                        OP_N, a_op, m, rows, jb, -1.0, solved, ldb, off, lda, beta, rest, ldb,
                    )
                }
            };
            status?;
        }
        Ok(())
    }

    /// The tree's blocked right-looking Cholesky (`src/recipes/cholesky`) on
    /// the GPU's stream, in place of a vendor `potrf`: `A = L Lᵀ` in the
    /// lower triangle of the n × n `a` at leading dimension n, its strict
    /// upper triangle neither read nor written. Each diagonal tile by the
    /// recipes' kernel, which also writes it packed into `tile`
    /// ([`CHOLESKY_TILE`]² doubles of the caller's scratch), the panel below
    /// it by [`Self::trsm`] against that copy, the trailing matrix by
    /// [`Self::syrk`]. `flag` is zeroed here and then holds the 1-based
    /// leading minor that is not positive definite, or 0, on the device;
    /// past one, the factor is not meaningful.
    ///
    /// # Errors
    /// The first launch's or library call's failure; what was enqueued
    /// before it stays enqueued.
    #[requires(
        fits_i32!(n.checked_mul(n)) && le!(n * n, a.len()),
        "a holds the n × n matrix, n² within the calls' i32"
    )]
    #[requires(
        le!(CHOLESKY_TILE * CHOLESKY_TILE, tile.len()),
        "tile holds a whole tile"
    )]
    #[requires(le!(1, flag.len()), "flag holds the flag")]
    #[track_caller]
    pub fn cholesky(
        &mut self,
        n: usize,
        a: &mut impl DeviceSliceMut<f64>,
        tile: &mut impl DeviceSliceMut<f64>,
        flag: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        self.stream.zero(flag)?;
        // n² fits an i32, so every count and offset below does.
        let ni = n as i32;
        let span = |rows: usize, cols: usize| n * (cols - 1) + rows;
        for j in (0..n).step_by(CHOLESKY_TILE) {
            let jb = CHOLESKY_TILE.min(n - j);
            {
                let mut diagonal = part_mut(a, j * (n + 1), span(jb, jb));
                let _current = self.current();
                let (d, t, f) = (diagonal.as_mut_ptr(), tile.as_mut_ptr(), flag.as_mut_ptr());
                // SAFETY: `diagonal` is the jb × jb tile at (j, j) at lda n,
                // `tile` a whole tile and `flag` one int (the cut and the
                // contracts), each its memory's only user until the enqueued
                // work finishes, so the three are apart.
                let status = unsafe {
                    fb_cholesky_diagonal(self.stream.raw(), jb as i32, j as i32, d, ni, t, f)
                };
                launched("fb_cholesky_diagonal", status)?;
            }
            let rest = n - j - jb;
            if rest == 0 {
                break;
            }
            // Columns [0, j + jb) hold the panel, rows [j + jb, n) of
            // columns [j, j + jb); the rest hold the trailing matrix.
            let (mut head, mut tail) = split_at_mut(a, (j + jb) * n);
            let mut panel = part_mut(&mut head, j * n + j + jb, span(rest, jb));
            let mut trailing = part_mut(&mut tail, j + jb, span(rest, rest));
            let l11 = part(tile, 0, jb * jb);
            let (jbi, resti) = (jb as i32, rest as i32);
            // The panel: X L11ᵀ = A21.
            self.trsm(
                wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT,
                wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                OP_T,
                wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
                resti,
                jbi,
                1.0,
                &l11,
                jbi,
                &mut panel,
                ni,
            )?;
            // The trailing matrix: A22 -= L21 L21ᵀ, its lower triangle.
            self.syrk(
                wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                OP_N,
                resti,
                jbi,
                -1.0,
                &panel,
                ni,
                1.0,
                &mut trailing,
                ni,
            )?;
        }
        Ok(())
    }
}

impl Drop for GpuResources {
    /// Drain the stream, destroy the handles, then release the workspace and
    /// the stream; each failure logged, none stopping the rest.
    fn drop(&mut self) {
        let location = Location::caller();
        log_destroy("gpu resources drain", &self.stream.sync(), location);
        let current = ScopedDevice::try_new(self.stream.physical());
        log_destroy("gpu resources destroy", &current, location);
        // SAFETY: created in `new` and destroyed once, here, with nothing in
        // flight.
        let solver = unsafe { wwr_sys::wwr_solver_dn_destroy(self.solver) };
        log_destroy(
            "solver destroy",
            &solver_status("solver destroy", solver),
            location,
        );
        // SAFETY: as above.
        let blas = unsafe { wwr_sys::wwr_blas_destroy(self.blas) };
        log_destroy("blas destroy", &blas_status("blas destroy", blas), location);
    }
}
